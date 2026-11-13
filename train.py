import os
# ---- ★ 可复现性 (Mac M5 / Apple Silicon + Metal 后端) ----
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import model
import common
from read_and_feature import load_data


@nnx.jit
def train_step(
    net: model.HardMonotonicRiskNet,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y_st: jax.Array,
    y_lt: jax.Array,
    weights: Dict[str, float]
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    def loss_fn(m):
        outputs = m(x, training=True)
        loss, metrics = common.compute_loss_hard(outputs, y_st, y_lt, weights)
        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(net)
    optimizer.update(net, grads)
    return loss, metrics


@nnx.jit
def predict_batch(net: model.HardMonotonicRiskNet, x: jax.Array):
    outputs = net(x, training=False)
    return outputs["pred_st"], outputs["pred_lt"], outputs["logit_st"], outputs["logit_lt"]


def evaluate(
    net: model.HardMonotonicRiskNet,
    X: np.ndarray,
    y_st: np.ndarray,
    y_lt: np.ndarray,
    batch_size: int = 8192
) -> Dict[str, float]:
    """
    对输入数据集进行批量推理并计算评估指标:
    - 短期 (MOB2) AUC & KS
    - 长期 (MOB6) AUC & KS
    - 严格硬单调违规率 (P_mob2 > P_mob6 的样本比例)
    """
    preds_st_list, preds_lt_list = [], []
    logits_st_list, logits_lt_list = [], []
    for i in range(0, len(X), batch_size):
        xb = jnp.array(X[i:i + batch_size])
        pst, plt, lst, llt = predict_batch(net, xb)
        preds_st_list.append(np.asarray(pst))
        preds_lt_list.append(np.asarray(plt))
        logits_st_list.append(np.asarray(lst))
        logits_lt_list.append(np.asarray(llt))

    pst = np.concatenate(preds_st_list, axis=0)
    plt = np.concatenate(preds_lt_list, axis=0)
    lst = np.concatenate(logits_st_list, axis=0)
    llt = np.concatenate(logits_lt_list, axis=0)

    auc_st, ks_st = common.compute_metrics_cpu(pst, y_st)
    auc_lt, ks_lt = common.compute_metrics_cpu(plt, y_lt)
    
    loss_task_st = float(jnp.mean(optax.sigmoid_binary_cross_entropy(jnp.array(lst), jnp.array(y_st, dtype=jnp.float32))))
    loss_task_lt = float(jnp.mean(optax.sigmoid_binary_cross_entropy(jnp.array(llt), jnp.array(y_lt, dtype=jnp.float32))))
    loss_task = loss_task_st + loss_task_lt

    # 违规率: 短期风险严格高于长期风险的样本占比
    violations = int(np.sum(pst > plt))
    violation_rate = float(violations / len(pst))

    return {
        "auc_mob2": float(auc_st),
        "ks_mob2": float(ks_st),
        "auc_mob6": float(auc_lt),
        "ks_mob6": float(ks_lt),
        "loss_task": float(loss_task),
        "loss_mob6": float(loss_task_lt),
        "loss_mob2": float(loss_task_st),
        "violation_rate": violation_rate,
        "violations_count": violations,
        "total_samples": len(pst)
    }


def train_single_seed(
    seed: int,
    data: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    w_orth: float = 0.1,
    patience: int = 20,
    verbose: bool = True
) -> Dict[str, float]:
    """
    使用单个指定随机种子训练 HardMonotonicRiskNet 并在测试集上评估
    """
    np.random.seed(seed)
    rngs = nnx.Rngs(seed)

    X_train = data["X_train"]
    y_train_st = data["y_train_mob2"]
    y_train_lt = data["y_train_mob6"]

    X_valid = data["X_valid"]
    y_valid_st = data["y_valid_mob2"]
    y_valid_lt = data["y_valid_mob6"]

    X_test = data["X_test"]
    y_test_st = data["y_test_mob2"]
    y_test_lt = data["y_test_mob6"]

    num_features = X_train.shape[1]
    net = model.HardMonotonicRiskNet(
        rngs=rngs,
        num_static_features=num_features,
        bin_edges=bin_edges,
        static_hidden=128,
        dropout_rate=0.5
    )
    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)
    weights = {"w_orth": w_orth}

    n_samples = len(X_train)
    n_batches = int(np.ceil(n_samples / batch_size))

    best_val_score = 9999
    best_state = None
    best_epoch = 0
    wait = 0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_samples)
        epoch_losses = []
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = jnp.array(X_train[idx])
            ysb = jnp.array(y_train_st[idx], dtype=jnp.float32)
            ylb = jnp.array(y_train_lt[idx], dtype=jnp.float32)
            loss, _ = train_step(net, optimizer, xb, ysb, ylb, weights)
            epoch_losses.append(float(loss))

        val_metrics = evaluate(net, X_valid, y_valid_st, y_valid_lt)
        # 以验证集主任务交叉熵损失最小作为模型最佳权重选取标准
        val_score = val_metrics["loss_task"]
        if val_score < best_val_score:
            best_val_score = val_score
            best_state = jax.tree.map(jnp.array, nnx.state(net))
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose:
            print(f"  [Seed {seed:4d}] Epoch {epoch}/{epochs} - Train Loss: {np.mean(epoch_losses):.4f} | "
                  f"Val Loss: {val_score:.4f} | "
                  f"Val MOB2 AUC={val_metrics['auc_mob2']:.4f}, KS={val_metrics['ks_mob2']:.4f} | "
                  f"Val MOB6 AUC={val_metrics['auc_mob6']:.4f}, KS={val_metrics['ks_mob6']:.4f} | "
                  f"Viol={val_metrics['violation_rate']:.4%}")

    # 载入验证集表现最佳的模型权重
    if best_state is not None:
        nnx.update(net, best_state)

    train_time = time.time() - t_start

    # 在测试集 (OOT) 上进行最终严谨评估
    test_metrics = evaluate(net, X_test, y_test_st, y_test_lt)
    test_metrics["seed"] = seed
    test_metrics["best_epoch"] = best_epoch
    test_metrics["train_time_sec"] = train_time

    if verbose:
        print(f"  >>> [Seed {seed:4d}] 测试集评估结果: "
              f"MOB2 AUC={test_metrics['auc_mob2']:.4f}, KS={test_metrics['ks_mob2']:.4f}, LogLoss={test_metrics['loss_mob2']:.4f} | "
              f"MOB6 AUC={test_metrics['auc_mob6']:.4f}, KS={test_metrics['ks_mob6']:.4f}, LogLoss={test_metrics['loss_mob6']:.4f} | "
              f"违规率={test_metrics['violation_rate']:.4%} ({test_metrics['violations_count']}/{test_metrics['total_samples']}) | "
              f"耗时={train_time:.1f}s")

    return test_metrics


def run_10_seeds_experiment(
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    w_orth: float = 0.1,
    patience: int = 20
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    运行 10 个独立随机种子实验并统计均值与标准差 (与 read_lend.ipynb 配置严格一致)
    """
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999, 200, 300, 400, 500, 600]

    print("=" * 80)
    print("🚀 启动 HardMonotonicRiskNet 10 随机种子基准测试实验")
    print(f"• 种子列表 ({len(seeds)} 个): {seeds}")
    print(f"• 训练超参: epochs={epochs}, batch_size={batch_size}, lr={lr}, w_orth={w_orth}")
    print("=" * 80)

    # 1. 加载数据
    data = load_data()
    X_train = data["X_train"]
    print(f"数据集加载就绪: 训练集={X_train.shape}, 验证集={data['X_valid'].shape}, 测试集={data['X_test'].shape}")

    # 2. 计算分箱边界 (PLE)
    print("正在基于训练集计算分段线性分箱切分点 (PLE Bin Edges)...")
    bin_edges = common.compute_bin_edges(X_train, n_bins=20)

    # 3. 循环 10 个随机种子训练
    results = []
    total_start = time.time()
    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] 正在训练随机种子 Seed={seed} ...")
        res = train_single_seed(
            seed=seed,
            data=data,
            bin_edges=bin_edges,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            w_orth=w_orth,
            patience = patience,
            verbose=True
        )
        results.append(res)

    total_time = time.time() - total_start

    # 4. 整理统计指标
    df_results = pd.DataFrame(results)

    metrics_cols = ["auc_mob2", "ks_mob2", "auc_mob6", "ks_mob6", "loss_mob2", "loss_mob6", "violation_rate"]
    stats_summary = {}
    for col in metrics_cols:
        mean_val = float(df_results[col].mean())
        std_val = float(df_results[col].std())
        stats_summary[col] = {"mean": mean_val, "std": std_val}

    # 5. 打印汇总报表
    print("\n" + "=" * 80)
    print("📊 HardMonotonicRiskNet 10 独立随机种子测试集评测总榜 (Test Set)")
    print("=" * 80)
    print(f"{'Seed':<8}{'MOB2 AUC':<14}{'MOB2 KS':<14}{'MOB2 LogLoss':<14}{'MOB6 AUC':<14}{'MOB6 KS':<14}{'MOB6 LogLoss':<14}{'违规率 (Viol)':<14}{'耗时':<8}")
    print("-" * 80)
    for _, row in df_results.iterrows():
        print(f"{int(row['seed']):<8}"
              f"{row['auc_mob2']:<14.4f}"
              f"{row['ks_mob2']:<14.4f}"
              f"{row['LogLoss mob2']:<14.4f}"
              f"{row['auc_mob6']:<14.4f}"
              f"{row['ks_mob6']:<14.4f}"
              f"{row['LogLoss mob6']:<14.4f}"
              f"{row['violation_rate']:<14.4%}"
              f"{row['train_time_sec']:<6.1f}s")
    print("-" * 80)
    print(f"{'均值 (Mean)':<8}"
          f"{stats_summary['auc_mob2']['mean']:<14.4f}"
          f"{stats_summary['ks_mob2']['mean']:<14.4f}"
          f"{stats_summary['loss_mob2']['mean']:<14.4f}"
          f"{stats_summary['auc_mob6']['mean']:<14.4f}"
          f"{stats_summary['ks_mob6']['mean']:<14.4f}"
          f"{stats_summary['loss_mob6']['mean']:<14.4f}"
          f"{stats_summary['violation_rate']['mean']:<14.4%}")
    print(f"{'标准差 (Std)':<8}"
          f"{stats_summary['auc_mob2']['std']:<14.4f}"
          f"{stats_summary['ks_mob2']['std']:<14.4f}"
          f"{stats_summary['loss_mob2']['std']:<14.4f}"
          f"{stats_summary['auc_mob6']['std']:<14.4f}"
          f"{stats_summary['ks_mob6']['std']:<14.4f}"
          f"{stats_summary['loss_mob6']['std']:<14.4f}"
          f"{stats_summary['violation_rate']['std']:<14.4%}")
    print("=" * 80)
    print(f"✅ 10 种子测试完成，总耗时: {total_time:.1f} 秒\n")

    return df_results, stats_summary


if __name__ == "__main__":
    df_res, stats = run_10_seeds_experiment()
