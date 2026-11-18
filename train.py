import os
# ---- ★ 可复现性 (Mac M5 / Apple Silicon + Metal 后端) ----
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import json
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

import model
import common
from read_and_feature import load_data


# ====================================================================
# 通用评估与汇总格式化函数
# ====================================================================

def compute_standard_eval_metrics(
    p2: np.ndarray,
    p6: np.ndarray,
    y2: np.ndarray,
    y6: np.ndarray,
    compute_rectified: bool = False,
    eps_vr: float = 0.0
) -> Dict[str, float]:
    """
    统一计算风控双目标的核心评估指标:
    - 短期 (MOB2): AUC, KS, LogLoss
    - 长期 (MOB6): AUC, KS, LogLoss
    - 跨期单调违规率 VR: P(MOB6) < P(MOB2) - eps
    - 可选后验规则修正: P(MOB6_rect) = max(P(MOB2), P(MOB6))
    """
    y2_a = np.asarray(y2, dtype=np.float32)
    y6_a = np.asarray(y6, dtype=np.float32)
    p2_a = np.asarray(p2, dtype=np.float32)
    p6_a = np.asarray(p6, dtype=np.float32)

    auc2, ks2 = common.compute_metrics_cpu(p2_a, y2_a)
    auc6, ks6 = common.compute_metrics_cpu(p6_a, y6_a)

    p2_clip = np.clip(p2_a, 1e-15, 1.0 - 1e-15)
    p6_clip = np.clip(p6_a, 1e-15, 1.0 - 1e-15)
    loss2 = float(log_loss(y2_a, p2_clip))
    loss6 = float(log_loss(y6_a, p6_clip))

    # 跨期倒挂违规统计
    is_violation = (p6_a < p2_a - eps_vr)
    n_violations = int(np.sum(is_violation))
    vr_pct = float(np.mean(is_violation)) * 100.0

    metrics: Dict[str, float] = {
        "auc_mob2": float(auc2),
        "ks_mob2": float(ks2),
        "loss_mob2": float(loss2),
        "auc_mob6": float(auc6),
        "ks_mob6": float(ks6),
        "loss_mob6": float(loss6),
        "violation_rate": float(vr_pct / 100.0),
        "vr_pct": float(vr_pct),
        "violations_count": float(n_violations),
        "total_samples": float(len(p6_a)),
    }

    if compute_rectified:
        p6_rect = np.maximum(p2_a, p6_a)
        auc6_rect, ks6_rect = common.compute_metrics_cpu(p6_rect, y6_a)
        p6_rect_clip = np.clip(p6_rect, 1e-15, 1.0 - 1e-15)
        loss6_rect = float(log_loss(y6_a, p6_rect_clip))
        metrics.update({
            "auc_mob6_rect": float(auc6_rect),
            "ks_mob6_rect": float(ks6_rect),
            "loss_mob6_rect": float(loss6_rect),
            "vr_pct_rect": 0.0,
        })

    return metrics


def to_device_arrays(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    预先将训练集、验证集和测试集特征及标签转化为 JAX Device Array 驻留显存，
    消除各批次循环切片时 Host-to-Device 流式通信的内存分配延迟与调度抖动。
    """
    return {
        "X_train": jnp.asarray(data["X_train"], dtype=jnp.float32),
        "y_train_mob2": jnp.asarray(data["y_train_mob2"], dtype=jnp.float32),
        "y_train_mob6": jnp.asarray(data["y_train_mob6"], dtype=jnp.float32),
        "X_valid": jnp.asarray(data["X_valid"], dtype=jnp.float32),
        "y_valid_mob2": jnp.asarray(data["y_valid_mob2"], dtype=jnp.float32),
        "y_valid_mob6": jnp.asarray(data["y_valid_mob6"], dtype=jnp.float32),
        "X_test": jnp.asarray(data["X_test"], dtype=jnp.float32),
        "y_test_mob2": data["y_test_mob2"],
        "y_test_mob6": data["y_test_mob6"],
    }


def compute_stats_summary(df_results: pd.DataFrame, has_rectified: bool = False) -> Dict[str, Dict[str, float]]:
    cols = ["auc_mob2", "ks_mob2", "loss_mob2", "auc_mob6", "ks_mob6", "loss_mob6", "vr_pct"]
    if has_rectified and "auc_mob6_rect" in df_results.columns:
        cols += ["auc_mob6_rect", "ks_mob6_rect", "loss_mob6_rect", "vr_pct_rect"]

    stats: Dict[str, Dict[str, float]] = {}
    for col in cols:
        if col in df_results.columns:
            m = float(df_results[col].mean())
            s = float(df_results[col].std(ddof=1)) if len(df_results) > 1 else 0.0
            stats[col] = {"mean": m, "std": s}
    return stats


def print_experiment_summary(
    model_name: str,
    df_results: pd.DataFrame,
    stats_summary: Dict[str, Dict[str, float]],
    has_rectified: bool = False
):
    print("\n" + "=" * 96)
    print(f"📊 {model_name} 10 独立随机种子测试集评测总榜 (Test Set)")
    print("=" * 96)
    cols_show = ["seed", "best_epoch", "auc_mob2", "ks_mob2", "loss_mob2", "auc_mob6", "ks_mob6", "loss_mob6", "vr_pct"]
    cols_exist = [c for c in cols_show if c in df_results.columns]
    print(df_results[cols_exist].to_string(index=False))

    print("\n" + "-" * 96)
    print(f"🏆 【顶刊论文标准汇报格式 (Mean ± Std, {len(df_results)} Seeds)】: {model_name}")
    print("-" * 96)
    m2 = stats_summary["auc_mob2"]["mean"]
    s2 = stats_summary["auc_mob2"]["std"]
    k2 = stats_summary["ks_mob2"]["mean"]
    sk2 = stats_summary["ks_mob2"]["std"]
    l2 = stats_summary["loss_mob2"]["mean"]
    sl2 = stats_summary["loss_mob2"]["std"]

    m6 = stats_summary["auc_mob6"]["mean"]
    s6 = stats_summary["auc_mob6"]["std"]
    k6 = stats_summary["ks_mob6"]["mean"]
    sk6 = stats_summary["ks_mob6"]["std"]
    l6 = stats_summary["loss_mob6"]["mean"]
    sl6 = stats_summary["loss_mob6"]["std"]

    mvr = stats_summary["vr_pct"]["mean"]
    svr = stats_summary["vr_pct"]["std"]

    print(f"  • MOB2 (短期2月): AUC = {m2:.4f} ± {s2:.4f} | KS = {k2:.4f} ± {sk2:.4f} | LogLoss = {l2:.5f} ± {sl2:.5f}")
    print(f"  • MOB6 (长期6月): AUC = {m6:.4f} ± {s6:.4f} | KS = {k6:.4f} ± {sk6:.4f} | LogLoss = {l6:.5f} ± {sl6:.5f}")
    print(f"  • 跨期倒挂违规率 VR: {mvr:.2f}% ± {svr:.2f}%")

    if has_rectified and "auc_mob6_rect" in stats_summary:
        m6r = stats_summary["auc_mob6_rect"]["mean"]
        s6r = stats_summary["auc_mob6_rect"]["std"]
        k6r = stats_summary["ks_mob6_rect"]["mean"]
        sk6r = stats_summary["ks_mob6_rect"]["std"]
        l6r = stats_summary["loss_mob6_rect"]["mean"]
        sl6r = stats_summary["loss_mob6_rect"]["std"]
        print(f"  • MOB6 (Post-hoc Max 修正后): AUC = {m6r:.4f} ± {s6r:.4f} | KS = {k6r:.4f} ± {sk6r:.4f} | LogLoss = {l6r:.5f} ± {sl6r:.5f} | VR = 0.00%")
    print("=" * 96)


# ====================================================================
# 1. HardMonotonicRiskNet (硬单调因果解耦网络，本研究核心提出模型)
# ====================================================================

@nnx.jit
def train_step_hard(
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
def predict_batch_hard(net: model.HardMonotonicRiskNet, x: jax.Array):
    outputs = net(x, training=False)
    return outputs["pred_st"], outputs["pred_lt"], outputs["logit_st"], outputs["logit_lt"]


def evaluate_hard(
    net: model.HardMonotonicRiskNet,
    X: np.ndarray,
    y_st: np.ndarray,
    y_lt: np.ndarray,
    batch_size: int = 8192
) -> Dict[str, float]:
    preds_st_list, preds_lt_list = [], []
    logits_st_list, logits_lt_list = [], []
    for i in range(0, len(X), batch_size):
        xb = jnp.asarray(X[i:i + batch_size])
        pst, plt, lst, llt = predict_batch_hard(net, xb)
        preds_st_list.append(np.asarray(pst))
        preds_lt_list.append(np.asarray(plt))
        logits_st_list.append(np.asarray(lst))
        logits_lt_list.append(np.asarray(llt))

    pst = np.concatenate(preds_st_list, axis=0)
    plt = np.concatenate(preds_lt_list, axis=0)
    lst = np.concatenate(logits_st_list, axis=0)
    llt = np.concatenate(logits_lt_list, axis=0)

    loss_task_st = float(jnp.mean(optax.sigmoid_binary_cross_entropy(jnp.array(lst), jnp.array(y_st, dtype=jnp.float32))))
    loss_task_lt = float(jnp.mean(optax.sigmoid_binary_cross_entropy(jnp.array(llt), jnp.array(y_lt, dtype=jnp.float32))))
    loss_task = loss_task_st + loss_task_lt

    metrics = compute_standard_eval_metrics(pst, plt, y_st, y_lt, compute_rectified=False)
    metrics["loss_task"] = loss_task
    metrics["loss_task_st"] = loss_task_st
    metrics["loss_task_lt"] = loss_task_lt
    return metrics


def train_single_seed_hard_monotonic(
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
) -> Tuple[Dict[str, float], model.HardMonotonicRiskNet]:
    np.random.seed(seed)
    rngs = nnx.Rngs(seed)

    X_tr_arr = jnp.asarray(data["X_train"], dtype=jnp.float32)
    y_tr_st_arr = jnp.asarray(data["y_train_mob2"], dtype=jnp.float32)
    y_tr_lt_arr = jnp.asarray(data["y_train_mob6"], dtype=jnp.float32)
    X_va_arr = jnp.asarray(data["X_valid"], dtype=jnp.float32)
    y_valid_st = data["y_valid_mob2"]
    y_valid_lt = data["y_valid_mob6"]
    X_te_arr = jnp.asarray(data["X_test"], dtype=jnp.float32)
    y_test_st = data["y_test_mob2"]
    y_test_lt = data["y_test_mob6"]

    net = model.HardMonotonicRiskNet(
        rngs=rngs,
        num_static_features=X_tr_arr.shape[1],
        bin_edges=bin_edges,
        static_hidden=128,
        dropout_rate=0.5
    )
    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)
    weights = {"w_orth": w_orth}

    n_samples = len(X_tr_arr)
    n_batches = int(np.ceil(n_samples / batch_size))

    best_val_score = 9999.0
    best_state = None
    best_epoch = 0
    wait = 0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_samples)
        epoch_losses = []
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = X_tr_arr[idx]
            ysb = y_tr_st_arr[idx]
            ylb = y_tr_lt_arr[idx]
            loss, _ = train_step_hard(net, optimizer, xb, ysb, ylb, weights)
            epoch_losses.append(float(loss))

        val_metrics = evaluate_hard(net, X_va_arr, y_valid_st, y_valid_lt)
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

    if best_state is not None:
        nnx.update(net, best_state)

    train_time = time.time() - t_start
    test_metrics = evaluate_hard(net, X_te_arr, y_test_st, y_test_lt)
    test_metrics["seed"] = seed
    test_metrics["best_epoch"] = best_epoch
    test_metrics["train_time_sec"] = train_time

    if verbose:
        print(f"  ✓ [Seed {seed:4d}] 完成 (耗时 {train_time:.1f}s, Best Epoch {best_epoch}): "
              f"MOB6 AUC={test_metrics['auc_mob6']:.4f}, KS={test_metrics['ks_mob6']:.4f} | "
              f"MOB2 AUC={test_metrics['auc_mob2']:.4f}, KS={test_metrics['ks_mob2']:.4f} | "
              f"VR={test_metrics['vr_pct']:.2f}%")

    return test_metrics, net


def run_10_seeds_hard_monotonic(
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    w_orth: float = 0.1,
    patience: int = 20,
    return_best_model: bool = False
) -> Union[Tuple[pd.DataFrame, Dict[str, Dict[str, float]]], Tuple[pd.DataFrame, Dict[str, Dict[str, float]], model.HardMonotonicRiskNet, Dict[str, np.ndarray]]]:
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999, 200, 300, 400, 500, 600]

    print("=" * 90)
    print(f"🚀 开始 HardMonotonicRiskNet (硬单调因果解耦网络) 训练评估 (Seeds: {seeds})")
    print("=" * 90)

    data = load_data()
    bin_edges = common.compute_bin_edges(data["X_train"], n_bins=20)
    dev_data = to_device_arrays(data)

    results = []
    best_model = None
    seed_42_model = None
    best_loss = 9999.0
    t_start = time.time()

    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] 正在运行 Seed = {seed} ...")
        res, trained_net = train_single_seed_hard_monotonic(
            seed=seed,
            data=dev_data,
            bin_edges=bin_edges,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            w_orth=w_orth,
            patience=patience,
            verbose=True
        )
        results.append(res)
        if seed == 42:
            seed_42_model = trained_net
        if best_model is None or res.get("loss_task", 9999.0) < best_loss:
            best_loss = res.get("loss_task", 9999.0)
            best_model = trained_net

    df_results = pd.DataFrame(results)
    stats_summary = compute_stats_summary(df_results, has_rectified=False)
    print_experiment_summary("HardMonotonicRiskNet", df_results, stats_summary, has_rectified=False)

    selected_model = seed_42_model if seed_42_model is not None else best_model
    if return_best_model:
        return df_results, stats_summary, selected_model, data
    return df_results, stats_summary


# 向后兼容接口
run_10_seeds_experiment = run_10_seeds_hard_monotonic
train_single_seed = train_single_seed_hard_monotonic
evaluate = evaluate_hard


# ====================================================================
# 2. BaseMTL (标准共享底层多任务网络, Caruana 1997)
# ====================================================================

@nnx.jit
def train_step_base_mtl(model_net: model.BaseMTL, optimizer: nnx.Optimizer, batch_x: jax.Array, y2: jax.Array, y6: jax.Array, w_reg: float = 0.0):
    def loss_fn(m):
        logits_mob2, logits_mob6 = m(batch_x, training=True)
        loss6 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_mob6.squeeze(-1), y6))
        loss2 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_mob2.squeeze(-1), y2))
        pred_st = jax.nn.sigmoid(logits_mob2)
        pred_lt = jax.nn.sigmoid(logits_mob6)
        reg_loss = jnp.mean(jnp.maximum(0.0, pred_st - pred_lt))
        return loss6 + loss2 + w_reg * reg_loss

    loss, grads = nnx.value_and_grad(loss_fn)(model_net)
    optimizer.update(model_net, grads)
    return loss


@nnx.jit
def eval_forward_base_mtl(model_net: model.BaseMTL, batch_x: jax.Array):
    logits_mob2, logits_mob6 = model_net(batch_x, training=False)
    p_mob2 = jax.nn.sigmoid(logits_mob2).squeeze(-1)
    p_mob6 = jax.nn.sigmoid(logits_mob6).squeeze(-1)
    return p_mob2, p_mob6


def evaluate_base_mtl(model_net: model.BaseMTL, X: np.ndarray, y2: np.ndarray, y6: np.ndarray, eval_bs: int = 1024) -> Tuple[float, Dict[str, float]]:
    N = len(y6)
    p2_list, p6_list = [], []
    for s in range(0, N, eval_bs):
        e = min(s + eval_bs, N)
        p2_b, p6_b = eval_forward_base_mtl(model_net, X[s:e])
        p2_list.append(np.asarray(p2_b))
        p6_list.append(np.asarray(p6_b))

    p2 = np.concatenate(p2_list)
    p6 = np.concatenate(p6_list)
    metrics = compute_standard_eval_metrics(p2, p6, y2, y6, compute_rectified=False)
    total_loss = metrics["loss_mob2"] + metrics["loss_mob6"]
    return total_loss, metrics


def train_single_seed_base_mtl(
    seed: int,
    data: Dict[str, np.ndarray],
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    w_reg: float = 5.0,
    verbose: bool = True
) -> Tuple[Dict[str, float], model.BaseMTL]:
    np.random.seed(seed)
    rngs = nnx.Rngs(seed)

    X_tr_arr = jnp.asarray(data["X_train"], dtype=jnp.float32)
    y_train_2 = jnp.asarray(data["y_train_mob2"], dtype=jnp.float32)
    y_train_6 = jnp.asarray(data["y_train_mob6"], dtype=jnp.float32)
    X_va_arr = jnp.asarray(data["X_valid"], dtype=jnp.float32)
    y_valid_2 = jnp.asarray(data["y_valid_mob2"], dtype=jnp.float32)
    y_valid_6 = jnp.asarray(data["y_valid_mob6"], dtype=jnp.float32)
    X_te_arr = jnp.asarray(data["X_test"], dtype=jnp.float32)
    y_test_2 = data["y_test_mob2"]
    y_test_6 = data["y_test_mob6"]

    net = model.BaseMTL(
        rngs=rngs,
        num_static_features=X_tr_arr.shape[1],
        shared_hidden=128,
        head_hidden=64,
        dropout_rate=0.5
    )
    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)

    N_train = len(y_train_6)
    best_val_loss = 9999.0
    best_state = None
    best_epoch = 0
    wait = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N_train)
        for s in range(0, N_train, batch_size):
            e = min(s + batch_size, N_train)
            idx = perm[s:e]
            train_step_base_mtl(net, optimizer, X_tr_arr[idx], y_train_2[idx], y_train_6[idx], w_reg=w_reg)

        val_loss, _ = evaluate_base_mtl(net, X_va_arr, y_valid_2, y_valid_6)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = jax.tree.map(jnp.array, nnx.state(net))
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        nnx.update(net, best_state)

    train_time = time.time() - t_start
    _, test_metrics = evaluate_base_mtl(net, X_te_arr, y_test_2, y_test_6)
    test_metrics["seed"] = seed
    test_metrics["best_epoch"] = best_epoch
    test_metrics["train_time_sec"] = train_time

    if verbose:
        print(f"  ✓ [Seed {seed:4d}] 完成 (耗时 {train_time:.1f}s, Best Epoch {best_epoch}): "
              f"MOB6 AUC={test_metrics['auc_mob6']:.4f}, KS={test_metrics['ks_mob6']:.4f} | "
              f"MOB2 AUC={test_metrics['auc_mob2']:.4f}, KS={test_metrics['ks_mob2']:.4f} | "
              f"VR={test_metrics['vr_pct']:.2f}%")

    return test_metrics, net


def run_10_seeds_base_mtl(
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    w_reg: float = 5.0,
    return_best_model: bool = False
) -> Union[Tuple[pd.DataFrame, Dict[str, Dict[str, float]]], Tuple[pd.DataFrame, Dict[str, Dict[str, float]], model.BaseMTL, Dict[str, np.ndarray]]]:
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999, 200, 300, 400, 500, 600]

    print("=" * 90)
    print(f"🚀 开始 BaseMTL (Caruana 1997 / Shared-Bottom) 训练评估 (Seeds: {seeds})")
    print("=" * 90)

    data = load_data()
    dev_data = to_device_arrays(data)
    results = []
    best_model = None
    best_loss = 9999.0

    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] 正在运行 Seed = {seed} ...")
        res, trained_net = train_single_seed_base_mtl(
            seed=seed, data=dev_data, epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, w_reg=w_reg, verbose=True
        )
        results.append(res)
        cur_loss = res["loss_mob2"] + res["loss_mob6"]
        if best_model is None or cur_loss < best_loss:
            best_loss = cur_loss
            best_model = trained_net

    df_results = pd.DataFrame(results)
    stats_summary = compute_stats_summary(df_results, has_rectified=False)
    print_experiment_summary("BaseMTL", df_results, stats_summary, has_rectified=False)

    if return_best_model:
        return df_results, stats_summary, best_model, data
    return df_results, stats_summary


# ====================================================================
# 3. HybridCreditModel_STL (独立双单任务 MLP + Post-hoc 修正)
# ====================================================================

@nnx.jit
def train_step_stl_weighted(model_net: model.HybridCreditModel_STL, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array, pos_weight: float = 1.0):
    def loss_fn(m):
        logits = m(x, training=True).squeeze(-1)
        bce = optax.sigmoid_binary_cross_entropy(logits, y)
        weights = jnp.where(y == 1.0, pos_weight, 1.0)
        return jnp.mean(weights * bce)

    loss, grads = nnx.value_and_grad(loss_fn)(model_net)
    optimizer.update(model_net, grads)
    return loss


def predict_proba_stl(model_net: model.HybridCreditModel_STL, X: Union[np.ndarray, jax.Array], batch_size: int = 2048) -> np.ndarray:
    N = len(X)
    preds = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        logits = model_net(X[s:e], training=False)
        p = jax.nn.sigmoid(logits).squeeze(-1)
        preds.append(np.asarray(p))
    return np.concatenate(preds)


def eval_loss_stl(model_net: model.HybridCreditModel_STL, X: Union[np.ndarray, jax.Array], y: np.ndarray, batch_size: int = 1024) -> float:
    p = predict_proba_stl(model_net, X, batch_size=batch_size)
    p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(log_loss(y, p_clip))


def train_single_mlp_stl(
    task_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    pos_weight: float = 1.0
) -> Tuple[model.HybridCreditModel_STL, int]:
    np.random.seed(seed)
    rngs = nnx.Rngs(seed)
    net = model.HybridCreditModel_STL(
        rngs=rngs,
        num_static_features=X_train.shape[1],
        embed_hidden=128,
        dropout_rate=0.5
    )
    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)

    X_tr_arr = jnp.asarray(X_train, dtype=jnp.float32)
    y_train_jax = jnp.asarray(y_train, dtype=jnp.float32)
    X_va_arr = jnp.asarray(X_valid, dtype=jnp.float32)

    N_train = len(y_train)
    best_val_loss = 9999.0
    best_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N_train)
        for s in range(0, N_train, batch_size):
            e = min(s + batch_size, N_train)
            idx = perm[s:e]
            train_step_stl_weighted(net, optimizer, X_tr_arr[idx], y_train_jax[idx], pos_weight=pos_weight)

        val_loss = eval_loss_stl(net, X_va_arr, y_valid, batch_size=1024)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = jax.tree.map(jnp.array, nnx.state(net))
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        nnx.update(net, best_state)

    return net, best_epoch


def train_single_seed_mlp_stl(
    seed: int,
    data: Dict[str, np.ndarray],
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    verbose: bool = True
) -> Tuple[Dict[str, float], Tuple[model.HybridCreditModel_STL, model.HybridCreditModel_STL]]:
    t_start = time.time()
    X_train = data["X_train"]
    X_valid = data["X_valid"]
    X_te_arr = jnp.asarray(data["X_test"], dtype=jnp.float32)

    # 1. 独立训练 MOB2 模型 (依据 notebook 设置 pos_weight=20.0)
    net_mob2, best_ep_2 = train_single_mlp_stl(
        "MOB2", X_train, data["y_train_mob2"], X_valid, data["y_valid_mob2"],
        seed=seed, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        patience=patience, pos_weight=20.0
    )
    p2_test = predict_proba_stl(net_mob2, X_te_arr)

    # 2. 独立训练 MOB6 模型 (依据 notebook 设置 pos_weight=8.0)
    net_mob6, best_ep_6 = train_single_mlp_stl(
        "MOB6", X_train, data["y_train_mob6"], X_valid, data["y_valid_mob6"],
        seed=seed, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        patience=patience, pos_weight=8.0
    )
    p6_test = predict_proba_stl(net_mob6, X_te_arr)

    # 3. 计算指标 (含 raw 与 post-hoc 修正)
    metrics = compute_standard_eval_metrics(p2_test, p6_test, data["y_test_mob2"], data["y_test_mob6"], compute_rectified=True)
    train_time = time.time() - t_start
    metrics["seed"] = seed
    metrics["best_ep_mob2"] = best_ep_2
    metrics["best_ep_mob6"] = best_ep_6
    metrics["best_epoch"] = max(best_ep_2, best_ep_6)
    metrics["train_time_sec"] = train_time

    if verbose:
        print(f"  ✓ [Seed {seed:4d}] 完成 (耗时 {train_time:.1f}s): "
              f"MOB6 AUC={metrics['auc_mob6']:.4f} (修: {metrics['auc_mob6_rect']:.4f}) | "
              f"MOB2 AUC={metrics['auc_mob2']:.4f} | "
              f"VR={metrics['vr_pct']:.2f}%")

    return metrics, (net_mob2, net_mob6)


def run_10_seeds_mlp_stl(
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    return_best_model: bool = False
) -> Union[Tuple[pd.DataFrame, Dict[str, Dict[str, float]]], Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Any, Dict[str, np.ndarray]]]:
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999, 200, 300, 400, 500, 600]

    print("=" * 90)
    print(f"🚀 开始 MLP (STL 双期独立单任务网络) 训练评估 (Seeds: {seeds})")
    print("=" * 90)

    data = load_data()
    dev_data = to_device_arrays(data)
    results = []
    best_models = None
    best_loss = 9999.0

    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] 正在运行 Seed = {seed} ...")
        res, trained_nets = train_single_seed_mlp_stl(
            seed=seed, data=dev_data, epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, verbose=True
        )
        results.append(res)
        cur_loss = res["loss_mob2"] + res["loss_mob6"]
        if best_models is None or cur_loss < best_loss:
            best_loss = cur_loss
            best_models = trained_nets

    df_results = pd.DataFrame(results)
    stats_summary = compute_stats_summary(df_results, has_rectified=True)
    print_experiment_summary("MLP (STL)", df_results, stats_summary, has_rectified=True)

    if return_best_model:
        return df_results, stats_summary, best_models, data
    return df_results, stats_summary


# ====================================================================
# 4. HistGBDT (双期独立 GBDT 模型、超参数调优 + Post-hoc 修正)
# ====================================================================

# 默认最优参数 (严格来源于 read_lend.ipynb Cell 18/21 100-iter 随机搜索最优配置)
HGBDT_PARAMS_MOB2 = {
    "l2_regularization": 11.1559,
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_features": 0.6,
    "max_iter": 500,
    "min_samples_leaf": 10,
    "n_iter_no_change": 20
}

HGBDT_PARAMS_MOB6 = {
    "l2_regularization": 84.772,
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_features": 0.8,
    "max_iter": 500,
    "min_samples_leaf": 100,
    "n_iter_no_change": 20
}


def load_saved_hgbdt_params(verbose: bool = False) -> bool:
    """
    若本地存在持久化调优参数文件 hgbdt_best_params.json，则自动热载入
    """
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hgbdt_best_params.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                loaded_keys = []
                if "hgbdt_best_params_mob2" in saved:
                    HGBDT_PARAMS_MOB2.update(saved["hgbdt_best_params_mob2"])
                    loaded_keys.append("MOB2")
                if "hgbdt_best_params_mob6" in saved:
                    HGBDT_PARAMS_MOB6.update(saved["hgbdt_best_params_mob6"])
                    loaded_keys.append("MOB6")
                if verbose:
                    print(f"📄 已成功从配置文件加载 HistGBDT 最优超参数: {json_path} (目标: {loaded_keys})")
                return True
        except Exception as e:
            if verbose:
                print(f"⚠️ 读取 {json_path} 异常: {e}，将使用内置默认调优参数")
            pass
    elif verbose:
        print(f"ℹ️ 未发现持久化配置文件 {json_path}，使用内置默认调优参数")
    return False

load_saved_hgbdt_params(verbose=False)


class RoundedLogUniform:
    """对数均匀分布采样辅助类 (RandomizedSearchCV 专用)"""
    def __init__(self, a: float, b: float, decimals: int = 4):
        self.dist = loguniform(a, b)
        self.decimals = decimals

    def rvs(self, size: Optional[int] = None, random_state: Optional[int] = None):
        vals = self.dist.rvs(size=size, random_state=random_state)
        return np.round(vals, self.decimals)


class HistGBDTWithValSet(HistGradientBoostingClassifier):
    """
    带验证集注入的 HistGradientBoostingClassifier 包装类:
    在 RandomizedSearchCV 每次采样评估时，自动将验证集注入底层 fit(..., X_val=..., y_val=...)，
    使得超参数搜索阶段各候选模型均严格执行验证集早停截断 (Early Stopping)。
    ★ 具备跨进程支持: 通过系统级共享文件向 loky 多进程并发 worker 安全同步验证集数据。
    """
    _val_X = None
    _val_y = None

    @classmethod
    def get_val_data(cls):
        if cls._val_X is not None and cls._val_y is not None:
            return cls._val_X, cls._val_y
        cache_path = os.environ.get("_HGBDT_VAL_CACHE")
        if cache_path and os.path.exists(cache_path):
            try:
                loaded = np.load(cache_path)
                cls._val_X = loaded["X_val"]
                cls._val_y = loaded["y_val"]
                return cls._val_X, cls._val_y
            except Exception:
                pass
        return None, None

    def __init__(
        self,
        *,
        loss='log_loss',
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features='from_dtype',
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping='auto',
        scoring='loss',
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-07,
        verbose=0,
        random_state=None,
        class_weight=None
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            class_weight=class_weight,
        )

    def fit(self, X, y, sample_weight=None):
        val_x, val_y = self.get_val_data()
        if val_x is not None and val_y is not None:
            return super().fit(
                X, y,
                sample_weight=sample_weight,
                X_val=val_x,
                y_val=val_y
            )
        return super().fit(X, y, sample_weight=sample_weight)


def tune_single_hist_gbdt_task(
    task_name: str,
    data: Optional[Dict[str, np.ndarray]] = None,
    n_iter: int = 100,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    针对单个期别任务 (MOB2 或 MOB6) 执行 HistGBDT 随机搜索超参数调优 (验证集早停截断)
    """
    task = task_name.upper()
    if data is None:
        data = load_data()

    X_train = data["X_train"]
    X_valid = data["X_valid"]
    X_test = data["X_test"]

    if task == "MOB2":
        y_train = data["y_train_mob2"]
        y_valid = data["y_valid_mob2"]
        y_test = data["y_test_mob2"]
    elif task == "MOB6":
        y_train = data["y_train_mob6"]
        y_valid = data["y_valid_mob6"]
        y_test = data["y_test_mob6"]
    else:
        raise ValueError(f"未知任务: {task_name}，仅支持 'MOB2' 或 'MOB6'")

    print("\n" + "=" * 85)
    print(f"🎯 开始 HistGBDT 超参数随机搜索调优 [{task}] (n_iter={n_iter}, seed={seed})")
    print(f"• 样本规模: 训练集={len(X_train):,}, 验证集={len(X_valid):,}, 测试集={len(X_test):,}")
    print("=" * 85)

    # 1. 采用 PredefinedSplit 固定验证集策略 (训练集为 -1，验证集为 0)
    X_all = np.concatenate([X_train, X_valid], axis=0)
    y_all = np.concatenate([y_train, y_valid], axis=0)
    test_fold = np.concatenate([
        -np.ones(len(X_train), dtype=np.int32),
        np.zeros(len(X_valid), dtype=np.int32)
    ])
    ps = PredefinedSplit(test_fold)

    # 2. 注入外部验证集供 wrapper 内部做早停 (通过临时缓存文件支持多进程 loky 安全共享)
    HistGBDTWithValSet._val_X = X_valid
    HistGBDTWithValSet._val_y = y_valid
    val_cache_file = os.path.join(tempfile.gettempdir(), f"hgbdt_val_cache_{task}_{os.getpid()}.npz")
    np.savez(val_cache_file, X_val=X_valid, y_val=y_valid)
    os.environ["_HGBDT_VAL_CACHE"] = val_cache_file

    # 3. 参数搜索空间 (严格复现 read_lend.ipynb Cell 18/21 配置)
    hgbdt_param_dist = {
        'max_iter':           [500],
        'learning_rate':      [0.01, 0.05],
        'max_depth':          [3, 4, 5, 6, 7, 8],
        'min_samples_leaf':   [10, 20, 50, 100],
        'l2_regularization':  RoundedLogUniform(1e-3, 100.0, decimals=4),
        'max_features':       [0.6, 0.7, 0.8, 0.9, 1.0],
        'n_iter_no_change':   [20],
    }

    base_estimator = HistGBDTWithValSet(
        random_state=seed,
        early_stopping=True,
        validation_fraction=None,
    )

    random_search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=hgbdt_param_dist,
        n_iter=n_iter,
        scoring='neg_log_loss',
        cv=ps,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=seed,
        refit=False,
    )

    t0 = time.time()
    try:
        random_search.fit(X_all, y_all)
    finally:
        # 4. 清理类变量与临时共享缓存，防止内存泄露
        HistGBDTWithValSet._val_X = None
        HistGBDTWithValSet._val_y = None
        if os.environ.get("_HGBDT_VAL_CACHE") == val_cache_file:
            os.environ.pop("_HGBDT_VAL_CACHE", None)
        if os.path.exists(val_cache_file):
            try:
                os.remove(val_cache_file)
            except Exception:
                pass
    search_time = time.time() - t0

    best_params = random_search.best_params_.copy()
    best_loss = -random_search.best_score_

    print("\n" + "-" * 85)
    print(f"✅ [{task}] 超参数搜索完成 (耗时 {search_time:.1f}s, 最优验证集 LogLoss={best_loss:.5f})")
    print("🏆 最优超参数组合:")
    for k, v in best_params.items():
        print(f"   • {k}: {v}")

    # Top-10 候选超参评估表
    results_df = pd.DataFrame(random_search.cv_results_).sort_values('rank_test_score')
    cols_show = [
        'rank_test_score', 'mean_test_score', 'param_learning_rate',
        'param_max_depth', 'param_min_samples_leaf', 'param_l2_regularization', 'param_max_features'
    ]
    cols_exist = [c for c in cols_show if c in results_df.columns]
    print(f"\n📊 [{task}] Top-10 候选超参评估表:")
    print(results_df[cols_exist].head(10).to_string(index=False))

    # 5. 用最优参数在全训练集重新训练并在三集上验证
    best_gbdt = HistGradientBoostingClassifier(
        random_state=seed,
        **best_params,
        early_stopping=True,
        validation_fraction=None,
    )
    best_gbdt.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)

    p_tr = best_gbdt.predict_proba(X_train)[:, 1]
    p_va = best_gbdt.predict_proba(X_valid)[:, 1]
    p_te = best_gbdt.predict_proba(X_test)[:, 1]

    auc_tr, ks_tr = common.compute_metrics_cpu(p_tr, y_train)
    auc_va, ks_va = common.compute_metrics_cpu(p_va, y_valid)
    auc_te, ks_te = common.compute_metrics_cpu(p_te, y_test)

    loss_tr = float(log_loss(y_train, np.clip(p_tr, 1e-15, 1 - 1e-15)))
    loss_va = float(log_loss(y_valid, np.clip(p_va, 1e-15, 1 - 1e-15)))
    loss_te = float(log_loss(y_test, np.clip(p_te, 1e-15, 1 - 1e-15)))

    print(f"\n📈 [{task}] 最优模型三集综合评估 (Actual Iterations = {best_gbdt.n_iter_}):")
    print(f"{'数据集':<12} {'AUC':>10} {'KS':>10} {'LogLoss':>12}")
    print(f"{'-'*48}")
    print(f"{'训练集':<12} {auc_tr:>10.4f} {ks_tr:>10.4f} {loss_tr:>12.5f}")
    print(f"{'验证集':<12} {auc_va:>10.4f} {ks_va:>10.4f} {loss_va:>12.5f}")
    print(f"{'测试集':<12} {auc_te:>10.4f} {ks_te:>10.4f} {loss_te:>12.5f}")
    print("=" * 85 + "\n")

    return best_params


def tune_hist_gbdt(
    task: str = "both",
    n_iter: int = 100,
    seed: int = 42,
    n_jobs: int = -1,
    save_json: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    HistGBDT 自动化超参数调优入口函数 (支持 MOB2、MOB6 或两者全部寻优并自动持久化)
    """
    data = load_data()
    best_results = {}
    task_lower = task.lower().strip()

    if task_lower in ["mob2", "both"]:
        params_mob2 = tune_single_hist_gbdt_task("MOB2", data=data, n_iter=n_iter, seed=seed, n_jobs=n_jobs)
        best_results["hgbdt_best_params_mob2"] = params_mob2
        HGBDT_PARAMS_MOB2.update(params_mob2)

    if task_lower in ["mob6", "both"]:
        params_mob6 = tune_single_hist_gbdt_task("MOB6", data=data, n_iter=n_iter, seed=seed, n_jobs=n_jobs)
        best_results["hgbdt_best_params_mob6"] = params_mob6
        HGBDT_PARAMS_MOB6.update(params_mob6)

    if save_json:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hgbdt_best_params.json")
        to_save = {
            "hgbdt_best_params_mob2": HGBDT_PARAMS_MOB2,
            "hgbdt_best_params_mob6": HGBDT_PARAMS_MOB6,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=4, ensure_ascii=False)
        print(f"💾 最优超参数已持久化保存至: {json_path}")

    return best_results


def train_single_seed_hist_gbdt(
    seed: int,
    data: Dict[str, np.ndarray],
    params_mob2: Optional[Dict[str, Any]] = None,
    params_mob6: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[Dict[str, float], Tuple[HistGradientBoostingClassifier, HistGradientBoostingClassifier]]:
    t_start = time.time()
    X_train = data["X_train"]
    X_valid = data["X_valid"]
    X_test = data["X_test"]
    y_tr_2 = data["y_train_mob2"]
    y_va_2 = data["y_valid_mob2"]
    y_te_2 = data["y_test_mob2"]
    y_tr_6 = data["y_train_mob6"]
    y_va_6 = data["y_valid_mob6"]
    y_te_6 = data["y_test_mob6"]

    cfg_mob2 = params_mob2 if params_mob2 is not None else HGBDT_PARAMS_MOB2
    cfg_mob6 = params_mob6 if params_mob6 is not None else HGBDT_PARAMS_MOB6

    # ---- A. 训练 MOB2 模型 (验证集早停) ----
    model_mob2 = HistGradientBoostingClassifier(
        random_state=seed,
        **cfg_mob2,
        early_stopping=True,
        validation_fraction=None,
    )
    model_mob2.fit(X_train, y_tr_2, X_val=X_valid, y_val=y_va_2)
    p2_test = model_mob2.predict_proba(X_test)[:, 1]

    # ---- B. 训练 MOB6 模型 (验证集早停) ----
    model_mob6 = HistGradientBoostingClassifier(
        random_state=seed,
        **cfg_mob6,
        early_stopping=True,
        validation_fraction=None,
    )
    model_mob6.fit(X_train, y_tr_6, X_val=X_valid, y_val=y_va_6)
    p6_test = model_mob6.predict_proba(X_test)[:, 1]

    # ---- C. 计算指标 (含 raw 与 post-hoc 修正) ----
    metrics = compute_standard_eval_metrics(p2_test, p6_test, y_te_2, y_te_6, compute_rectified=True)
    train_time = time.time() - t_start
    metrics["seed"] = seed
    metrics["iter_mob2"] = model_mob2.n_iter_
    metrics["iter_mob6"] = model_mob6.n_iter_
    metrics["best_epoch"] = max(model_mob2.n_iter_, model_mob6.n_iter_)
    metrics["train_time_sec"] = train_time

    if verbose:
        print(f"  ✓ [Seed {seed:4d}] 完成 (耗时 {train_time:.1f}s, Iter2={model_mob2.n_iter_}, Iter6={model_mob6.n_iter_}): "
              f"MOB6 AUC={metrics['auc_mob6']:.4f} (修: {metrics['auc_mob6_rect']:.4f}) | "
              f"MOB2 AUC={metrics['auc_mob2']:.4f} | "
              f"VR={metrics['vr_pct']:.2f}%")

    return metrics, (model_mob2, model_mob6)


def run_10_seeds_hist_gbdt(
    seeds: Optional[List[int]] = None,
    params_mob2: Optional[Dict[str, Any]] = None,
    params_mob6: Optional[Dict[str, Any]] = None,
    return_best_model: bool = False
) -> Union[Tuple[pd.DataFrame, Dict[str, Dict[str, float]]], Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Any, Dict[str, np.ndarray]]]:
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999, 200, 300, 400, 500, 600]

    print("=" * 90)
    print(f"🚀 开始 HistGBDT (双期独立梯度提升树) 训练评估 (Seeds: {seeds})")
    if params_mob2 is None and params_mob6 is None:
        load_saved_hgbdt_params(verbose=True)
    cfg_mob2 = params_mob2 if params_mob2 is not None else HGBDT_PARAMS_MOB2
    cfg_mob6 = params_mob6 if params_mob6 is not None else HGBDT_PARAMS_MOB6
    print(f"• MOB2 训练超参数: {cfg_mob2}")
    print(f"• MOB6 训练超参数: {cfg_mob6}")
    print("=" * 90)

    data = load_data()
    results = []
    best_models = None
    best_loss = 9999.0

    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] 正在运行 Seed = {seed} ...")
        res, trained_nets = train_single_seed_hist_gbdt(
            seed=seed, data=data, params_mob2=params_mob2, params_mob6=params_mob6, verbose=True
        )
        results.append(res)
        cur_loss = res["loss_mob2"] + res["loss_mob6"]
        if best_models is None or cur_loss < best_loss:
            best_loss = cur_loss
            best_models = trained_nets

    df_results = pd.DataFrame(results)
    stats_summary = compute_stats_summary(df_results, has_rectified=True)
    print_experiment_summary("HistGBDT", df_results, stats_summary, has_rectified=True)

    if return_best_model:
        return df_results, stats_summary, best_models, data
    return df_results, stats_summary


# ====================================================================
# 5. NnetSurvival (离散时间生存模型, Gensheimer et al. PeerJ 2019)
# ====================================================================

@nnx.jit
def train_step_surv(net: model.NnetSurvival, optimizer: nnx.Optimizer, batch_x: jax.Array, y2: jax.Array, y6: jax.Array):
    def loss_fn(m):
        logit_h1, logit_h2, _, _ = m(batch_x, training=True)
        loss_h1 = optax.sigmoid_binary_cross_entropy(logit_h1, y2)
        loss_h2 = optax.sigmoid_binary_cross_entropy(logit_h2, y6)
        surv_mask = 1.0 - y2
        loss = jnp.mean(loss_h1 + surv_mask * loss_h2)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(net)
    optimizer.update(net, grads)
    return loss


@nnx.jit
def eval_forward_surv(net: model.NnetSurvival, batch_x: jax.Array, y2: jax.Array, y6: jax.Array):
    logit_h1, logit_h2, p_mob2, p_mob6 = net(batch_x, training=False)
    loss_h1 = optax.sigmoid_binary_cross_entropy(logit_h1, y2)
    loss_h2 = optax.sigmoid_binary_cross_entropy(logit_h2, y6)
    surv_mask = 1.0 - y2
    total_loss = jnp.mean(loss_h1 + surv_mask * loss_h2)
    return total_loss, p_mob2, p_mob6


def evaluate_surv_loss(net: model.NnetSurvival, X: Union[np.ndarray, jax.Array], y2: jax.Array, y6: jax.Array, eval_bs: int = 1024) -> float:
    N = len(y6)
    total_loss_sum = 0.0
    for s in range(0, N, eval_bs):
        e = min(s + eval_bs, N)
        b_loss, _, _ = eval_forward_surv(net, X[s:e], y2[s:e], y6[s:e])
        total_loss_sum += float(b_loss) * (e - s)
    return total_loss_sum / N


def evaluate_surv_test(net: model.NnetSurvival, X: Union[np.ndarray, jax.Array], y2: np.ndarray, y6: np.ndarray, eval_bs: int = 1024) -> Dict[str, float]:
    N = len(y6)
    p2_all, p6_all = [], []
    for s in range(0, N, eval_bs):
        e = min(s + eval_bs, N)
        _, p2, p6 = eval_forward_surv(net, X[s:e], jnp.asarray(y2[s:e]), jnp.asarray(y6[s:e]))
        p2_all.append(np.asarray(p2.reshape(-1)))
        p6_all.append(np.asarray(p6.reshape(-1)))

    p2 = np.concatenate(p2_all)
    p6 = np.concatenate(p6_all)
    # 生存模型具有理论保单调性，使用 1e-6 滤除浮点精度误差
    return compute_standard_eval_metrics(p2, p6, y2, y6, compute_rectified=False, eps_vr=1e-6)


def train_single_seed_nnet_survival(
    seed: int,
    data: Dict[str, np.ndarray],
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    restore_best: bool = False,
    verbose: bool = True
) -> Tuple[Dict[str, float], model.NnetSurvival]:
    np.random.seed(seed)
    rngs = nnx.Rngs(seed)

    X_tr_arr = jnp.asarray(data["X_train"], dtype=jnp.float32)
    y_train_2 = jnp.asarray(data["y_train_mob2"], dtype=jnp.float32)
    y_train_6 = jnp.asarray(data["y_train_mob6"], dtype=jnp.float32)
    X_va_arr = jnp.asarray(data["X_valid"], dtype=jnp.float32)
    y_valid_2 = jnp.asarray(data["y_valid_mob2"], dtype=jnp.float32)
    y_valid_6 = jnp.asarray(data["y_valid_mob6"], dtype=jnp.float32)
    X_te_arr = jnp.asarray(data["X_test"], dtype=jnp.float32)
    y_test_2 = data["y_test_mob2"]
    y_test_6 = data["y_test_mob6"]

    net = model.NnetSurvival(
        rngs=rngs,
        num_static_features=X_tr_arr.shape[1],
        shared_hidden=128,
        head_hidden=64,
        dropout_rate=0.3
    )
    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)

    N_train = len(y_train_6)
    best_val_loss = 9999.0
    best_state = None
    best_epoch = 0
    wait = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N_train)
        for s in range(0, N_train, batch_size):
            e = min(s + batch_size, N_train)
            idx = perm[s:e]
            train_step_surv(net, optimizer, X_tr_arr[idx], y_train_2[idx], y_train_6[idx])

        val_loss = evaluate_surv_loss(net, X_va_arr, y_valid_2, y_valid_6)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if restore_best:
                best_state = jax.tree.map(jnp.array, nnx.state(net))
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if restore_best and best_state is not None:
        nnx.update(net, best_state)

    train_time = time.time() - t_start
    test_metrics = evaluate_surv_test(net, X_te_arr, y_test_2, y_test_6)
    test_metrics["seed"] = seed
    test_metrics["best_epoch"] = best_epoch
    test_metrics["train_time_sec"] = train_time

    if verbose:
        print(f"  ✓ [Seed {seed:4d}] 完成 (耗时 {train_time:.1f}s, Best Epoch {best_epoch}): "
              f"MOB6 AUC={test_metrics['auc_mob6']:.4f}, KS={test_metrics['ks_mob6']:.4f} | "
              f"MOB2 AUC={test_metrics['auc_mob2']:.4f}, KS={test_metrics['ks_mob2']:.4f} | "
              f"VR={test_metrics['vr_pct']:.2f}%")

    return test_metrics, net


def run_10_seeds_nnet_survival(
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    restore_best: bool = False,
    return_best_model: bool = False
) -> Union[Tuple[pd.DataFrame, Dict[str, Dict[str, float]]], Tuple[pd.DataFrame, Dict[str, Dict[str, float]], model.NnetSurvival, Dict[str, np.ndarray]]]:
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999, 200, 300, 400, 500, 600]

    print("=" * 90)
    print(f"🚀 开始 NnetSurvival (离散时间生存模型) 训练评估 (Seeds: {seeds})")
    print("=" * 90)

    data = load_data()
    dev_data = to_device_arrays(data)
    results = []
    best_model = None
    best_loss = 9999.0

    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] 正在运行 Seed = {seed} ...")
        res, trained_net = train_single_seed_nnet_survival(
            seed=seed, data=dev_data, epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, restore_best=restore_best, verbose=True
        )
        results.append(res)
        cur_loss = res["loss_mob2"] + res["loss_mob6"]
        if best_model is None or cur_loss < best_loss:
            best_loss = cur_loss
            best_model = trained_net

    df_results = pd.DataFrame(results)
    stats_summary = compute_stats_summary(df_results, has_rectified=False)
    print_experiment_summary("NnetSurvival", df_results, stats_summary, has_rectified=False)

    if return_best_model:
        return df_results, stats_summary, best_model, data
    return df_results, stats_summary


# ====================================================================
# 6. TabMDualTargetNet (TabM 双目标模型, ICLR 2025)
# ====================================================================

@nnx.jit
def train_step_tabm(model_net: model.TabMDualTargetNet, optimizer: nnx.Optimizer, batch_x: jax.Array, y2: jax.Array, y6: jax.Array, w_reg: float = 0.0):
    def loss_fn(m):
        out = m(batch_x, training=True)
        loss2 = jnp.mean(optax.sigmoid_binary_cross_entropy(out["logits_mob2_k"], y2[:, None]))
        loss6 = jnp.mean(optax.sigmoid_binary_cross_entropy(out["logits_mob6_k"], y6[:, None]))
        reg_mono = jnp.mean(jax.nn.relu(out["p_mob2_k"] - out["p_mob6_k"]))
        return loss2 + loss6 + w_reg * reg_mono

    loss, grads = nnx.value_and_grad(loss_fn)(model_net)
    optimizer.update(model_net, grads)
    return loss


@nnx.jit
def eval_forward_tabm(model_net: model.TabMDualTargetNet, batch_x: jax.Array):
    out = model_net(batch_x, training=False)
    return out["pred_mob2"], out["pred_mob6"]


def evaluate_tabm_loss(model_net: model.TabMDualTargetNet, X: Union[np.ndarray, jax.Array], y2: jax.Array, y6: jax.Array, eval_bs: int = 1024) -> float:
    N = len(y6)
    total_loss = 0.0
    for s in range(0, N, eval_bs):
        e = min(s + eval_bs, N)
        bx = X[s:e]
        by2 = y2[s:e]
        by6 = y6[s:e]
        out = model_net(bx, training=False)
        l2 = jnp.mean(optax.sigmoid_binary_cross_entropy(out["logits_mob2_k"], by2[:, None]))
        l6 = jnp.mean(optax.sigmoid_binary_cross_entropy(out["logits_mob6_k"], by6[:, None]))
        total_loss += float(l2 + l6) * (e - s)
    return total_loss / N


def evaluate_tabm_test(model_net: model.TabMDualTargetNet, X: Union[np.ndarray, jax.Array], y2: np.ndarray, y6: np.ndarray, eval_bs: int = 1024) -> Dict[str, float]:
    N = len(y6)
    p2_list, p6_list = [], []
    for s in range(0, N, eval_bs):
        e = min(s + eval_bs, N)
        p2_b, p6_b = eval_forward_tabm(model_net, X[s:e])
        p2_list.append(np.asarray(p2_b))
        p6_list.append(np.asarray(p6_b))

    p2 = np.concatenate(p2_list)
    p6 = np.concatenate(p6_list)
    return compute_standard_eval_metrics(p2, p6, y2, y6, compute_rectified=True)


def train_single_seed_tabm(
    seed: int,
    data: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    w_reg: float = 0.0,
    restore_best: bool = False,
    verbose: bool = True
) -> Tuple[Dict[str, float], model.TabMDualTargetNet]:
    np.random.seed(seed)
    rngs = nnx.Rngs(seed)

    X_tr_arr = jnp.asarray(data["X_train"], dtype=jnp.float32)
    y_train_2 = jnp.asarray(data["y_train_mob2"], dtype=jnp.float32)
    y_train_6 = jnp.asarray(data["y_train_mob6"], dtype=jnp.float32)
    X_va_arr = jnp.asarray(data["X_valid"], dtype=jnp.float32)
    y_valid_2 = jnp.asarray(data["y_valid_mob2"], dtype=jnp.float32)
    y_valid_6 = jnp.asarray(data["y_valid_mob6"], dtype=jnp.float32)
    X_te_arr = jnp.asarray(data["X_test"], dtype=jnp.float32)
    y_test_2 = data["y_test_mob2"]
    y_test_6 = data["y_test_mob6"]

    net = model.TabMDualTargetNet(
        rngs=rngs,
        num_features=X_tr_arr.shape[1],
        d_hidden=128,
        n_layers=3,
        k=16,
        dropout_rate=0.5,
        bin_edges=bin_edges
    )
    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)

    N_train = len(y_train_6)
    best_val_loss = 9999.0
    best_state = None
    best_epoch = 0
    wait = 0
    t_start = time.time()

    for epoch in range(epochs):
        perm = np.random.permutation(N_train)
        for s in range(0, N_train, batch_size):
            e = min(s + batch_size, N_train)
            idx = perm[s:e]
            train_step_tabm(net, optimizer, X_tr_arr[idx], y_train_2[idx], y_train_6[idx], w_reg=w_reg)

        val_loss = evaluate_tabm_loss(net, X_va_arr, y_valid_2, y_valid_6)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if restore_best:
                best_state = jax.tree.map(jnp.array, nnx.state(net))
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if restore_best and best_state is not None:
        nnx.update(net, best_state)

    train_time = time.time() - t_start
    test_metrics = evaluate_tabm_test(net, X_te_arr, y_test_2, y_test_6)
    test_metrics["seed"] = seed
    test_metrics["best_epoch"] = best_epoch
    test_metrics["train_time_sec"] = train_time

    if verbose:
        n_viol = int(test_metrics.get("violations_count", 0))
        n_tot = int(test_metrics.get("total_samples", len(y_test_6)))
        print(f"  ✓ Seed {seed} 完成 (耗时 {train_time:.1f}s, Best Epoch {best_epoch}):")
        print(f"    - MOB6: AUC = {test_metrics['auc_mob6']:.4f}, KS = {test_metrics['ks_mob6']:.4f}, LogLoss = {test_metrics['loss_mob6']:.5f}")
        print(f"    - MOB2: AUC = {test_metrics['auc_mob2']:.4f}, KS = {test_metrics['ks_mob2']:.4f}, LogLoss = {test_metrics['loss_mob2']:.5f}")
        print(f"    - 跨期单调违规率 P(MOB6) < P(MOB2): {test_metrics['vr_pct']:.2f}% ({n_viol:,}/{n_tot:,} 笔倒挂)")
        if "auc_mob6_rect" in test_metrics:
            print(f"    - Post-hoc 修正后 MOB6: AUC = {test_metrics['auc_mob6_rect']:.4f}, KS = {test_metrics['ks_mob6_rect']:.4f}, LogLoss = {test_metrics.get('loss_mob6_rect', test_metrics['loss_mob6']):.5f}")

    return test_metrics, net


def run_10_seeds_tabm(
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    w_reg: float = 0.0,
    restore_best: bool = False,
    return_best_model: bool = False
) -> Union[Tuple[pd.DataFrame, Dict[str, Dict[str, float]]], Tuple[pd.DataFrame, Dict[str, Dict[str, float]], model.TabMDualTargetNet, Dict[str, np.ndarray]]]:
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999, 200, 300, 400, 500, 600]

    print("=" * 90)
    print(f"🚀 开始 TabM (双目标参数高效集成网络, ICLR 2025) 训练评估 (Seeds: {seeds})")
    print("=" * 90)

    data = load_data()
    bin_edges = common.compute_bin_edges(data["X_train"], n_bins=20)
    dev_data = to_device_arrays(data)

    results = []
    best_model = None
    best_loss = 9999.0

    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] 正在运行 Seed = {seed} ...")
        res, trained_net = train_single_seed_tabm(
            seed=seed, data=dev_data, bin_edges=bin_edges, epochs=epochs, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay, patience=patience, w_reg=w_reg, restore_best=restore_best, verbose=True
        )
        results.append(res)
        cur_loss = res["loss_mob2"] + res["loss_mob6"]
        if best_model is None or cur_loss < best_loss:
            best_loss = cur_loss
            best_model = trained_net

    df_results = pd.DataFrame(results)
    stats_summary = compute_stats_summary(df_results, has_rectified=True)
    print_experiment_summary("TabMDualTargetNet", df_results, stats_summary, has_rectified=True)

    if return_best_model:
        return df_results, stats_summary, best_model, data
    return df_results, stats_summary


# ====================================================================
# 7. PLE_MTL (渐进式分层抽取网络, Tang et al. RecSys 2020)
# ====================================================================

@nnx.jit
def train_step_ple(model_net: model.PLE_MTL, optimizer: nnx.Optimizer, batch_x: jax.Array, y2: jax.Array, y6: jax.Array, lambda_mono: float = 0.0):
    def loss_fn(m):
        logits_mob2, logits_mob6 = m(batch_x, training=True)
        loss6 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_mob6.squeeze(-1), y6))
        loss2 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_mob2.squeeze(-1), y2))
        pred_st = jax.nn.sigmoid(logits_mob2)
        pred_lt = jax.nn.sigmoid(logits_mob6)
        reg_loss = jnp.mean(jnp.maximum(0.0, pred_st - pred_lt))
        return loss6 + loss2 + lambda_mono * reg_loss

    loss, grads = nnx.value_and_grad(loss_fn)(model_net)
    optimizer.update(model_net, grads)
    return loss


@nnx.jit
def eval_forward_ple(model_net: model.PLE_MTL, batch_x: jax.Array):
    logits_mob2, logits_mob6 = model_net(batch_x, training=False)
    p_mob2 = jax.nn.sigmoid(logits_mob2).squeeze(-1)
    p_mob6 = jax.nn.sigmoid(logits_mob6).squeeze(-1)
    return p_mob2, p_mob6


def evaluate_ple(model_net: model.PLE_MTL, X: np.ndarray, y2: np.ndarray, y6: np.ndarray, eval_bs: int = 1024) -> Tuple[float, Dict[str, float]]:
    N = len(y6)
    p2_list, p6_list = [], []
    for s in range(0, N, eval_bs):
        e = min(s + eval_bs, N)
        p2_b, p6_b = eval_forward_ple(model_net, X[s:e])
        p2_list.append(np.asarray(p2_b))
        p6_list.append(np.asarray(p6_b))

    p2 = np.concatenate(p2_list)
    p6 = np.concatenate(p6_list)
    metrics = compute_standard_eval_metrics(p2, p6, y2, y6, compute_rectified=False)
    total_loss = metrics["loss_mob2"] + metrics["loss_mob6"]
    return total_loss, metrics


def train_single_seed_ple(
    seed: int,
    data: Dict[str, np.ndarray],
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    lambda_mono: float = 0.0,
    verbose: bool = True
) -> Tuple[Dict[str, float], model.PLE_MTL]:
    np.random.seed(seed)
    rngs = nnx.Rngs(seed)

    X_tr_arr = jnp.asarray(data["X_train"], dtype=jnp.float32)
    y_train_2 = jnp.asarray(data["y_train_mob2"], dtype=jnp.float32)
    y_train_6 = jnp.asarray(data["y_train_mob6"], dtype=jnp.float32)
    X_va_arr = jnp.asarray(data["X_valid"], dtype=jnp.float32)
    y_valid_2 = jnp.asarray(data["y_valid_mob2"], dtype=jnp.float32)
    y_valid_6 = jnp.asarray(data["y_valid_mob6"], dtype=jnp.float32)
    X_te_arr = jnp.asarray(data["X_test"], dtype=jnp.float32)
    y_test_2 = data["y_test_mob2"]
    y_test_6 = data["y_test_mob6"]

    net = model.PLE_MTL(
        rngs=rngs,
        num_static_features=X_tr_arr.shape[1],
        expert_hidden=128,
        head_hidden=64,
        num_task1_experts=2,
        num_task2_experts=2,
        num_shared_experts=2,
        num_layers=1,
        dropout_rate=0.5
    )
    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)

    N_train = len(y_train_6)
    best_val_loss = 9999.0
    best_state = None
    best_epoch = 0
    wait = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N_train)
        for s in range(0, N_train, batch_size):
            e = min(s + batch_size, N_train)
            idx = perm[s:e]
            train_step_ple(net, optimizer, X_tr_arr[idx], y_train_2[idx], y_train_6[idx], lambda_mono=lambda_mono)

        val_loss, _ = evaluate_ple(net, X_va_arr, y_valid_2, y_valid_6)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = jax.tree.map(jnp.array, nnx.state(net))
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        nnx.update(net, best_state)

    train_time = time.time() - t_start
    _, test_metrics = evaluate_ple(net, X_te_arr, y_test_2, y_test_6)
    test_metrics["seed"] = seed
    test_metrics["best_epoch"] = best_epoch
    test_metrics["train_time_sec"] = train_time

    if verbose:
        print(f"  ✓ [Seed {seed:4d}] 完成 (耗时 {train_time:.1f}s, Best Epoch {best_epoch}): "
              f"MOB6 AUC={test_metrics['auc_mob6']:.4f}, KS={test_metrics['ks_mob6']:.4f} | "
              f"MOB2 AUC={test_metrics['auc_mob2']:.4f}, KS={test_metrics['ks_mob2']:.4f} | "
              f"VR={test_metrics['vr_pct']:.2f}%")

    return test_metrics, net


def run_10_seeds_ple(
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    patience: int = 20,
    lambda_mono: float = 0.0,
    return_best_model: bool = False
) -> Union[Tuple[pd.DataFrame, Dict[str, Dict[str, float]]], Tuple[pd.DataFrame, Dict[str, Dict[str, float]], model.PLE_MTL, Dict[str, np.ndarray]]]:
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999, 200, 300, 400, 500, 600]

    print("=" * 90)
    print(f"🚀 开始 PLE-MTL (Progressive Layered Extraction) 训练评估 (Seeds: {seeds})")
    print("=" * 90)

    data = load_data()
    dev_data = to_device_arrays(data)
    results = []
    best_model = None
    best_loss = 9999.0

    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] 正在运行 Seed = {seed} ...")
        res, trained_net = train_single_seed_ple(
            seed=seed, data=dev_data, epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, lambda_mono=lambda_mono, verbose=True
        )
        results.append(res)
        cur_loss = res["loss_mob2"] + res["loss_mob6"]
        if best_model is None or cur_loss < best_loss:
            best_loss = cur_loss
            best_model = trained_net

    df_results = pd.DataFrame(results)
    stats_summary = compute_stats_summary(df_results, has_rectified=False)
    print_experiment_summary("PLE-MTL", df_results, stats_summary, has_rectified=False)

    if return_best_model:
        return df_results, stats_summary, best_model, data
    return df_results, stats_summary


# ====================================================================
# 8. 统一调度入口与全模型 Benchmark 对比
# ====================================================================

MODEL_DISPATCHER = {
    "hard_monotonic": run_10_seeds_hard_monotonic,
    "HardMonotonicRiskNet": run_10_seeds_hard_monotonic,
    "base_mtl": run_10_seeds_base_mtl,
    "BaseMTL": run_10_seeds_base_mtl,
    "mlp_stl": run_10_seeds_mlp_stl,
    "MLP_STL": run_10_seeds_mlp_stl,
    "hist_gbdt": run_10_seeds_hist_gbdt,
    "HistGBDT": run_10_seeds_hist_gbdt,
    "nnet_survival": run_10_seeds_nnet_survival,
    "NnetSurvival": run_10_seeds_nnet_survival,
    "tabm": run_10_seeds_tabm,
    "TabMDualTargetNet": run_10_seeds_tabm,
    "ple_mtl": run_10_seeds_ple,
    "PLE_MTL": run_10_seeds_ple,
}


def run_model_experiment(
    model_name: str,
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 512,
    patience: int = 20,
    restore_best: bool = False,
    w_reg: float = 5.0,
    return_best_model: bool = False
):
    """
    通用单模型实验运行入口
    """
    key = model_name.strip()
    if key not in MODEL_DISPATCHER:
        raise ValueError(f"未知模型名称: {model_name}，可选模型列表: {list(MODEL_DISPATCHER.keys())}")

    fn = MODEL_DISPATCHER[key]
    if key in ["hist_gbdt", "HistGBDT"]:
        return fn(seeds=seeds, return_best_model=return_best_model)
    elif key in ["base_mtl", "BaseMTL"]:
        return fn(seeds=seeds, epochs=epochs, batch_size=batch_size, patience=patience, w_reg=w_reg, return_best_model=return_best_model)
    elif key in ["tabm", "TabMDualTargetNet"]:
        return fn(seeds=seeds, epochs=epochs, batch_size=batch_size, patience=patience, w_reg=w_reg, restore_best=restore_best, return_best_model=return_best_model)
    elif key in ["nnet_survival", "NnetSurvival"]:
        return fn(seeds=seeds, epochs=epochs, batch_size=batch_size, patience=patience, restore_best=restore_best, return_best_model=return_best_model)
    else:
        return fn(seeds=seeds, epochs=epochs, batch_size=batch_size, patience=patience, return_best_model=return_best_model)


def run_benchmark(
    model_names: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 512,
    patience: int = 20,
    restore_best: bool = False,
    w_reg: float = 5.0
) -> pd.DataFrame:
    """
    对多个基准模型运行自动化 Benchmark，输出顶刊论文综合对照大表
    """
    if model_names is None:
        model_names = [
            "HistGBDT",
            "MLP_STL",
            "BaseMTL",
            "PLE_MTL",
            "NnetSurvival",
            "TabMDualTargetNet",
            "HardMonotonicRiskNet",
        ]

    summary_rows = []
    print("\n" + "=" * 90)
    print(f"🏁 启动全模型对比 Benchmark (候选模型: {model_names})")
    print("=" * 90)

    for m_name in model_names:
        print(f"\n{'='*30} 正在评测模型: {m_name} {'='*30}")
        df_res, stats = run_model_experiment(
            m_name, seeds=seeds, epochs=epochs, batch_size=batch_size, patience=patience, restore_best=restore_best, w_reg=w_reg, return_best_model=False
        )

        row = {
            "Model": m_name,
            "MOB2 AUC": f"{stats['auc_mob2']['mean']:.4f} ± {stats['auc_mob2']['std']:.4f}",
            "MOB2 KS": f"{stats['ks_mob2']['mean']:.4f} ± {stats['ks_mob2']['std']:.4f}",
            "MOB2 LogLoss": f"{stats['loss_mob2']['mean']:.5f} ± {stats['loss_mob2']['std']:.5f}",
            "MOB6 AUC": f"{stats['auc_mob6']['mean']:.4f} ± {stats['auc_mob6']['std']:.4f}",
            "MOB6 KS": f"{stats['ks_mob6']['mean']:.4f} ± {stats['ks_mob6']['std']:.4f}",
            "MOB6 LogLoss": f"{stats['loss_mob6']['mean']:.5f} ± {stats['loss_mob6']['std']:.5f}",
            "Violation Rate (VR)": f"{stats['vr_pct']['mean']:.2f}% ± {stats['vr_pct']['std']:.2f}%",
        }
        summary_rows.append(row)

    df_benchmark = pd.DataFrame(summary_rows)
    print("\n\n" + "=" * 115)
    print("🏆 【顶刊论文全模型最终对比总表 (Grand Benchmark Comparison)】")
    print("=" * 115)
    print(df_benchmark.to_string(index=False))
    print("=" * 115 + "\n")

    return df_benchmark


if __name__ == "__main__":
    df_res, stats = run_10_seeds_hard_monotonic(seeds=[42])
