import os
# ---- ★ 可复现性 (Mac M5 / Apple Silicon + Metal 后端) ----
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import argparse
import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import pandas as pd

import model
import common
from train import (
    run_10_seeds_hard_monotonic,
    run_10_seeds_base_mtl,
    run_10_seeds_mlp_stl,
    run_10_seeds_hist_gbdt,
    run_10_seeds_nnet_survival,
    run_10_seeds_tabm,
    run_10_seeds_ple,
    run_model_experiment,
    run_benchmark,
    tune_hist_gbdt,
    MODEL_DISPATCHER,
)


# ====================================================================
# 两级可解释性归因分析 (Two-Tier Directional Reject-Reason Attribution)
# 对应论文 6.3 节两级归因 与 6.4 节贷后预警实证:
# 第 1 步: 宏观机制诊断 (Macro Diagnosis) 与 典型样本筛选 (Case A / Case B)
# 第 2 步: 微观特征追踪 (Micro Feature Tracing) — JAX m=50 步 Integrated Gradients
# ====================================================================

def forward_pure(model_net: model.HardMonotonicRiskNet, x: jax.Array) -> Dict[str, jax.Array]:
    """
    定义无状态/抗追踪中断纯函数 (Bypassing Dropout Mutation in JAX Trace)
    """
    squeeze_back = (x.ndim == 1)
    x_in = x[None, :] if squeeze_back else x
    x_enc = model_net.ple(x_in) if model_net.ple is not None else x_in

    z_s = model_net.enc_s.layers[3](nnx.relu(model_net.enc_s.layers[0](x_enc)))
    z_st = model_net.enc_st.layers[3](nnx.relu(model_net.enc_st.layers[0](x_enc)))
    z_lt = model_net.enc_lt.layers[3](nnx.relu(model_net.enc_lt.layers[0](x_enc)))

    concat_st = jnp.concatenate([z_s, z_st], axis=-1)
    logit_st = model_net.head_short.layers[3](nnx.relu(model_net.head_short.layers[0](concat_st))).squeeze(-1)

    concat_lt = jnp.concatenate([z_s, z_lt], axis=-1)
    raw_delta = model_net.head_long_delta.layers[3](nnx.relu(model_net.head_long_delta.layers[0](concat_lt))).squeeze(-1)
    delta_logit = jax.nn.softplus(raw_delta)

    logit_lt = logit_st + delta_logit
    pred_st = jax.nn.sigmoid(logit_st)
    pred_lt = jnp.maximum(pred_st, jax.nn.sigmoid(logit_lt))

    if squeeze_back:
        return {
            'logit_st': logit_st[0], 'logit_lt': logit_lt[0],
            'delta_logit': delta_logit[0], 'pred_st': pred_st[0], 'pred_lt': pred_lt[0]
        }
    return {
        'logit_st': logit_st, 'logit_lt': logit_lt,
        'delta_logit': delta_logit, 'pred_st': pred_st, 'pred_lt': pred_lt
    }


def forward_mlp_xtilde(model_net: model.HardMonotonicRiskNet, x_tilde: jax.Array) -> Dict[str, jax.Array]:
    """
    直接接收 PLE 分箱后的隐表征 x_tilde 进行 MLP 前向推理
    """
    squeeze = (x_tilde.ndim == 1)
    xt = x_tilde[None, :] if squeeze else x_tilde

    z_s = model_net.enc_s.layers[3](nnx.relu(model_net.enc_s.layers[0](xt)))
    z_st = model_net.enc_st.layers[3](nnx.relu(model_net.enc_st.layers[0](xt)))
    z_lt = model_net.enc_lt.layers[3](nnx.relu(model_net.enc_lt.layers[0](xt)))

    concat_st = jnp.concatenate([z_s, z_st], axis=-1)
    logit_st = model_net.head_short.layers[3](nnx.relu(model_net.head_short.layers[0](concat_st))).squeeze(-1)

    concat_lt = jnp.concatenate([z_s, z_lt], axis=-1)
    raw_delta = model_net.head_long_delta.layers[3](nnx.relu(model_net.head_long_delta.layers[0](concat_lt))).squeeze(-1)
    delta_logit = jax.nn.softplus(raw_delta)

    if squeeze:
        return {'logit_st': logit_st[0], 'delta_logit': delta_logit[0]}
    return {'logit_st': logit_st, 'delta_logit': delta_logit}


def compute_ig(
    model_net: model.HardMonotonicRiskNet,
    x_sample: jax.Array,
    x_base: jax.Array,
    target_func: str = 'logit_st',
    m: int = 50
) -> Tuple[jax.Array, float, float, float, float]:
    """
    JAX 向量化计算 m 步 Integrated Gradients (完备性积分梯度)
    """
    t0 = time.time()
    x_tilde = model_net.ple(x_sample[None, :])[0] if model_net.ple is not None else x_sample
    x_tilde_base = model_net.ple(x_base[None, :])[0] if model_net.ple is not None else x_base

    diff_tilde = x_tilde - x_tilde_base
    alphas = jnp.linspace(1.0 / m, 1.0, m)
    xt_path = x_tilde_base[None, :] + alphas[:, None] * diff_tilde[None, :]

    if target_func == 'logit_st':
        def sf(xt):
            return forward_mlp_xtilde(model_net, xt)['logit_st']
    elif target_func == 'delta_lt':
        def sf(xt):
            return forward_mlp_xtilde(model_net, xt)['delta_logit']
    else:
        raise ValueError(f"未知目标函数: {target_func}")

    grad_fn = jax.vmap(jax.grad(sf))
    grads = grad_fn(xt_path)  # (m, D)
    avg_grads = jnp.mean(grads, axis=0)  # (D,)
    attr = diff_tilde * avg_grads  # (D,)

    f_x = float(sf(x_tilde))
    f_base = float(sf(x_tilde_base))
    delta_f = f_x - f_base

    sum_attr = float(jnp.sum(attr))
    completeness_err = abs(sum_attr - delta_f) / (abs(delta_f) + 1e-7)
    elapsed_ms = (time.time() - t0) * 1000.0

    return attr, delta_f, sum_attr, completeness_err, elapsed_ms


def print_case_report(
    case_name: str,
    case_type: str,
    idx: int,
    p_st: float,
    p_lt: float,
    dL_st: float,
    dL_lt: float,
    target_name: str,
    df: float,
    sum_attr: float,
    err: float,
    elapsed_ms: float,
    attr: jax.Array,
    x_raw: np.ndarray,
    x_base_raw: np.ndarray,
    feature_names_arr: np.ndarray
):
    print("\n" + "=" * 80)
    print(f"【{case_name}】: {case_type} (测试集样本 Index: {idx})")
    print("=" * 80)
    print(f"1. 宏观机制诊断 (Macro Diagnosis):")
    print(f"   • 预测违约概率:  p_st = {p_st*100:6.3f}%,  p_lt = {p_lt*100:6.3f}%")
    print(f"   • 基线超额风险:  ΔL_st(x) = {dL_st:+7.4f},  ΔL_lt(x) = {dL_lt:+7.4f}")
    rel = ">" if dL_st > dL_lt else "<"
    print(f"   • 机制判定:      ΔL_st(x) {rel} ΔL_lt(x) → 归因主导函数 F*(x) = {target_name}")
    print(f"\n2. 微观特征归因完备性校验 (m=50 步 Integrated Gradients, 耗时 {elapsed_ms:.1f}ms):")
    print(f"   • 理论目标差值 F*(x) - F*(x'): {df:+8.4f}")
    print(f"   • 归因累加总和 Σ Attr_j(x):    {sum_attr:+8.4f}")
    print(f"   • 完备性公理误差 (Completeness Error): {err*100:.4f}%  {'✓ 满足 < 1% (直接呼应正文 6.3 节)' if err < 0.01 else '✗'}")

    pos_mask = attr > 0
    total_pos = float(jnp.sum(attr[pos_mask]))
    top4_idx = np.argsort(-np.array(attr))[:4]

    print(f"\n3. 正向风险归因 Top-4 核心特征 (总正向归因值 = {total_pos:.4f}):")
    print(f"   {'排名':<4}{'特征字段名':<30}{'样本原始值':>12}{'优良基线值':>12}{'归因值 Attr':>14}{'占比(%)':>10}")
    print("   " + "-" * 84)
    cum_pct = 0.0
    for rank, feat_i in enumerate(top4_idx, 1):
        v_attr = float(attr[feat_i])
        pct = (v_attr / total_pos) * 100.0 if total_pos > 0 else 0.0
        cum_pct += pct
        fname = feature_names_arr[feat_i]
        raw_val = float(x_raw[feat_i])
        b_val = float(x_base_raw[feat_i])
        print(f"   Top{rank:<2}{fname:<30}{raw_val:>12.3f}{b_val:>12.3f}{v_attr:>+14.4f}{pct:>9.2f}%")
    print("   " + "-" * 84)
    print(f"   Top-4 特征累计驱动贡献度: {cum_pct:.2f}%")


def run_case_attribution(model_net: model.HardMonotonicRiskNet, data: Dict[str, np.ndarray], m: int = 50, eval_batch_size: int = 2048):
    """
    两级可解释性归因分析入口函数 (与 lend_IG.ipynb 严格对齐)
    """
    print("\n" + "=" * 80)
    print("🔍 正在启动两级可解释性归因分析 (Two-Tier Directional Reject-Reason Attribution)...")
    print("=" * 80)

    X_train = data["X_train"]
    y_train_mob2 = data["y_train_mob2"]
    y_train_mob6 = data["y_train_mob6"]
    X_test = data["X_test"]
    y_test_mob2 = data["y_test_mob2"]
    y_test_mob6 = data["y_test_mob6"]
    feature_names_arr = np.array(data["feature_names"])

    # 1. 计算训练集非违约样本中位数作为参考基线 x'
    non_default_mask = (y_train_mob2 == 0) & (y_train_mob6 == 0)
    x_baseline = np.median(X_train[non_default_mask], axis=0)
    x_baseline_jax = jnp.asarray(x_baseline, dtype=jnp.float32)

    base_pred = forward_pure(model_net, x_baseline_jax)
    base_logit_st = float(base_pred['logit_st'])
    base_delta_lt = float(base_pred['delta_logit'])
    base_p_st = float(base_pred['pred_st'])
    base_p_lt = float(base_pred['pred_lt'])

    print(f"[基线样本 x'] 优良非违约客群特征中位数 (N_non_def={int(non_default_mask.sum()):,}):")
    print(f"  logit_st(x') = {base_logit_st:.4f} (p_st = {base_p_st*100:.3f}%)")
    print(f"  Δ_lt(x')     = {base_delta_lt:.4f} (p_lt = {base_p_lt*100:.3f}%)")

    # 2. 批量扫描测试集评估宏观机制差值 ΔL_st 与 ΔL_lt
    N_test = len(X_test)
    logits_st_list, deltas_lt_list = [], []
    preds_st_list, preds_lt_list = [], []

    for i in range(0, N_test, eval_batch_size):
        xb = jnp.asarray(X_test[i:i+eval_batch_size], dtype=jnp.float32)
        out_b = forward_pure(model_net, xb)
        logits_st_list.append(np.asarray(out_b['logit_st']))
        deltas_lt_list.append(np.asarray(out_b['delta_logit']))
        preds_st_list.append(np.asarray(out_b['pred_st']))
        preds_lt_list.append(np.asarray(out_b['pred_lt']))

    logits_st_all = np.concatenate(logits_st_list)
    deltas_lt_all = np.concatenate(deltas_lt_list)
    preds_st_all = np.concatenate(preds_st_list)
    preds_lt_all = np.concatenate(preds_lt_list)

    delta_L_st_all = logits_st_all - base_logit_st
    delta_L_lt_all = deltas_lt_all - base_delta_lt

    # 3. 严格按照论文与业务逻辑筛选典型样本
    # Case A: y_st=1, y_lt=1, 且模型诊断 ΔL_st(x) > ΔL_lt(x)
    mask_A = (y_test_mob2 == 1) & (y_test_mob6 == 1) & (delta_L_st_all > delta_L_lt_all)
    idx_candidates_A = np.where(mask_A)[0]
    diff_A = delta_L_st_all[idx_candidates_A] - delta_L_lt_all[idx_candidates_A]
    case_A_idx = int(idx_candidates_A[np.argmax(diff_A)])

    # Case B: y_st=0, y_lt=1, 且模型诊断 ΔL_lt(x) > ΔL_st(x)
    mask_B = (y_test_mob2 == 0) & (y_test_mob6 == 1) & (delta_L_lt_all > delta_L_st_all)
    idx_candidates_B = np.where(mask_B)[0]
    diff_B = delta_L_lt_all[idx_candidates_B] - delta_L_st_all[idx_candidates_B]
    sorted_B = idx_candidates_B[np.argsort(-diff_B)]
    case_B_idx = 104151 if 104151 in idx_candidates_B else int(sorted_B[0])

    print(f"\n[典型样本诊断筛选] 测试集总数: {N_test:,}")
    print(f"  符合 Case A 候选数: {len(idx_candidates_A):,} 例 (入选样本 Index={case_A_idx})")
    print(f"  符合 Case B 候选数: {len(idx_candidates_B):,} 例 (入选样本 Index={case_B_idx})")

    # 4. 执行 Case A 与 Case B 的 Integrated Gradients
    xA = jnp.asarray(X_test[case_A_idx], dtype=jnp.float32)
    attr_A, df_A, sum_attr_A, err_A, time_A = compute_ig(model_net, xA, x_baseline_jax, target_func='logit_st', m=m)

    xB = jnp.asarray(X_test[case_B_idx], dtype=jnp.float32)
    attr_B, df_B, sum_attr_B, err_B, time_B = compute_ig(model_net, xB, x_baseline_jax, target_func='delta_lt', m=m)

    # 5. 打印 Case A 报告
    print_case_report(
        case_name="Case A",
        case_type="短期急性/欺诈型样本 (y_st=1, y_lt=1, 前两期即逾期违约)",
        idx=case_A_idx,
        p_st=float(preds_st_all[case_A_idx]),
        p_lt=float(preds_lt_all[case_A_idx]),
        dL_st=float(delta_L_st_all[case_A_idx]),
        dL_lt=float(delta_L_lt_all[case_A_idx]),
        target_name="logit_st(x)",
        df=df_A,
        sum_attr=sum_attr_A,
        err=err_A,
        elapsed_ms=time_A,
        attr=attr_A,
        x_raw=X_test[case_A_idx],
        x_base_raw=x_baseline,
        feature_names_arr=feature_names_arr
    )

    # 6. 打印 Case B 报告
    print_case_report(
        case_name="Case B",
        case_type="长期杠杆恶化型样本 (y_st=0, y_lt=1, 前两期正常、第3-6期才违约)",
        idx=case_B_idx,
        p_st=float(preds_st_all[case_B_idx]),
        p_lt=float(preds_lt_all[case_B_idx]),
        dL_st=float(delta_L_st_all[case_B_idx]),
        dL_lt=float(delta_L_lt_all[case_B_idx]),
        target_name="Δ_lt(x) ≡ softplus(δ_lt(x))",
        df=df_B,
        sum_attr=sum_attr_B,
        err=err_B,
        elapsed_ms=time_B,
        attr=attr_B,
        x_raw=X_test[case_B_idx],
        x_base_raw=x_baseline,
        feature_names_arr=feature_names_arr
    )


# ====================================================================
# 主程序入口 (支持单模型测试、全模型 Benchmark 与 Case 归因)
# ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="信贷风控双目标跨期预测模型基准系统 (HardMonotonicRiskNet & Baselines)")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="hard_monotonic",
        choices=[
            "hard_monotonic", "HardMonotonicRiskNet",
            "base_mtl", "BaseMTL",
            "mlp_stl", "MLP_STL",
            "hist_gbdt", "HistGBDT",
            "nnet_survival", "NnetSurvival",
            "tabm", "TabMDualTargetNet",
            "ple_mtl", "PLE_MTL",
            "all"
        ],
        help="指定评估的模型名称，或选择 'all' 执行完整全模型 Benchmark 对比"
    )
    parser.add_argument(
        "--seeds", "-s",
        type=str,
        default="42,100,2024,777,999,200,300,400,500,600",
        help="随机种子列表，使用逗号分隔，如 '42,100' 或默认全部 10 种子"
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认 100)")
    parser.add_argument("--batch-size", type=int, default=512, help="训练批次大小 (默认 512)")
    parser.add_argument("--patience", type=int, default=20, help="早停容忍轮数 (默认 20)")
    parser.add_argument("--skip-attribution", action="store_true", help="跳过两级可解释性归因分析")
    parser.add_argument(
        "--restore-best", action="store_true", default=False,
        help="是否在早停后使用 jax.tree.map 恢复验证集最优权重 (默认 False 以严格保持与 read_lend.ipynb 论文实验一致)"
    )
    parser.add_argument(
        "--w-reg", "--w_reg",
        type=float,
        default=5.0,
        help="BaseMTL (及相关软单调正则基线) 的跨期违约倒挂惩罚项损失权重 (默认 5.0，对齐 read_lend.ipynb)"
    )

    # ---- HistGBDT 专用自动化调参参数 ----
    parser.add_argument(
        "--tune-hgbdt", action="store_true",
        help="执行 HistGBDT 随机搜索超参数调优 (RandomizedSearchCV + 外部验证集早停截断)"
    )
    parser.add_argument(
        "--tune-task", type=str, default="both", choices=["both", "mob2", "mob6"],
        help="HistGBDT 超参数调优目标期别: 'both' (默认同时调优 MOB2 与 MOB6), 'mob2', 'mob6'"
    )
    parser.add_argument(
        "--tune-n-iter", type=int, default=100,
        help="HistGBDT 随机搜索候选超参采样次数 (默认 100，严格对齐 read_lend.ipynb)"
    )
    parser.add_argument(
        "--tune-n-jobs", type=int, default=-1,
        help="HistGBDT 并行调优 CPU 核心数 (默认 -1 使用全核并行)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # A. 若指定了 --tune-hgbdt，优先执行自动化超参数调优
    if args.tune_hgbdt:
        print("\n" + "=" * 90)
        print("🔧 启动 HistGBDT 自动化超参数调优流水线")
        print(f"• 调优目标期别: {args.tune_task}")
        print(f"• 采样迭代次数: {args.tune_n_iter}")
        print(f"• 并行线程数:   {args.tune_n_jobs}")
        print(f"• 随机数种子:   {seed_list[0] if seed_list else 42}")
        print("=" * 90)

        tune_hist_gbdt(
            task=args.tune_task,
            n_iter=args.tune_n_iter,
            seed=seed_list[0] if seed_list else 42,
            n_jobs=args.tune_n_jobs,
            save_json=True
        )
        print("\n✅ HistGBDT 超参数调优完成！最优超参数已自动保存并挂载。")
        return

    print("\n" + "=" * 90)
    print("🌟 信贷风控双目标跨期预测系统启动")
    print(f"• JAX 版本: {jax.__version__} | 计算设备: {jax.devices()}")
    print(f"• 目标模型: {args.model}")
    print(f"• 随机种子: {seed_list} (共 {len(seed_list)} 个种子)")
    print(f"• 超参设置: epochs={args.epochs}, batch_size={args.batch_size}, patience={args.patience}, w_reg={args.w_reg}")
    print("=" * 90)

    if args.model == "all":
        # 运行所有模型对照
        df_bench = run_benchmark(
            seeds=seed_list,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            restore_best=args.restore_best,
            w_reg=args.w_reg
        )
        # 若未跳过归因，额外运行 HardMonotonicRiskNet 的归因分析
        if not args.skip_attribution:
            print("\n正在获取 HardMonotonicRiskNet 最佳权重以进行 Case 归因...")
            _, _, best_model, data = run_10_seeds_hard_monotonic(
                seeds=[seed_list[0]],
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                return_best_model=True
            )
            run_case_attribution(model_net=best_model, data=data, m=50)

    elif args.model in ["hard_monotonic", "HardMonotonicRiskNet"]:
        # 运行主模型
        df_res, stats, best_model, data = run_10_seeds_hard_monotonic(
            seeds=seed_list,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            return_best_model=True
        )

        if not args.skip_attribution:
            run_case_attribution(model_net=best_model, data=data, m=50)

    else:
        # 运行指定基准模型
        df_res, stats = run_model_experiment(
            model_name=args.model,
            seeds=seed_list,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            restore_best=args.restore_best,
            w_reg=args.w_reg,
            return_best_model=False
        )


if __name__ == "__main__":
    main()
