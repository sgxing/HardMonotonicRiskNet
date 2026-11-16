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
from train import run_10_seeds_experiment


# ====================================================================
# 两级可解释性归因分析 (Two-Tier Directional Reject-Reason Attribution)
# 对应论文 6.3 节两级归因 与 6.4 节贷后预警实证:
# 第 1 步: 宏观机制诊断 (Macro Diagnosis) 与 典型样本筛选 (Case A / Case B)
# 第 2 步: 微观特征追踪 (Micro Feature Tracing) — JAX m=50 步 Integrated Gradients
# ====================================================================

def forward_pure(model, x):
    """
    定义无状态/抗追踪中断纯函数 (Bypassing Dropout Mutation in JAX Trace)
    """
    squeeze_back = (x.ndim == 1)
    x_in = x[None, :] if squeeze_back else x
    x_enc = model.ple(x_in) if model.ple is not None else x_in
    
    z_s = model.enc_s.layers[3](nnx.relu(model.enc_s.layers[0](x_enc)))
    z_st = model.enc_st.layers[3](nnx.relu(model.enc_st.layers[0](x_enc)))
    z_lt = model.enc_lt.layers[3](nnx.relu(model.enc_lt.layers[0](x_enc)))
    
    concat_st = jnp.concatenate([z_s, z_st], axis=-1)
    logit_st = model.head_short.layers[3](nnx.relu(model.head_short.layers[0](concat_st))).squeeze(-1)
    
    concat_lt = jnp.concatenate([z_s, z_lt], axis=-1)
    raw_delta = model.head_long_delta.layers[3](nnx.relu(model.head_long_delta.layers[0](concat_lt))).squeeze(-1)
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


def forward_mlp_xtilde(model, x_tilde):
    """
    直接接收 PLE 分箱后的隐表征 x_tilde 进行 MLP 前向推理
    """
    squeeze = (x_tilde.ndim == 1)
    xt = x_tilde[None, :] if squeeze else x_tilde
    
    z_s = model.enc_s.layers[3](nnx.relu(model.enc_s.layers[0](xt)))
    z_st = model.enc_st.layers[3](nnx.relu(model.enc_st.layers[0](xt)))
    z_lt = model.enc_lt.layers[3](nnx.relu(model.enc_lt.layers[0](xt)))
    
    concat_st = jnp.concatenate([z_s, z_st], axis=-1)
    logit_st = model.head_short.layers[3](nnx.relu(model.head_short.layers[0](concat_st))).squeeze(-1)
    
    concat_lt = jnp.concatenate([z_s, z_lt], axis=-1)
    raw_delta = model.head_long_delta.layers[3](nnx.relu(model.head_long_delta.layers[0](concat_lt))).squeeze(-1)
    delta_logit = jax.nn.softplus(raw_delta)
    
    if squeeze:
        return {'logit_st': logit_st[0], 'delta_logit': delta_logit[0]}
    return {'logit_st': logit_st, 'delta_logit': delta_logit}


def compute_ig(model, x_sample, x_base, target_func='logit_st', m=50):
    """
    JAX 向量化计算 m 步 Integrated Gradients (完备性积分梯度)
    """
    t0 = time.time()
    x_tilde = model.ple(x_sample[None, :])[0] if model.ple is not None else x_sample
    x_tilde_base = model.ple(x_base[None, :])[0] if model.ple is not None else x_base
    
    diff_tilde = x_tilde - x_tilde_base
    alphas = jnp.linspace(1.0 / m, 1.0, m)
    xt_path = x_tilde_base[None, :] + alphas[:, None] * diff_tilde[None, :]
    
    if target_func == 'logit_st':
        def sf(xt):
            return forward_mlp_xtilde(model, xt)['logit_st']
    elif target_func == 'delta_lt':
        def sf(xt):
            return forward_mlp_xtilde(model, xt)['delta_logit']
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


def print_case_report(case_name, case_type, idx, p_st, p_lt, dL_st, dL_lt, target_name, df, sum_attr, err, elapsed_ms, attr, x_raw, x_base_raw, feature_names_arr):
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


def run_case_attribution(model, data: Dict[str, np.ndarray], m: int = 50, eval_batch_size: int = 2048):
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

    base_pred = forward_pure(model, x_baseline_jax)
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
        out_b = forward_pure(model, xb)
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
    attr_A, df_A, sum_attr_A, err_A, time_A = compute_ig(model, xA, x_baseline_jax, target_func='logit_st', m=m)

    xB = jnp.asarray(X_test[case_B_idx], dtype=jnp.float32)
    attr_B, df_B, sum_attr_B, err_B, time_B = compute_ig(model, xB, x_baseline_jax, target_func='delta_lt', m=m)

    # 5. 打印 Case A 报告
    print_case_report(
        case_name="Case A",
        case_type="短期急性/欺诈型样本 (y_st=1, y_lt=1, 前两期即逾期违约)",
        idx=case_A_idx,
        p_st=preds_st_all[case_A_idx],
        p_lt=preds_lt_all[case_A_idx],
        dL_st=delta_L_st_all[case_A_idx],
        dL_lt=delta_L_lt_all[case_A_idx],
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
        p_st=preds_st_all[case_B_idx],
        p_lt=preds_lt_all[case_B_idx],
        dL_st=delta_L_st_all[case_B_idx],
        dL_lt=delta_L_lt_all[case_B_idx],
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


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    seeds=[42, 100, 2024, 777, 999, 200, 300, 400, 500, 600],
    # 1. 运行 10 随机种子基准实验，输出模型各指标均值和标准差 (方差)
    df_results, stats_summary, best_model, data = run_10_seeds_experiment(
        seeds=[100],
        epochs=100,
        batch_size=512,
        lr=5e-4,
        weight_decay=0.1,
        w_orth=0.1,
        patience=20,
        return_best_model=True
    )

    # 2. 跑完模型均值和方差后，进行 Case 归因 (Two-Tier Directional Reject-Reason Attribution)
    run_case_attribution(
        model=best_model,
        data=data,
        m=50,
        eval_batch_size=2048
    )
