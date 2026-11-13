import os
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from datetime import datetime
import time
import pandas as pd
import glob
from typing import Optional, Dict, Tuple



def calc_cross_corr_orth(z1: jax.Array, z2: jax.Array, eps: float = 1e-5):
    """
    批次级互相关解耦损失 (DSN / Barlow Twins 经典范式)
    惩罚 z1 与 z2 各维度在整个 Batch 样本上的 Pearson 互相关系数平方
    """
    B = z1.shape[0]
    
    # 1. 沿 Batch 维度中心化 (减均值)
    z1_c = z1 - jnp.mean(z1, axis=0, keepdims=True)
    z2_c = z2 - jnp.mean(z2, axis=0, keepdims=True)
    
    # 2. 沿 Batch 维度做方差归一化
    var1 = jnp.sqrt(jnp.var(z1_c, axis=0, keepdims=True) + eps)
    var2 = jnp.sqrt(jnp.var(z2_c, axis=0, keepdims=True) + eps)
    z1_norm = z1_c / var1
    z2_norm = z2_c / var2
    
    # 3. 计算互相关矩阵 C: 形状为 (d1, d2)
    corr_matrix = jnp.matmul(z1_norm.T, z2_norm) / B
    
    # 4. 惩罚所有相关系数的平方均值
    return jnp.mean(corr_matrix ** 2)



def compute_metrics_cpu(probs, labels):
    """在 CPU 上计算 AUC 和 KS"""
    from sklearn.metrics import roc_auc_score
    p = np.asarray(probs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    auc = roc_auc_score(y, p)
    desc = np.argsort(p)[::-1]
    y_sorted = y[desc]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    tpr = np.cumsum(y_sorted) / max(n_pos, 1)
    fpr = np.cumsum(1 - y_sorted) / max(n_neg, 1)
    ks = float(np.max(np.abs(tpr - fpr)))
    return auc, ks

def compute_loss_hard(
    outputs: Dict[str, jax.Array],
    y_st: jax.Array,
    y_lt: jax.Array,
    weights: Dict[str, float]
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    
    # 1. 主预测损失 (BCE 交叉熵)
    loss_task_st = jnp.mean(optax.sigmoid_binary_cross_entropy(outputs["logit_st"], y_st))
    loss_task_lt = jnp.mean(optax.sigmoid_binary_cross_entropy(outputs["logit_lt"], y_lt))
    loss_task = loss_task_st + loss_task_lt

    loss_orth = (
            calc_cross_corr_orth(outputs["z_s"], outputs["z_st"]) +
            calc_cross_corr_orth(outputs["z_s"], outputs["z_lt"]) +
            calc_cross_corr_orth(outputs["z_st"], outputs["z_lt"])
        )
    total_loss = loss_task + weights.get("w_orth", 0.0) * loss_orth

    metrics = {
        "loss_total": total_loss,
        "loss_task": loss_task,
        "loss_task_st": loss_task_st,
        "loss_task_lt": loss_task_lt,
        "loss_orth": loss_orth,
    }
    return total_loss, metrics


def calc_cross_corr_orth(z1: jax.Array, z2: jax.Array, eps: float = 1e-5):
    """
    批次级互相关解耦损失 (DSN / Barlow Twins 经典范式)
    惩罚 z1 与 z2 各维度在整个 Batch 样本上的 Pearson 互相关系数平方
    """
    B = z1.shape[0]
    
    # 1. 沿 Batch 维度中心化 (减均值)
    z1_c = z1 - jnp.mean(z1, axis=0, keepdims=True)
    z2_c = z2 - jnp.mean(z2, axis=0, keepdims=True)
    
    # 2. 沿 Batch 维度做方差归一化
    var1 = jnp.sqrt(jnp.var(z1_c, axis=0, keepdims=True) + eps)
    var2 = jnp.sqrt(jnp.var(z2_c, axis=0, keepdims=True) + eps)
    z1_norm = z1_c / var1
    z2_norm = z2_c / var2
    
    # 3. 计算互相关矩阵 C: 形状为 (d1, d2)
    corr_matrix = jnp.matmul(z1_norm.T, z2_norm) / B
    
    # 4. 惩罚所有相关系数的平方均值
    return jnp.mean(corr_matrix ** 2)


def compute_bin_edges(X: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """
    计算每个特征维度的分位数切分点 (模拟 LightGBM 的直方图分箱)
    返回: bin_edges 矩阵，形状为 (num_features, n_bins + 1)
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(X, quantiles, axis=0).T  # (D, n_bins + 1)
    return edges