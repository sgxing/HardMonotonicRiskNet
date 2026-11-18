import os
# ---- ★ 可复现性 (Mac M5 / Apple Silicon + Metal 后端) ----
#os.environ['PYTHONHASHSEED'] = '0'                       # Python dict/set 遍历顺序确定
#os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'   # 关闭 XLA 自动调优 (Metal 也适用)
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import jax
#jax.config.update('jax_default_matmul_precision', 'highest')  # float32 精度, 不用 bfloat16 近似
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from datetime import datetime
import time
import pandas as pd
import glob
from typing import Optional, Dict, Tuple


print(f"JAX version: {jax.__version__}")
print(jax.devices())


# ── ★ 方案一核心: 分段线性分箱嵌入 (Piecewise Linear Encoding, NeurIPS 2022) ────────
class PiecewiseLinearEncoding(nnx.Module):
    r"""
    分段线性分箱编码层 (Piecewise Linear Encoding, PLE)
    严格遵循论文公式 (5)、(6)、(7) 定义:
    
    1. 中间分箱 k = 2, ..., T - 1 (公式 5):
       e_k(x_j) = clip((x_j - b_{j, k-1}) / (b_{j, k} - b_{j, k-1}), 0.0, 1.0)
       
    2. 首个分箱 k = 1 (公式 6):
       e_1(x_j) = min(1, (x_j - b_{j, 0}) / (b_{j, 1} - b_{j, 0}))
       [左侧不下截断，允许负值线性外推]
       
    3. 末尾分箱 k = T (公式 7):
       e_T(x_j) = max(0, (x_j - b_{j, T-1}) / (b_{j, T} - b_{j, T-1}))
       [右侧不上截断，允许大于1线性外推]
       
    4. 分段线性重构与参数自适应:
       	ilde{x}_j = b_{j, 0} + \sum_{k=1}^T w_{j, k} \cdot e_k(x_j)
       初始化时 w_{j, k} = b_{j, k} - b_{j, k-1}, b_j = b_{j, 0}，在全实数域 R 上严格恒等映射 x_out ≡ x。
    """
    def __init__(self, bin_edges: jax.Array, rngs: nnx.Rngs):
        super().__init__()
        # bin_edges 形状: (D, T + 1)
        D, T_plus_1 = bin_edges.shape
        self.T = T_plus_1 - 1
        self.b_left = bin_edges[:, :-1]   # (D, T), 对应 [b_{j,0}, b_{j,1}, ..., b_{j,T-1}]
        self.b_right = bin_edges[:, 1:]   # (D, T), 对应 [b_{j,1}, b_{j,2}, ..., b_{j,T}]
        self.widths = jnp.maximum(self.b_right - self.b_left, 1e-5) # (D, T), 杜绝重合分位数导致除零
        
        # 初始斜率与截距满足恒等变换: f(x) = b_0 + \sum w_k e_k(x) ≡ x
        self.w = nnx.Param(self.widths)          # 初始斜率 = 区间宽度
        self.b = nnx.Param(self.b_left[:, 0])    # 初始截距 = 最小值 b_{j,0}

    def compute_encoding(self, x: jax.Array) -> jax.Array:
        """
        依据论文公式 (5)、(6)、(7) 计算分段线性编码张量 e ∈ R^(B, D, T)
        """
        # 归一化跨度比值: (B, D, T)
        ratio = (x[:, :, None] - self.b_left[None, :, :]) / self.widths[None, :, :]
        
        # 公式 (6): 首个分箱 (k = 1)
        e_1 = jnp.minimum(1.0, ratio[:, :, :1])
        
        # 公式 (5): 中间分箱 (k = 2, ..., T - 1)
        e_mid = jnp.clip(ratio[:, :, 1:-1], 0.0, 1.0)
        
        # 公式 (7): 末尾分箱 (k = T)
        e_T = jnp.maximum(0.0, ratio[:, :, -1:])
        
        # 沿分箱维度拼接得到完整的 (B, D, T) 编码
        return jnp.concatenate([e_1, e_mid, e_T], axis=-1)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, D)
        # 1. 严格依据公式 (5)(6)(7) 计算编码张量 e ∈ R^(B, D, T)
        e = self.compute_encoding(x)
        
        # 2. 分段线性映射累加: (B, D)
        delta_x = jnp.sum(e * self.w[...][None, :, :], axis=-1) + self.b[...][None, :]
        return delta_x




class HardMonotonicRiskNet(nnx.Module):
    """
    硬单调三路因果分解风控网络 (HardMonotonicRiskNet + PLE 分段切分增强)
    - 方案一: 输入端采用分段线性分箱嵌入 (PLE)，赋予神经网络拟合 GBDT 阶跃切分的能力
    - 严格硬单调增量头: logit_lt = logit_st + softplus(delta) >= logit_st
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_static_features: int,
        static_hidden: int = 128,
        bin_edges: Optional[jax.Array] = None,
        d_s: int = 64,
        d_st: int = 64,
        d_lt: int = 64,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # 1. 方案一核心: GBDT式分段线性切分层
        if bin_edges is not None:
            self.ple = PiecewiseLinearEncoding(jnp.array(bin_edges), rngs=rngs)
        else:
            self.ple = None

        # 2. 三路独立特征编码器 (吃进经过分段非线性切分增强后的特征)
        self.enc_s = nnx.Sequential(
            nnx.Linear(num_static_features, static_hidden, rngs=rngs), 
            nnx.relu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(static_hidden, d_s, rngs=rngs)
        )
        self.enc_st = nnx.Sequential(
            nnx.Linear(num_static_features, static_hidden, rngs=rngs), 
            nnx.relu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(static_hidden, d_st, rngs=rngs)
        )
        self.enc_lt = nnx.Sequential(
            nnx.Linear(num_static_features, static_hidden, rngs=rngs), 
            nnx.relu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(static_hidden, d_lt, rngs=rngs)
        )

        # 3. 严格硬单调预测头
        self.head_short = nnx.Sequential(
            nnx.Linear(d_s + d_st, 64, rngs=rngs), 
            nnx.relu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(64, 1, rngs=rngs)
        )
        self.head_long_delta = nnx.Sequential(
            nnx.Linear(d_s + d_lt, 64, rngs=rngs), 
            nnx.relu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(64, 1, rngs=rngs)
        )

    def __call__(self, x, training=False):
        for _, m in nnx.iter_modules(self):
            if isinstance(m, nnx.Dropout):
                m.deterministic = not training
        
        # 1. GBDT式分段非线性切分增强
        x_in = self.ple(x) if self.ple is not None else x
        # 2. 三路因果分解独立编码
        z_s = self.enc_s(x_in)
        z_st = self.enc_st(x_in)
        z_lt = self.enc_lt(x_in)
       
        # 3. 严格硬单调预测机制
        logit_st = self.head_short(jnp.concatenate([z_s, z_st], axis=-1)).squeeze(-1)
        raw_delta = self.head_long_delta(jnp.concatenate([z_s, z_lt], axis=-1)).squeeze(-1)
        delta_logit = jax.nn.softplus(raw_delta)
        
        # 长期 logit 恒严格大于等于 短期 logit
        logit_lt = logit_st + delta_logit
        
        # 概率映射 (恒满足 pred_lt >= pred_st，消除硬件浮点舍入误差)
        pred_st = jax.nn.sigmoid(logit_st)
        pred_lt = jnp.maximum(pred_st, jax.nn.sigmoid(logit_lt))

        return {
            "z_s": z_s, "z_st": z_st, "z_lt": z_lt,
            "logit_st": logit_st, "logit_lt": logit_lt,
            "pred_st": pred_st, "pred_lt": pred_lt,
        }
        
class BaseMTL(nnx.Module):
    """
    标准共享底层多任务学习网络 (Standard Shared-Bottom MTL, Caruana 1997)
    
    架构设计:
    1. 共享底层主干 (Shared Bottom Trunk):
       多层 MLP 将原始输入特征投影至统一的全局共享隐空间 h = SharedTrunk(x);
    2. 独立任务塔 (Independent Task Towers):
       - Tower MOB2: 针对短期违约任务的专用多层输出头 h -> logits_mob2;
       - Tower MOB6: 针对长期累积违约任务的专用多层输出头 h -> logits_mob6;
    3. 特点: 
       各任务独立预测，无任何硬单调约束或正交先验，作为纯正的经典多任务基线 (Baseline)。
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_static_features: int,
        shared_hidden: int = 128,
        head_hidden: int = 64,
        dropout_rate: float = 0.5,
        **kwargs  # 兼容外部传入的多余超参 (如 static_hidden, embed_hidden, num_experts 等)
    ):
        super().__init__()
        hidden_dim = kwargs.get("static_hidden", shared_hidden)
        
        # 1. 共享底层主干 (Shared Bottom)
        self.shared_fc1 = nnx.Linear(num_static_features, hidden_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        
        # 2. 短期 (MOB2) 独立任务输出塔
        self.head_mob2_fc = nnx.Linear(hidden_dim, head_hidden, rngs=rngs)
        self.head_mob2_out = nnx.Linear(head_hidden, 1, rngs=rngs)
        
        # 3. 长期 (MOB6) 独立任务输出塔
        self.head_mob6_fc = nnx.Linear(hidden_dim, head_hidden, rngs=rngs)
        self.head_mob6_out = nnx.Linear(head_hidden, 1, rngs=rngs)

    def __call__(self, x_static: jax.Array, training: bool = False):
        # 1. 共享底层特征抽取
        h = nnx.relu(self.shared_fc1(x_static))
        h = self.dropout1(h, deterministic=not training)
        # 2. 独立任务塔并行预测
        h_mob2 = nnx.relu(self.head_mob2_fc(h))
        logits_mob2 = self.head_mob2_out(h_mob2)
        
        h_mob6 = nnx.relu(self.head_mob6_fc(h))
        logits_mob6 = self.head_mob6_out(h_mob6)
        
        return logits_mob2, logits_mob6
    
# MLP
class HybridCreditModel_STL(nnx.Module):
    def __init__(self, rngs, num_static_features, embed_hidden=128, dropout_rate=0.5):
        self.input_dim = num_static_features
        self.deep_in_1st = nnx.Linear(self.input_dim, embed_hidden, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.deep_in_2nd = nnx.Linear(embed_hidden, embed_hidden // 2, rngs=rngs)
        self.deep_out = nnx.Linear(embed_hidden // 2, 1, rngs=rngs)

    def __call__(self, x_static, training=False):
        h = nnx.relu(self.deep_in_1st(x_static))
        h = self.dropout(h, deterministic=not training)
        h = nnx.relu(self.deep_in_2nd(h))
        h = self.dropout(h, deterministic=not training)
        logits = self.deep_out(h)
        return logits
    
    
    
class NnetSurvival(nnx.Module):
    r"""
    离散时间神经网络生存模型 (Nnet-survival / Discrete-Time Hazard Model)
    - 输入特征 x ∈ ℝ^D
    - 预测每个离散时间区间的条件危险率 (Conditional Hazard Rate):
        h_1(x) = σ(logit_1(x)) = P(T ≤ MOB2 | x)
        h_2(x) = σ(logit_2(x)) = P(MOB2 < T ≤ MOB6 | T > MOB2, x)
    - 依据生存分析链式乘法法则计算累积违约概率:
        P(MOB2=1) = h_1(x)
        P(MOB6=1) = 1 - S_2(x) = 1 - (1 - h_1(x)) * (1 - h_2(x))
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_static_features: int,
        shared_hidden: int = 128,
        head_hidden: int = 64,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        super().__init__()
        # 1. 共享多层特征主干
        self.fc1 = nnx.Linear(num_static_features, shared_hidden, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        
        # 2. 离散区间条件危险率 (Hazard) 预测头
        # h1: [0, MOB2] 早期违约险率
        self.head_h1_fc = nnx.Linear(shared_hidden, head_hidden, rngs=rngs)
        self.head_h1_out = nnx.Linear(head_hidden, 1, rngs=rngs)
        
        # h2: (MOB2, MOB6] 中长期条件违约险率
        self.head_h2_fc = nnx.Linear(shared_hidden, head_hidden, rngs=rngs)
        self.head_h2_out = nnx.Linear(head_hidden, 1, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool = False):
        deterministic = not training
        h = nnx.relu(self.fc1(x))
        h = self.dropout(h, deterministic=deterministic)
        
        # 计算两区间的对数险率 logits
        logit_h1 = self.head_h1_out(nnx.relu(self.head_h1_fc(h))).squeeze(-1)
        logit_h2 = self.head_h2_out(nnx.relu(self.head_h2_fc(h))).squeeze(-1)
        
        # 条件危险率 hazard ∈ (0, 1)
        h1 = jax.nn.sigmoid(logit_h1)
        h2 = jax.nn.sigmoid(logit_h2)
        
        # 累积违约率: P2 = h1, P6 = 1 - (1 - h1)(1 - h2)
        p_mob2 = h1
        p_mob6 = 1.0 - (1.0 - h1) * (1.0 - h2)
        
        return logit_h1, logit_h2, p_mob2, p_mob6
    
    
# TabM 核心单元: 线性层
    
class LinearBatchEnsemble(nnx.Module):
    """
    BatchEnsemble 线性层 (TabM 核心构成单元, Gorishniy et al., ICLR 2025)
    - 共享骨干主权重: W in R^(d_in x d_out)
    - 成员特异性秩-1 缩放向量: r in R^(k x d_in), s in R^(k x d_out)
    - 成员特异性偏置: b in R^(k x d_out)
    - 前向计算: y_i = ((x_i * r_i) W) * s_i + b_i
    """
    def __init__(self, in_features: int, out_features: int, k: int, rngs: nnx.Rngs, is_first: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.is_first = is_first

        # 1. 共享权重矩阵 W (LeCun / He 均匀初始化)
        bound = 1.0 / np.sqrt(in_features)
        self.w = nnx.Param(jax.random.uniform(rngs.params(), (in_features, out_features), minval=-bound, maxval=bound))

        # 2. 成员特异性 rank-1 缩放向量 r (官方 tabm_init)
        if is_first:
            # 首层: Rademacher {-1, +1} 随机符号分布 (p=0.5)
            bernoulli = jax.random.bernoulli(rngs.params(), p=0.5, shape=(k, in_features))
            self.r = nnx.Param(bernoulli.astype(jnp.float32) * 2.0 - 1.0)
        else:
            # 深层: 全 1 初始化
            self.r = nnx.Param(jnp.ones((k, in_features), dtype=jnp.float32))

        # 3. 成员特异性 rank-1 缩放向量 s (全 1 初始化)
        self.s = nnx.Param(jnp.ones((k, out_features), dtype=jnp.float32))

        # 4. 成员特异性偏置 b (均值一致的轻微随机偏置)
        b_init = jax.random.uniform(rngs.params(), (out_features,), minval=-bound, maxval=bound)
        self.b = nnx.Param(jnp.broadcast_to(b_init[None, :], (k, out_features)).copy())

    def __call__(self, x: jax.Array) -> jax.Array:
        # x 形状: (B, k, in_features)
        x_scaled = x * self.r[None, :, :]
        h = jnp.matmul(x_scaled, self.w[...])
        out = h * self.s[None, :, :] + self.b[None, :, :]
        return out


class TabMBlock(nnx.Module):
    """
    TabM 官方前馈块: LinearBatchEnsemble + ReLU + Dropout
    (避免跨成员共享 LayerNorm 强行抹平各子模型在隐空间中的多样性)
    """
    def __init__(self, d_hidden: int, k: int, dropout_rate: float, rngs: nnx.Rngs):
        super().__init__()
        self.linear = LinearBatchEnsemble(d_hidden, d_hidden, k, rngs, is_first=False)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        h = jax.nn.relu(self.linear(x))
        if training:
            h = self.dropout(h)
        return h


class TabMDualTargetNet(nnx.Module):
    """
    TabM 双目标信贷风控预测网络 (ICLR 2025 优化版)
    - 特征投影层: 首层应用 tabm_init (Rademacher 随机符号)
    - 隐层骨干: d_hidden = 128 (对齐 read_lend.ipynb 运行配置)
    - 任务专属头: 两组独立的 BatchEnsemble 线性分类头 (d_hidden -> 1)
    - 集成汇聚: 经由 Sigmoid 变换后在成员维度求均值 (Ensemble Mean Probability)
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_features: int,
        d_hidden: int = 128,
        n_layers: int = 3,
        k: int = 16,
        dropout_rate: float = 0.5,
        bin_edges=None,
    ):
        super().__init__()
        
        if bin_edges is not None:
            self.ple = PiecewiseLinearEncoding(jnp.array(bin_edges), rngs=rngs)
        else:
            self.ple = None
                    
        self.k = k
        self.d_hidden = d_hidden

        # 1. 输入投影层 (含首层 Rademacher tabm_init)
        self.input_layer = LinearBatchEnsemble(num_features, d_hidden, k, rngs, is_first=True)
        self.input_dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # 2. 深度 BatchEnsemble 骨干 (n_layers - 1 个中间隐层)
        self.blocks = nnx.List([
            TabMBlock(d_hidden, k, dropout_rate, rngs)
            for _ in range(n_layers - 1)
        ])

        # 3. 双目标专属预测头
        self.head_mob2 = LinearBatchEnsemble(d_hidden, 1, k, rngs, is_first=False)
        self.head_mob6 = LinearBatchEnsemble(d_hidden, 1, k, rngs, is_first=False)

    def __call__(self, x: jax.Array, training: bool = False):
        for _, m in nnx.iter_modules(self):
            if isinstance(m, nnx.Dropout):
                m.deterministic = not training
        x_in = self.ple(x) if self.ple is not None else x
        # 广播复制输入特征至 k 份集成流: (B, D) -> (B, k, D)
        x_k = jnp.repeat(x_in[:, None, :], self.k, axis=1)

        # 骨干前向
        h = jax.nn.relu(self.input_layer(x_k))
        if training:
            h = self.input_dropout(h)

        for block in self.blocks:
            h = block(h, training=training)

        # 双目标输出头: (B, k, 1) -> squeeze(-1) -> (B, k)
        logits_mob2_k = self.head_mob2(h).squeeze(-1)
        logits_mob6_k = self.head_mob6(h).squeeze(-1)

        p_mob2_k = jax.nn.sigmoid(logits_mob2_k)
        p_mob6_k = jax.nn.sigmoid(logits_mob6_k)

        # 成员均值集成汇聚预测 (Ensemble Mean Probability)
        pred_mob2 = jnp.mean(p_mob2_k, axis=1)
        pred_mob6 = jnp.mean(p_mob6_k, axis=1)

        return {
            "logits_mob2_k": logits_mob2_k,
            "logits_mob6_k": logits_mob6_k,
            "p_mob2_k": p_mob2_k,
            "p_mob6_k": p_mob6_k,
            "pred_mob2": pred_mob2,
            "pred_mob6": pred_mob6,
        }
        
        
class ExtractionLayer(nnx.Module):
    r"""
    PLE / CGC 抽取层 (Customized Gate Control Extraction Layer)
    包含: 任务专属专家 (Task-specific Experts) + 共享专家 (Shared Experts) + 定制门控路由 (Task-specific Gates)
    """
    def __init__(
        self,
        in_dim: int,
        expert_hidden: int,
        num_task1_experts: int,
        num_task2_experts: int,
        num_shared_experts: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.5,
        is_last_layer: bool = False,
    ):
        super().__init__()
        self.is_last_layer = is_last_layer
        self.num_task1_experts = num_task1_experts
        self.num_task2_experts = num_task2_experts
        self.num_shared_experts = num_shared_experts

        def make_expert():
            return nnx.Sequential(
                nnx.Linear(in_dim, expert_hidden, rngs=rngs),
                nnx.relu,
                nnx.Dropout(dropout_rate, rngs=rngs),
                nnx.Linear(expert_hidden, expert_hidden, rngs=rngs),
                nnx.relu,
            )

        # 1. 专家网络集合 (使用 nnx.List 保证 PyTree 正确追踪)
        self.task1_experts = nnx.List([make_expert() for _ in range(num_task1_experts)])
        self.task2_experts = nnx.List([make_expert() for _ in range(num_task2_experts)])
        self.shared_experts = nnx.List([make_expert() for _ in range(num_shared_experts)])

        # 2. 定制门控网络
        # Gate 1: 仅在 [Task1 Experts, Shared Experts] 之间路由
        self.gate_task1 = nnx.Linear(in_dim, num_task1_experts + num_shared_experts, rngs=rngs)
        # Gate 2: 仅在 [Task2 Experts, Shared Experts] 之间路由
        self.gate_task2 = nnx.Linear(in_dim, num_task2_experts + num_shared_experts, rngs=rngs)

        # 若非最后一层，保留共享门控汇聚全体专家以抽取高阶泛化语义
        if not is_last_layer:
            self.gate_shared = nnx.Linear(in_dim, num_task1_experts + num_shared_experts + num_task2_experts, rngs=rngs)
        else:
            self.gate_shared = None

    def __call__(self, x_task1: jax.Array, x_task2: jax.Array, x_shared: Optional[jax.Array] = None):
        if x_shared is None:
            x_shared = x_task1

        # 计算所有专家的隐层表征: 各 (B, expert_hidden)
        out_t1 = [exp(x_task1) for exp in self.task1_experts]
        out_t2 = [exp(x_task2) for exp in self.task2_experts]
        out_shared = [exp(x_shared) for exp in self.shared_experts]

        # Gate 1 融合: [Task1 专家, 共享专家]
        cand_t1 = jnp.stack(out_t1 + out_shared, axis=1) # (B, N1 + Ns, H)
        w_t1 = jax.nn.softmax(self.gate_task1(x_task1), axis=-1)
        rep_t1 = jnp.sum(cand_t1 * w_t1[:, :, None], axis=1)

        # Gate 2 融合: [Task2 专家, 共享专家]
        cand_t2 = jnp.stack(out_t2 + out_shared, axis=1) # (B, N2 + Ns, H)
        w_t2 = jax.nn.softmax(self.gate_task2(x_task2), axis=-1)
        rep_t2 = jnp.sum(cand_t2 * w_t2[:, :, None], axis=1)

        if self.is_last_layer:
            return rep_t1, rep_t2
        else:
            # Gate Shared 融合: 全体专家 [Task1, Shared, Task2]
            cand_shared = jnp.stack(out_t1 + out_shared + out_t2, axis=1) # (B, N1 + Ns + N2, H)
            w_shared = jax.nn.softmax(self.gate_shared(x_shared), axis=-1)
            rep_shared = jnp.sum(cand_shared * w_shared[:, :, None], axis=1)
            return rep_t1, rep_t2, rep_shared


class PLE_MTL(nnx.Module):
    r"""
    渐进式分层抽取多任务学习网络 (Progressive Layered Extraction, PLE / CGC)
    论文出处: Tang et al. (RecSys 2020) "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations"
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_static_features: int,
        expert_hidden: int = 128,
        head_hidden: int = 64,
        num_task1_experts: int = 2,
        num_task2_experts: int = 2,
        num_shared_experts: int = 2,
        num_layers: int = 1,
        bin_edges: Optional[jax.Array] = None,
        d_embedding: Optional[int] = None,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_embedding = d_embedding

        # 1. 可选分段线性切分层 (Piecewise Linear Encoding)
        if bin_edges is not None:
            self.ple = PiecewiseLinearEncoding(jnp.asarray(bin_edges), rngs=rngs)
            in_dim = bin_edges.shape[0]
        else:
            self.ple = None
            in_dim = num_static_features

        # 2. 渐进式多层抽取 (PLE Layers)
        layers = []
        curr_in = in_dim
        for l in range(num_layers):
            is_last = (l == num_layers - 1)
            layer = ExtractionLayer(
                in_dim=curr_in,
                expert_hidden=expert_hidden,
                num_task1_experts=num_task1_experts,
                num_task2_experts=num_task2_experts,
                num_shared_experts=num_shared_experts,
                rngs=rngs,
                dropout_rate=dropout_rate,
                is_last_layer=is_last,
            )
            layers.append(layer)
            curr_in = expert_hidden
        self.layers = nnx.List(layers)

        # 3. 独立任务预测塔 (Task Towers)
        self.head_mob2 = nnx.Sequential(
            nnx.Linear(expert_hidden, head_hidden, rngs=rngs),
            nnx.relu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(head_hidden, 1, rngs=rngs),
        )
        self.head_mob6 = nnx.Sequential(
            nnx.Linear(expert_hidden, head_hidden, rngs=rngs),
            nnx.relu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(head_hidden, 1, rngs=rngs),
        )

    def __call__(self, x_static: jax.Array, training: bool = False):
        for _, m in nnx.iter_modules(self):
            if isinstance(m, nnx.Dropout):
                m.deterministic = not training

        # 1. 输入特征预处理 (若启用分箱编码则重构/投影)
        if self.ple is not None:
            h0 = self.ple(x_static)
            if self.d_embedding is not None:
                h0 = h0.reshape((h0.shape[0], -1))
        else:
            h0 = x_static

        # 2. 逐层抽取
        h_t1, h_t2, h_shared = h0, h0, h0
        for layer in self.layers:
            if layer.is_last_layer:
                h_t1, h_t2 = layer(h_t1, h_t2, h_shared)
            else:
                h_t1, h_t2, h_shared = layer(h_t1, h_t2, h_shared)

        # 3. 独立任务塔输出 logits
        logits_mob2 = self.head_mob2(h_t1)
        logits_mob6 = self.head_mob6(h_t2)

        return logits_mob2, logits_mob6


__all__ = [
    "PiecewiseLinearEncoding",
    "HardMonotonicRiskNet",
    "BaseMTL",
    "HybridCreditModel_STL",
    "NnetSurvival",
    "LinearBatchEnsemble",
    "TabMBlock",
    "TabMDualTargetNet",
    "ExtractionLayer",
    "PLE_MTL",
]
