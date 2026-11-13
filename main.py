import os
# ---- ★ 可复现性 (Mac M5 / Apple Silicon + Metal 后端) ----
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import pandas as pd
from datetime import datetime
import time

from train import run_10_seeds_experiment


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # 运行 10 随机种子基准实验 (与 read_lend.ipynb 配置严格一致)
    df_results, stats_summary = run_10_seeds_experiment(
        seeds=[42, 100, 2024, 777, 999, 200, 300, 400, 500, 600],
        epochs=100,
        batch_size=512,
        lr=5e-4,
        weight_decay=0.1,
        w_orth=0.1,
        patience=20
    )
