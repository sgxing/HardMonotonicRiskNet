# HardMonotonicRiskNet: 硬单调因果解耦信贷风控跨期预测系统

## 1. 环境准备

```bash
pip install -r requirements.txt
```
*(若在 Mac Apple Silicon 芯片上使用 Metal 加速，可按需安装 `pip install jax-mps`)*

---

## 2. 核心实验与基准评测运行命令

```bash
# 1. 默认运行核心提出的 HardMonotonicRiskNet (10 种子) 并执行两级可解释性 Case A/B 归因
python3.12 main.py

# 2. 单独评测指定基准模型 (例如 HistGBDT 梯度提升树)
python3.12 main.py --model hist_gbdt

# 3. 单独评测指定MLP模型
python3.12 main.py --model mlp_stl

# 4. 单独评测离散时间生存模型 NnetSurvival
python3.12 main.py --model nnet_survival

# 5. 单独评测多任务学习网络 BaseMTL (支持通过 --w-reg 自定义软单调惩罚权重) 或 PLE-MTL
python3.12 main.py --model base_mtl --w-reg 5.0

# 6. 单独评测多任务学习网络 PLE-MTL
python3.12 main.py --model ple_mtl

# 7. 指定部分随机种子与训练轮数 (例如快速运行 Seed 42, 100)
python3.12 main.py --model tabm --seeds 42,100 --epochs 50

# 8. 一键运行全部 7 个模型的大规模 Benchmark 对比并生成顶刊论文总表
python3.12 main.py --model all
```

---

## 3. HistGBDT 自动化超参数调优 (RandomizedSearchCV + 外部验证集早停截断)

```bash
# 同时对 MOB2 和 MOB6 执行随机超参数搜索，调优结果自动持久化并挂载至模型
python3.12 main.py --tune-hgbdt --tune-n-iter 100

# 仅对 MOB2 或 MOB6 单独调优
python3.12 main.py --tune-hgbdt --tune-task mob2 --tune-n-iter 100
python3.12 main.py --tune-hgbdt --tune-task mob6 --tune-n-iter 100
```
