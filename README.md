# HardMonotonicRiskNet: A Monotonically Constrained Expert System for Multi-Horizon Credit Risk Assessment

## 1. Environment Setup

```bash
pip install -r requirements.txt
```
*(For Metal acceleration on Apple Silicon Macs, optionally `install pip install jax-mps`)*.

---

## 2. Core Experiments and Benchmark Evaluation Commands

```bash
# 1. Run the proposed HardMonotonicRiskNet by default (across 10 seeds) and perform two-level interpretability attribution (Cases A and B).
python3.12 main.py

# 2. Evaluate a specific baseline model individually (e.g., HistGBDT gradient boosted trees).
python3.12 main.py --model hist_gbdt

# 3.Evaluate the mlp_stl individually
python3.12 main.py --model mlp_stl

# 4.Evaluate the discrete-time survival model NnetSurvival individually.
python3.12 main.py --model nnet_survival

# 5.Evaluate the multi-task learning network BaseMTL individually (supports customizing the soft monotonicity penalty weight via --w-reg).
python3.12 main.py --model base_mtl --w-reg 5.0

# 6.Evaluate the multi-task learning network PLE-MTL individually
python3.12 main.py --model ple_mtl

# 7.Specify a subset of random seeds and training epochs (e.g., a quick run with Seeds 42 and 100).
python3.12 main.py --model tabm --seeds 42,100 --epochs 50

# 8.Run the full-scale benchmark comparison across all 7 models in one command and generate the publication-ready summary table.
python3.12 main.py --model all
```

---

## 3. Automated Hyperparameter Tuning for HistGBDT (RandomizedSearchCV + Early Stopping on External Validation Set)

```bash
# Perform randomized hyperparameter search for both MOB2 and MOB6 simultaneously, with tuning results automatically persisted and mounted to the model.
python3.12 main.py --tune-hgbdt --tune-n-iter 100

# Run hyperparameter tuning individually for either MOB2 or MOB6.
python3.12 main.py --tune-hgbdt --tune-task mob2 --tune-n-iter 100
python3.12 main.py --tune-hgbdt --tune-task mob6 --tune-n-iter 100
```
