# Getting Started with CAAA

This guide walks you through setting up, understanding, running, and evaluating the CAAA (Context-Aware Anomaly Attribution) framework from scratch.

## Prerequisites

- **Python 3.9+** (tested on 3.10)
- **pip** package manager
- **Git** for version control
- ~2 GB disk space (for dependencies including PyTorch)
- GPU optional — the system auto-detects CUDA but runs fine on CPU

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/Xhadou/CAAA.git
cd CAAA

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Dependency overview

| Package | Purpose |
|---------|---------|
| `torch>=2.6.0` | Neural model (CAAA, LSTM autoencoder) |
| `scikit-learn>=1.3.0` | StandardScaler, RandomForest, metrics |
| `xgboost>=1.7.0` | XGBoost baseline |
| `lightgbm>=4.0.0` | LightGBM baseline |
| `catboost>=1.2` | CatBoost baseline |
| `tabpfn>=2.0` | TabPFN foundation model baseline |
| `scipy>=1.11.0` | Signal processing (Welch PSD), statistics |
| `shap>=0.42.0` | Feature importance explanations |
| `ruptures>=1.1.0` | Change point detection |
| `matplotlib`, `seaborn` | Visualization |

## 2. Repository Structure

```
CAAA/
├── src/                        # Core source code
│   ├── main.py                 # Unified entry point (train + evaluate)
│   ├── data_loader/            # Data generation and loading
│   │   ├── data_types.py       # ServiceMetrics, AnomalyCase dataclasses
│   │   ├── synthetic_generator.py  # Normal metric + load spike generation
│   │   ├── fault_generator.py  # 11 fault injection types with AR(1) signals
│   │   ├── dataset.py          # Combined dataset generation
│   │   ├── rcaeval_loader.py   # RCAEval benchmark parser
│   │   └── utils.py            # Per-service baseline profiles
│   ├── features/               # Feature extraction
│   │   ├── feature_schema.py   # Single source of truth: 44 features, 6 groups
│   │   ├── extractors.py       # FeatureExtractor class (raw metrics → 44-dim vector)
│   │   └── context_features.py # Context signal computation
│   ├── models/                 # All model implementations
│   │   ├── caaa_model.py       # CAAA neural model (proposed method)
│   │   ├── feature_encoder.py  # MLP + LayerNorm + GELU encoder
│   │   ├── context_module.py   # FiLM conditioning + confidence gating
│   │   ├── anomaly_detector.py # LSTM autoencoder (optional Stage 1)
│   │   ├── baseline.py         # RF, XGBoost, LightGBM, CatBoost, TabPFN
│   │   └── classifier.py       # Multi-backend sklearn classifier
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # AdamW + scheduler + early stopping + calibration
│   │   └── losses.py           # Context Consistency Loss, FocalLoss
│   ├── evaluation/             # Metrics and visualization
│   │   ├── metrics.py          # Accuracy, F1, MCC, PR-AUC, McNemar, bootstrap CI
│   │   └── visualization.py    # SHAP plots, confusion matrices, reliability diagrams
│   └── utils/                  # Utilities
│       └── __init__.py         # set_seed(), resolve_device()
├── scripts/                    # Runnable scripts
│   ├── demo.py                 # Quick demo (small dataset)
│   ├── train.py                # Full training pipeline
│   └── ablation.py             # Systematic ablation study
├── tests/                      # Test suite
├── configs/
│   └── config.yaml             # Default configuration
└── docs/
    └── GETTING_STARTED.md      # This file
```

## 3. Key Concepts

### What CAAA does

CAAA classifies anomalies detected in microservice systems as:
- **FAULT** (class 0) — a real system issue requiring attention
- **EXPECTED_LOAD** (class 1) — a legitimate workload spike (scheduled event, traffic peak)
- **UNKNOWN** (class 2) — insufficient confidence for automatic classification

The key insight: a CPU spike during a known marketing campaign is expected load, not a fault. CAAA uses context signals (scheduled events, deployments, time-of-day) to make this distinction.

### The 44-dimensional feature vector

Every anomaly case is converted to a 44-dim feature vector organized into 6 groups:

| Group | Dims | What it captures |
|-------|------|------------------|
| Workload (0–5) | 6 | Does the metric change correlate with workload? |
| Behavioral (6–11) | 6 | Fault signatures (sharp onset, cascading) vs smooth load ramps |
| Context (12–16) | 5 | External signals — the key innovation |
| Statistical (17–29) | 13 | Standard metric statistics (mean, std, max) |
| Service-Level (30–35) | 6 | Cross-service aggregation patterns |
| Extended (36–43) | 8 | Frequency-domain, graph-structural, rate-of-change |

The schema is defined in `src/features/feature_schema.py` — this is the single source of truth.

### Two-stage pipeline

1. **Stage 1 (optional):** LSTM autoencoder detects anomalous time windows
2. **Stage 2:** CAAA model classifies anomalies using FiLM-conditioned context integration

### Data sources

- **Synthetic** (default): Generated on-the-fly with 11 fault types, per-service baselines, and AR(1) autocorrelated fault signals. No downloads needed.
- **RCAEval**: Real-world microservice failure traces from Zenodo (Online Boutique, Sock Shop, Train Ticket).

## 4. Running Your First Experiment

### Quick demo (~30 seconds)

```bash
python scripts/demo.py --n-fault 20 --n-load 20 --epochs 20
```

This generates 40 synthetic cases, trains CAAA for 20 epochs, and prints metrics.

### Full training with baselines (~2 minutes)

```bash
python scripts/train.py --n-fault 100 --n-load 100 --epochs 50 --baseline
```

This trains the CAAA model and compares against a RandomForest baseline.

### Using the unified entry point

```bash
# CAAA neural model
python -m src.main --n-fault 50 --n-load 50 --model caaa

# Random Forest baseline
python -m src.main --n-fault 50 --n-load 50 --model random_forest

# XGBoost baseline
python -m src.main --n-fault 50 --n-load 50 --model xgboost

# With GPU (auto-detected by default)
python -m src.main --n-fault 50 --n-load 50 --model caaa --device auto
```

### Using RCAEval real-world data

```bash
# Download dataset (one-time)
python -m src.main --download-data --dataset RE1 --system online-boutique

# Train on real data
python -m src.main --data rcaeval --dataset RE1 --system online-boutique --model caaa
```

## 5. Ablation Study

The ablation study systematically compares 14 model variants across 7 metrics:

```bash
# Default: 5 runs, 50 fault + 50 load cases, 30 epochs
python scripts/ablation.py

# Larger experiment
python scripts/ablation.py --n-fault 100 --n-load 100 --epochs 50 --n-runs 10

# With cross-validation instead of random splits
python scripts/ablation.py --cv-folds 5

# With SHAP analysis and calibration plots
python scripts/ablation.py --shap --calibration

# On real data
python scripts/ablation.py --data rcaeval --dataset RE1 --system online-boutique --epochs 50
```

### Model variants evaluated

| Variant | What it tests |
|---------|---------------|
| Full CAAA | Complete model (proposed method) |
| CAAA + Contrastive | Supervised contrastive loss variant |
| No Context Features | Context features zeroed out |
| No Context Loss | Standard cross-entropy only |
| No Behavioral | Behavioral features zeroed out |
| Context Only | Only context features |
| Statistical Only | Only statistical features |
| Stat + Service-Level | Statistical + service-level features |
| Baseline RF | RandomForest on raw features |
| XGBoost | XGBoost on raw features |
| LightGBM | LightGBM on raw features |
| CatBoost | CatBoost on raw features |
| Rule-Based | Simple threshold rules |
| Naive | Always predicts FAULT |

### Metrics reported

- **Accuracy** — overall classification accuracy
- **F1** — weighted F1 score
- **F1 Macro** — macro-averaged F1 (treats classes equally)
- **MCC** — Matthews Correlation Coefficient (robust to class imbalance)
- **FP Rate** — false positive rate for FAULT class
- **Fault Recall** — proportion of real faults correctly identified (target: >90%)
- **FP Reduction** — reduction vs naive baseline (target: >40%)

Results are saved to `outputs/results/ablation_results.csv`.

## 6. Running Tests

```bash
# Fast tests (models + features, ~40 seconds)
python -m pytest tests/test_models.py tests/test_features.py -v

# Data loader tests (~3 minutes, generates many synthetic cases)
python -m pytest tests/test_data_loader.py -v

# Full suite (includes integration tests with training, ~10 minutes)
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test files

| File | What it tests | Speed |
|------|---------------|-------|
| `test_models.py` | Model forward pass, baselines, cross-validation | Fast |
| `test_features.py` | Feature extraction, schema, fault-vs-load differences | Fast |
| `test_data_loader.py` | Synthetic generation, hard scenarios | Medium |
| `test_integration.py` | End-to-end pipeline (data → features → train → evaluate) | Slow |
| `test_plan_modules.py` | Sklearn classifier, feature importance | Slow |
| `test_rcaeval_pipeline.py` | RCAEval data loading (requires downloaded data) | Slow |

## 7. Understanding the Training Pipeline

The training pipeline follows a 3-way stratified split to prevent data leakage:

```
Full Dataset
    │
    ├── 60% Train ──── Used for model training
    │
    ├── 20% Val ────── Used for early stopping + temperature calibration
    │
    └── 20% Test ───── Touched ONLY for final metric computation
```

### Training details

- **Optimizer:** AdamW (weight decay built in)
- **Scheduler:** ReduceLROnPlateau (reduces LR when validation loss plateaus)
- **Gradient clipping:** max_norm=1.0 (prevents exploding gradients)
- **Early stopping:** Monitors validation loss with configurable patience
- **Normalization:** StandardScaler fit on train set, applied to val/test (neural models only; tree baselines use raw features)
- **Loss:** Context Consistency Loss with label smoothing (0.1) + FocalLoss option

### GPU support

The system auto-detects CUDA availability:

```bash
python -m src.main --device auto   # Auto-detect (default)
python -m src.main --device cuda   # Force GPU
python -m src.main --device cpu    # Force CPU
```

## 8. Configuration

All parameters can be set via `configs/config.yaml` or CLI arguments (CLI takes precedence).

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 44 | Feature vector size (don't change unless modifying feature_schema.py) |
| `hidden_dim` | 64 | Hidden layer size in encoder and classifier |
| `epochs` | 50 | Training epochs |
| `learning_rate` | 0.001 | Initial learning rate |
| `early_stopping_patience` | 10 | Epochs to wait before stopping |
| `batch_size` | 32 | Training batch size |

## 9. Extending CAAA

### Adding a new feature

1. Add the feature name to `src/features/feature_schema.py` (in the appropriate group)
2. Implement extraction in `src/features/extractors.py`
3. Update `N_FEATURES` and `ALL_FEATURE_NAMES` in the schema
4. Update `input_dim` in `configs/config.yaml` and all `input_dim=44` references
5. Update tests with the new feature count

### Adding a new baseline

1. Add a class to `src/models/baseline.py` following the existing pattern (lazy import, `fit`/`predict`/`predict_proba` interface)
2. Export it in `src/models/__init__.py`
3. Add a runner function in `scripts/ablation.py`
4. Add the variant name to the `variants` list in ablation.py

### Adding a new fault type

1. Add a method to `src/data_loader/fault_generator.py`
2. Register it in the `FAULT_TYPES` list
3. Add test coverage in `tests/test_data_loader.py`

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from the repo root, or use `pip install -e .` |
| `torch.load` security warning | Already fixed — all calls use `weights_only=True` |
| Tests hang or laptop freezes | Run only fast tests: `pytest tests/test_models.py tests/test_features.py -v` |
| `ImportError: lightgbm/catboost/tabpfn` | Install: `pip install lightgbm catboost tabpfn` |
| CUDA out of memory | Use `--device cpu` or reduce `--batch-size` |
| Ablation takes too long | Reduce: `--n-fault 20 --n-load 20 --epochs 10 --n-runs 1` |

## 11. Performance Targets

| Metric | Target | What it means |
|--------|--------|---------------|
| FP Reduction | >40% | Fewer false alarms vs naive baseline |
| Fault Recall | >90% | Real faults are still caught |
| Overall Accuracy | >80% | Correct classification rate |

The ablation study (`scripts/ablation.py`) is the primary tool for verifying these targets across model variants and data configurations.
