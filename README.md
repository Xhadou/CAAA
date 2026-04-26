# CAAA: Context-Aware Anomaly Attribution

> A research framework for false positive reduction in cloud microservice anomaly detection through context-aware classification.

**New here?** See the **[Getting Started Guide](GETTING_STARTED.md)** for step-by-step setup and experiment instructions.

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Research Questions](#research-questions)
- [Novel Contributions](#novel-contributions)
- [Architecture](#architecture)
- [Feature Vector](#feature-vector)
- [Data Sources](#data-sources)
- [Performance Targets](#performance-targets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Analyzing Results](#analyzing-results)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Related Work](#related-work)
- [Citation](#citation)
- [License](#license)

---

## Research Motivation

Modern cloud-native applications built on microservice architectures generate a high volume of monitoring alerts. Production deployments report **false positive rates of 6–28%**, and even state-of-the-art root cause analysis methods achieve only **46–63% localization accuracy** on large-scale systems ([RCAEval, WWW 2025](https://zenodo.org/records/14590730)). Alert fatigue from false positives leads operators to ignore or delay responses, directly undermining the reliability benefits that monitoring is supposed to provide.

A core reason for this gap is that existing anomaly detection methods lack **contextual awareness**. A sudden spike in CPU utilization and request latency looks identical to an anomaly detector whether it is caused by a cascading fault or by a planned marketing campaign driving a legitimate traffic surge. Current approaches treat all anomalous-looking patterns as faults, producing false alerts whenever normal operational events (scheduled load tests, auto-scaling events, time-of-day traffic peaks, or recent deployments) cause metric deviations.

CAAA addresses this gap by integrating external context signals — scheduled events, deployment history, and temporal seasonality — directly into the anomaly classification pipeline. Rather than simply detecting anomalies, CAAA **attributes** them: classifying each anomaly as a **FAULT** (actual system issue), **EXPECTED_LOAD** (legitimate workload spike), or **UNKNOWN** (insufficient confidence for automatic classification).

## Research Questions

1. **RQ1**: Can integrating operational context signals (scheduled events, deployments, time-of-day patterns) into anomaly classification reduce false positives by >40% while maintaining >90% fault recall?
2. **RQ2**: How does the proposed Context Consistency Loss compare to standard cross-entropy for training context-aware classifiers?
3. **RQ3**: Which context features contribute most to false positive reduction, and how does performance degrade as context availability decreases?

## Novel Contributions

1. **Context-Aware Anomaly Attribution** — The first framework to explicitly distinguish workload-induced anomalies from actual faults using context signals within a unified classification pipeline. Supports external context (operational metadata) and comparison context (counterfactual baseline deviation).
2. **Context Consistency Loss** — A composite loss function with gated unknown-context penalty and TADAM-style delta regularization, preventing FiLM noise amplification on small datasets.
3. **Context Integration Module** — FiLM conditioning with confidence gating (skeptical initialization, bias=-1.0) and feature-group dropout for noise robustness.
4. **Piecewise Linear Embeddings (PLE)** — Quantile-based bin encodings that give the MLP axis-aligned threshold capability similar to tree splits, closing the neural-tree gap on tabular data.
5. **CAAA+CatBoost Hybrid** — FiLM-conditioned neural embeddings concatenated with raw features, classified by CatBoost. The first work combining FiLM conditioning with tree-based downstream classifiers.
6. **Evaluation Methodology** — Severity-tiered synthetic data with disguised faults, pre-injection split for honest RCAEval evaluation, counterfactual baselines, and label leakage prevention via randomized context assignment.

## Architecture

CAAA uses an optional two-stage pipeline:

```
                          ┌─────────────────────────────────────────────┐
                          │         Stage 2: Anomaly Attribution        │
 Metrics ──► [Stage 1] ──►  ┌───────────┐  ┌───────────┐  ┌──────────┐  │
 (optional   Anomaly      │ │  Feature  │─►│ Context   │─►│ Classif. │──► FAULT / EXPECTED_LOAD / UNKNOWN
  LSTM-AE)   Detection    │ │  Encoder  │  │ Module    │  │   Head   │  │
                          │ │(MLP+LN)   │  │(FiLM+Gate)│  │  (GELU)  │  │
                          │ │ 44d → 64d │  │ 5 context │  │ 2-class  │  │
                          └─────────────────────────────────────────────┘
```

### Stage 1 (Optional): Anomaly Detection

An LSTM autoencoder trained on normal (expected-load) metrics identifies anomalous time windows via reconstruction error. The pre-filter checks **all services** in each case and takes the maximum anomaly score — a fault in any service triggers detection. Only anomalous windows proceed to Stage 2. Enable with the `--anomaly-detector` flag.

### Stage 2: Anomaly Attribution

The core CAAA model classifies detected anomalies:

| Component | Description |
|-----------|-------------|
| **Feature Encoder** | 2-layer MLP with LayerNorm and GELU, projecting 44-dim feature vectors into 64-dim representations |
| **Context Integration Module** | FiLM (Feature-wise Linear Modulation) conditioning with enriched confidence gating (3-feature gate) |
| **Classification Head** | 2-class output (FAULT vs EXPECTED_LOAD) with post-hoc UNKNOWN assignment via confidence thresholding |

#### Dual Context Processing

The full 44-dim feature vector (including context dims 12–16) is passed through the Feature Encoder so that the encoder can learn joint representations capturing interactions between context and metric features. Context features are *also* sliced out separately and fed into the Context Integration Module for explicit attention and confidence gating. This dual processing is intentional: the encoder builds a context-entangled hidden representation, while the gating module controls how much the explicit context signal modulates the final prediction.

#### Confidence Gate Initialization

The confidence gate is initialized with `bias=-1.0`, producing an initial sigmoid output of ~0.27. This makes the model **skeptical of context by default** -- it must learn to trust the context signal through training, preventing over-reliance on noisy context features early in training (Sheth et al., NeurIPS 2022 ICBINB Workshop).

#### Piecewise Linear Embeddings (PLE)

Each scalar feature is transformed into an 8-bin encoding via quantile boundaries computed from training data. This gives the MLP axis-aligned threshold capability similar to tree splits, addressing the fundamental sample-efficiency gap between neural networks and trees on small tabular data (Gorishniy et al., NeurIPS 2022).

### Context Consistency Loss

The loss function combines four terms:

- **Cross-entropy classification loss** (with label smoothing 0.1)
- **Context consistency penalty** -- penalizes predictions contradicting context signals, weighted by clamped confidence (floor=0.3) to ensure learning signal even for low-confidence cases
- **Gated unknown-context penalty** -- penalizes EXPECTED_LOAD predictions above 0.7 confidence when no context explains the anomaly (configurable via `--loss-variant`)
- **Confidence calibration loss** -- entropy regularization guided by context confidence scores

Three loss variants are supported via `--loss-variant`:
- `gated` (default): Unknown penalty fires only when load_prob > 0.7
- `clamp`: No unknown penalty, clamp-only consistency
- `full`: Original aggressive penalty (for comparison)

**TADAM-style regularization:** When using `--film-mode tadam`, the FiLM gamma is constrained to `1 + delta` with L2 penalty on delta, keeping the multiplicative path near identity and preventing noise amplification.

**Training methodology:**

- 3-way stratified split (60% train / 20% val / 20% test)
- AdamW optimizer (weight_decay=1e-3) with ReduceLROnPlateau scheduler and gradient clipping (max_norm=1.0)
- NaNSafeScaler normalization for neural models (tree baselines use raw features)
- Feature-group dropout: context features (slots 12-16) zeroed with 30% probability during training
- GPU auto-detection (`--device auto|cpu|cuda`)

## Feature Vector

The system extracts a 44-dimensional feature vector organized into 6 groups, defined centrally in `src/features/feature_schema.py`:

| Group | Dims | Features | Purpose |
|-------|------|----------|---------|
| **Workload** | 0–5 | `global_load_ratio`, `cpu_request_correlation`, `cross_service_sync`, `error_rate_delta`, `latency_cpu_correlation`, `change_point_magnitude` | Characterize whether metric changes correlate with workload |
| **Behavioral** | 6–11 | `onset_gradient`, `peak_duration`, `cascade_score`, `recovery_indicator`, `affected_service_ratio`, `variance_change_ratio` | Capture fault propagation signatures vs. smooth load ramps |
| **Context** | 12–16 | `cpu_deviation`, `error_rate_ratio`, `correlation_shift`, `latency_deviation`, `baseline_confidence` | Context signals derived by comparing the anomalous window to a counterfactual baseline. On synthetic data, external context (`event_active`, etc.) is used instead via `--context-mode external`. |
| **Statistical** | 17–29 | Mean/std of CPU, memory, requests, errors, latency, network; `max_error_rate` | Standard metric statistics |
| **Service-Level** | 30–35 | `n_services`, `max_cpu_service_ratio`, `max_error_service_ratio`, `cpu_spread`, `error_spread`, `latency_spread` | Cross-service aggregation patterns |
| **Extended** | 36–43 | `spectral_entropy`, `high_freq_energy_ratio`, `dominant_frequency`, `network_asymmetry`, `graph_anomaly_centrality`, `anomaly_spread`, `max_cpu_derivative`, `error_rate_slope` | Frequency-domain, graph-structural, and rate-of-change features |

## Data Sources

CAAA supports two data modes:

### Synthetic Data (Default)

Generated on-the-fly with no downloads required. Includes:

- Normal microservice metric generation with realistic temporal patterns
- **11 fault injection types**: CPU spike, memory leak, latency injection, error burst, cascading failure, network partition, disk I/O saturation, thread pool exhaustion, connection pool leak, garbage collection storm, and downstream timeout
- Load spike generation with configurable intensity and correlation patterns
- Adversarial and hard-negative scenarios for robust evaluation

### RCAEval Benchmark Data

Real-world microservice failure traces from [Zenodo](https://zenodo.org/records/14590730) covering three systems and 735 fault cases across 11 fault types:

| System | Services | Description |
|--------|----------|-------------|
| Online Boutique | 12 | Google's microservice demo app |
| Sock Shop | 15 | Weaveworks' microservice demo |
| Train Ticket | 64 | Large-scale train ticketing system |

Download with `--download-data` and specify the dataset (`RE1`, `RE2`, or `RE3`) and target system.

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| False Positive Reduction | >40% | Reduction in false positive rate compared to naive (no-context) baseline |
| Fault Recall | >90% | Proportion of actual faults correctly identified |
| Overall Accuracy | >80% | Classification accuracy across all classes |

## Installation

```bash
# Clone the repository
git clone https://github.com/Xhadou/CAAA.git
cd CAAA

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as editable package (uses pyproject.toml)
pip install -e ".[test]"
```

**Requirements:** Python 3.9+, PyTorch 2.6+, scikit-learn 1.3+, NumPy, Pandas, SciPy, XGBoost, LightGBM, CatBoost, SHAP, ruptures, Matplotlib, Seaborn, PyYAML.

Verify the installation:

```bash
python -c "import torch; import sklearn; import ruptures; print('Setup OK')"
```

## Quick Start

```bash
# Quick demo (small dataset, fast iteration — ~2 min)
python scripts/demo.py --n-fault 20 --n-load 20 --epochs 30

# Full training with baseline comparison (~10 min)
python scripts/train.py --n-fault 100 --n-load 100 --epochs 50 --baseline

# Training with config file
python scripts/train.py --config configs/config.yaml --baseline

# Or use the convenience script
bash run_experiment.sh
```

## Running Experiments

### Model Selection

CAAA supports multiple model backends. Run individual models via `src.main`:

```bash
# CAAA neural model (proposed method)
python -m src.main --n-fault 50 --n-load 50 --model caaa

# Baseline: Random Forest
python -m src.main --n-fault 50 --n-load 50 --model random_forest

# Baseline: XGBoost
python -m src.main --n-fault 50 --n-load 50 --model xgboost

# Baseline: Rule-based
python -m src.main --n-fault 50 --n-load 50 --model rule_based
```

### Ablation Study

The ablation script evaluates 20 model variants systematically to answer the research questions. Two variants are optional: `CAAA (pretrained)` is active with `--pretrain` on RCAEval runs, and `CAAA (temporal)` is active with `--temporal`.

```bash
# Standard ablation (5 runs per variant)
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5

# With SHAP feature importance analysis
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5 --shap

# With calibration reliability diagrams
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5 --calibration

# With cross-validation
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --cv-folds 5

# Include hard/adversarial scenarios
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5 --include-hard
```

**Ablation variants tested:**

| # | Variant | Purpose |
|---|---------|---------|
| 1 | Full CAAA | Complete proposed model (TADAM + context dropout) |
| 2 | CAAA + Contrastive | With contrastive learning |
| 3 | CAAA (clamp loss) | No unknown penalty, clamp-only consistency |
| 4 | CAAA (full penalty) | Original aggressive penalty (for comparison) |
| 5 | No Context Features | Ablate context inputs |
| 6 | No Context Loss | Standard cross-entropy only |
| 7 | No Behavioral | Ablate behavioral inputs |
| 8 | Context Only | Context features alone |
| 9 | Statistical Only | Statistical features alone |
| 10 | Stat + Service-Level | Combined traditional features |
| 11 | Baseline RF | Random Forest baseline |
| 12 | XGBoost | XGBoost baseline |
| 13 | LightGBM | LightGBM baseline |
| 14 | CatBoost | CatBoost baseline |
| 15 | CatBoost (with context) | Context-enabled tree upper bound |
| 16 | CAAA+CatBoost Hybrid | FiLM embeddings + raw features into CatBoost |
| 17 | CAAA (pretrained) | Synthetic pretrain + real-data fine-tuning (`--pretrain`) |
| 18 | CAAA (temporal) | Temporal encoder branch (`--temporal`) |
| 19 | Rule-Based | Heuristic rules |
| 20 | Naive | No-context naive classifier |

### Scaling Study

The scaling study compares neural and tree models across data sizes and difficulty levels:

```bash
# Default difficulty
python scripts/scaling_study.py

# Moderate difficulty (old hard profile)
python scripts/scaling_study.py --difficulty moderate

# Hard difficulty (highest-difficulty profile) + per-seed logs for significance tests
python scripts/scaling_study.py --difficulty hard --log-per-seed
```

Generated files are written to `outputs/results/` with suffixes by difficulty:
- `scaling_study.csv`, `scaling_curve.png`, `context_contribution.png`, `scaling_per_seed.json`
- `scaling_study_moderate.csv`, `scaling_curve_moderate.png`, `context_contribution_moderate.png`, `scaling_per_seed_moderate.json`
- `scaling_study_hard.csv`, `scaling_curve_hard.png`, `context_contribution_hard.png`, `scaling_per_seed_hard.json`

### Using RCAEval Real-World Data

CAAA evaluates on [RCAEval](https://zenodo.org/records/14590730) benchmark data using a **pre-injection split**: each case is split at `inject_time` into a FAULT half (post-injection) and a NORMAL half (pre-injection). Both halves come from the same real recording, eliminating distribution mismatch. Context is randomized (70% of NORMAL cases get context, 30% don't; 30% of FAULT cases get fake context) to prevent label leakage.

```bash
# Download dataset (one-time, requires network)
python -m src.main --download-data --dataset all --system all

# Ablation on all real data (per-system + combined summary)
python scripts/ablation.py --data rcaeval --dataset all --system all \
    --epochs 50 --n-runs 10

# Single system
python scripts/ablation.py --data rcaeval --dataset RE1 --system online-boutique \
    --epochs 50 --n-runs 10
```

### Full Pipeline with Anomaly Detector

The optional LSTM autoencoder pre-stage filters out normal windows before classification:

```bash
python -m src.main --n-fault 50 --n-load 50 --model caaa \
    --anomaly-detector --ad-epochs 50 --ad-threshold 95
```

## Analyzing Results

### Output Directory

Experiments write results to the `outputs/` directory (gitignored):

```
outputs/
└── results/
    ├── ablation_results_synthetic.csv
    ├── ablation_results_RE1_online-boutique.csv
    ├── ablation_results_RE1_sock-shop.csv
    ├── ablation_results_RE1_train-ticket.csv
    ├── ablation_results_combined.csv
    ├── ablation_results_all_pooled.csv
    ├── scaling_study.csv
    ├── scaling_study_moderate.csv
    ├── scaling_study_hard.csv
    ├── scaling_curve.png
    ├── scaling_curve_moderate.png
    ├── scaling_curve_hard.png
    ├── context_contribution.png
    ├── context_contribution_moderate.png
    ├── context_contribution_hard.png
    ├── scaling_per_seed.json
    ├── scaling_per_seed_moderate.json
    ├── scaling_per_seed_hard.json
    ├── shap_synthetic/
    ├── calibration_synthetic/
    ├── shap_RE*/
    └── calibration_RE*/
```

Model checkpoints are written under `models/final/` and pretraining checkpoints under `models/pretrained/`.

### Metrics Reported

Every experiment prints the following metrics to the console and logs them for ablation runs:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy (UNKNOWN counted as incorrect) |
| **Known Accuracy** | Accuracy computed only over definitive (non-UNKNOWN) predictions |
| **F1 Score** | Harmonic mean of precision and recall (on known predictions only) |
| **Coverage-Adjusted F1** | F1 × (1 − unknown_rate) — rewards both correctness and decisiveness |
| **FP Rate** | False positive rate (loads misclassified as faults). UNKNOWN predictions are excluded — see `known_fp_rate` for a coverage-adjusted view |
| **Known FP Rate** | FP rate computed only over samples with a definitive (non-UNKNOWN) prediction |
| **Fault Recall** | Proportion of true faults correctly identified |
| **FP Reduction** | Percentage improvement in FP rate vs. naive baseline |
| **Unknown Rate** | Fraction of predictions that are UNKNOWN (model deferred the decision) |

### Interpreting Ablation Results

The ablation CSVs (for example `outputs/results/ablation_results_synthetic.csv`, `outputs/results/ablation_results_RE1_online-boutique.csv`, and `outputs/results/ablation_results_combined.csv`) contain mean ± standard deviation for each metric across all runs. Compare variants to answer the research questions:

- **RQ1 (context value)**: Compare "Full CAAA" vs. "No Context Features" and "Naive" rows
- **RQ2 (loss function)**: Compare "Full CAAA" vs. "No Context Loss" rows
- **RQ3 (feature importance)**: Compare feature-group ablation rows (Context Only, Statistical Only, etc.) and review SHAP plots

### Visualization Outputs

- **SHAP plots** (`--shap` flag): Show which features contribute most to predictions, generated per model variant and per fault type
- **Reliability diagrams** (`--calibration` flag): Show calibration before and after temperature scaling
- **Confusion matrices**: Generated during training to show per-class performance
- **FP-vs-threshold coverage curve**: Shows the trade-off between coverage (fraction of definitive predictions) and FP rate as a function of confidence threshold — key diagnostic for comparing fixed and adaptive threshold strategies

## Project Structure

```
CAAA/
├── configs/
│   └── config.yaml                  # Model, training, and data configuration
├── scripts/
│   ├── demo.py                      # Quick demonstration (small dataset)
│   ├── train.py                     # Full training pipeline with baselines
│   ├── ablation.py                  # Ablation study framework
│   └── scaling_study.py             # Data-size scaling study (CAAA vs tree baselines)
├── src/
│   ├── main.py                      # Unified pipeline entry point
│   ├── data_loader/
│   │   ├── data_types.py            # ServiceMetrics, AnomalyCase dataclasses
│   │   ├── synthetic_generator.py   # Normal & load-spike metric generation
│   │   ├── fault_generator.py       # 11-type fault injection engine
│   │   ├── dataset.py              # Combined & research dataset generation
│   │   ├── download_data.py        # RCAEval dataset downloader
│   │   ├── rcaeval_loader.py       # RCAEval dataset parser
│   │   └── utils.py               # Shared base metrics generation utilities
│   ├── features/
│   │   ├── feature_schema.py       # Single source of truth for 44-dim layout
│   │   ├── extractors.py           # Feature extraction from raw metrics
│   │   └── context_features.py     # ContextFeatures dataclass (container only)
│   ├── models/
│   │   ├── caaa_model.py           # CAAA neural model (proposed)
│   │   ├── feature_encoder.py      # MLP-based feature encoder
│   │   ├── context_module.py       # Context integration with attention & gating
│   │   ├── anomaly_detector.py     # LSTM autoencoder for pre-stage detection
│   │   ├── classifier.py          # Multi-backend sklearn classifier
│   │   └── baseline.py            # RF, XGBoost, LightGBM, CatBoost, TabPFN baselines
│   ├── training/
│   │   ├── losses.py              # Context Consistency Loss (novel)
│   │   └── trainer.py             # PyTorch training harness with early stopping
│   ├── evaluation/
│   │   ├── metrics.py             # Evaluation metrics & FP reduction measurement
│   │   └── visualization.py       # Confusion matrices, feature importance plots
│   └── utils/                     # Shared utility helpers
├── tests/
│   ├── test_data_loader.py        # Data generation tests
│   ├── test_features.py           # Feature extraction tests
│   ├── test_models.py             # Model component tests
│   ├── test_integration.py        # End-to-end pipeline tests
│   ├── test_plan_modules.py       # Sklearn classifier tests
│   ├── test_review_fixes.py       # Code review fix verification
│   └── test_rcaeval_pipeline.py   # RCAEval integration tests
├── GETTING_STARTED.md               # Step-by-step starter guide
├── CAAA Literature Review.md        # 150+ paper literature review
├── requirements.txt
├── pyproject.toml
├── run_experiment.sh
└── README.md
```

## Configuration Reference

All model and training parameters are configurable via `configs/config.yaml`. Key settings:

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.caaa_model.input_dim` | 44 | Feature vector dimensionality |
| `model.caaa_model.hidden_dim` | 64 | Hidden layer size |
| `model.caaa_model.context_dim` | 5 | Number of context features |
| `model.caaa_model.n_classes` | 2 | Output classes (FAULT, EXPECTED_LOAD) |
| `model.caaa_model.dropout` | 0.1 | Dropout rate |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 50 | Training epochs |
| `training.batch_size` | 32 | Mini-batch size |
| `training.learning_rate` | 0.001 | Adam optimizer learning rate |
| `training.early_stopping_patience` | 10 | Epochs to wait before early stopping |
| `training.test_split` | 0.2 | Fraction of data reserved for testing |
| `training.val_split` | 0.2 | Fraction of training data for validation |

### Training Features

The trainer includes several features for stable and reliable training:

- **Gradient clipping**: Applied by default with `max_grad_norm=1.0` to prevent exploding gradients.
- **ReduceLROnPlateau scheduler**: Automatically halves the learning rate when validation loss plateaus (patience=5, min_lr=1e-6).
- **StandardScaler**: Input features should be standardized (zero mean, unit variance) before training neural models. Use `sklearn.preprocessing.StandardScaler` fitted on training data only. Tree-based baselines (RandomForest, XGBoost) are trained on unscaled features for consistency with their scale-invariant nature.
- **Temperature calibration**: Post-hoc temperature scaling for calibrated confidence scores, using validation data.
- **Instance-level RNG**: All data generators use `np.random.default_rng(seed)` for per-instance reproducibility — multiple generators can coexist without corrupting each other's random state. Additionally, per-case seeds (`case_seed` parameter) ensure that generation order does not create subtle statistical dependence between training and evaluation data.
- **Feature dimensionality**: All scripts import `N_FEATURES` from `src.features.feature_schema` rather than hard-coding `36`, maintaining a single source of truth for the feature vector layout.
- **Cached change-point detection**: The PELT change-point detection function (`_detect_change_point_cached`) uses `@functools.lru_cache(maxsize=4096)` for O(1) repeated lookups, reducing feature extraction time for repeated or similar cases.
- **Vectorized cross-service correlation**: The `cross_service_sync` feature uses `np.corrcoef` on the full CPU matrix instead of O(n²) pairwise Pearson correlations, with explicit filtering of constant series to avoid NaN values.
- **Optimized feature extraction**: Feature methods pre-extract metric arrays once per service instead of repeated `.values` lookups, and statistical features use `np.concatenate` + `np.mean`/`np.std` directly instead of `pd.concat().mean()` to avoid per-case DataFrame allocation overhead.

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evaluation.fp_reduction_target` | 0.40 | Target false positive reduction (>40%) |
| `evaluation.fault_recall_target` | 0.90 | Target fault recall (>90%) |

### Anomaly Detector Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.anomaly_detector.hidden_dim` | 64 | LSTM hidden dimension |
| `model.anomaly_detector.latent_dim` | 16 | Latent space dimension |
| `model.anomaly_detector.num_layers` | 2 | Number of LSTM layers |
| `model.anomaly_detector.threshold_percentile` | 95 | Anomaly threshold percentile |

### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.n_services` | 12 | Number of simulated microservices |
| `data.sequence_length` | 60 | Time-series window length |
| `features.window_size` | 60 | Feature extraction window |
| `features.stride` | 10 | Feature extraction stride |

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test modules
python -m pytest tests/test_integration.py -v    # End-to-end pipeline
python -m pytest tests/test_models.py -v          # Model components
python -m pytest tests/test_features.py -v        # Feature extraction
python -m pytest tests/test_data_loader.py -v     # Data generation
python -m pytest tests/test_plan_modules.py -v    # Sklearn classifiers
python -m pytest tests/test_rcaeval_pipeline.py -v # RCAEval integration
python -m pytest tests/test_review_fixes.py -v     # Code review fix verification
```

## Related Work

This project builds on and addresses gaps identified in the following research areas. A full literature review covering 150+ papers (2020–2025) is available in [`CAAA Literature Review.md`](CAAA%20Literature%20Review.md).

- **Anomaly Detection**: Anomaly Transformer (ICLR 2022), DCdetector (KDD 2023), USAD (KDD 2020), OmniAnomaly (KDD 2019)
- **Root Cause Analysis**: RCAEval benchmark (WWW 2025), BARO (FSE 2024), CIRCA (KDD 2022), RCD (NeurIPS 2022), DynaCausal (2025)
- **GNN-based Fault Localization**: MicroRCA (NOMS 2020), MicroIRC (JSS 2024), CHASE (2024), DiagFusion (IEEE TSC 2023)
- **Multi-modal Fusion**: AnoFusion (KDD 2023), DeepTraLog (ICSE 2022)
- **LLM for AIOps**: RCACopilot (EuroSys 2024), RCAgent (CIKM 2024)

**Key gap addressed**: No prior work dynamically adjusts anomaly classification based on workload context (time-of-day patterns, known events, deployment changes). CAAA is the first to integrate these signals into a unified attribution framework.

## Known Limitations

- **No GNN component**: The architecture uses feature-level graph information (service adjacency, anomaly centrality) but does not include a full Graph Neural Network for structural reasoning over the service topology.
- **Synthetic data caveats**: While the synthetic data generator produces realistic fault patterns (AR(1) autocorrelation, per-service baselines, load jitter), it cannot capture all real-world failure modes. RCAEval benchmark evaluation helps validate on real traces.
- **Small dataset regime**: With typical dataset sizes <10K samples, deep learning baselines may not reach their full potential. TabPFN is included as a foundation model baseline designed for this regime.
- **Binary classification**: The model classifies as FAULT vs EXPECTED_LOAD (with post-hoc UNKNOWN). Multi-class fault type classification is out of scope.

## Citation

```bibtex
@article{caaa2025,
  title={CAAA: Context-Aware Anomaly Attribution for False Positive
         Reduction in Cloud Microservice Monitoring},
  author={Jain, Pratyush},
  year={2025},
  institution={Shiv Nadar University}
}
```

## License

This project is developed as academic research at Shiv Nadar University.

## Author

Pratyush Jain — Shiv Nadar University
