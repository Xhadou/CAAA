# CAAA: Context-Aware Anomaly Attribution

> A research framework for false positive reduction in cloud microservice anomaly detection through context-aware classification.

## Research Motivation

Modern cloud-native applications built on microservice architectures generate a high volume of monitoring alerts. Production deployments report **false positive rates of 6–28%**, and even state-of-the-art root cause analysis methods achieve only **46–63% localization accuracy** on large-scale systems ([RCAEval, WWW 2025](https://zenodo.org/records/14590730)). Alert fatigue from false positives leads operators to ignore or delay responses, directly undermining the reliability benefits that monitoring is supposed to provide.

A core reason for this gap is that existing anomaly detection methods lack **contextual awareness**. A sudden spike in CPU utilization and request latency looks identical to an anomaly detector whether it is caused by a cascading fault or by a planned marketing campaign driving a legitimate traffic surge. Current approaches treat all anomalous-looking patterns as faults, producing false alerts whenever normal operational events (scheduled load tests, auto-scaling events, time-of-day traffic peaks, or recent deployments) cause metric deviations.

CAAA addresses this gap by integrating external context signals — scheduled events, deployment history, and temporal seasonality — directly into the anomaly classification pipeline. Rather than simply detecting anomalies, CAAA **attributes** them: classifying each anomaly as a **FAULT** (actual system issue), **EXPECTED_LOAD** (legitimate workload spike), or **UNKNOWN** (insufficient confidence for automatic classification).

### Research Questions

1. **RQ1**: Can integrating operational context signals (scheduled events, deployments, time-of-day patterns) into anomaly classification reduce false positives by >40% while maintaining >90% fault recall?
2. **RQ2**: How does the proposed Context Consistency Loss compare to standard cross-entropy for training context-aware classifiers?
3. **RQ3**: Which context features contribute most to false positive reduction, and how does performance degrade as context availability decreases?

### Novel Contributions

1. **Context-Aware Anomaly Attribution** — The first framework to explicitly distinguish workload-induced anomalies from actual faults using external context signals within a unified classification pipeline.
2. **Context Consistency Loss** — A novel composite loss function that penalizes predictions contradicting available context signals while calibrating prediction confidence based on context reliability.
3. **Context Integration Module** — An attention-based module with confidence gating that learns to weight and integrate heterogeneous context features into the classification decision.

## Architecture

CAAA uses an optional two-stage pipeline:

```
                          ┌─────────────────────────────────────────────┐
                          │         Stage 2: Anomaly Attribution        │
 Metrics ──► [Stage 1] ──►  ┌───────────┐  ┌───────────┐  ┌──────────┐  │
 (optional   Anomaly      │ │  Feature  │─►│ Context   │─►│ Classif. │──► FAULT / EXPECTED_LOAD / UNKNOWN
  LSTM-AE)   Detection    │ │  Encoder  │  │ Module    │  │   Head   │  │
                          │ │(MLP+LN)  │  │(FiLM+Gate)│  │  (GELU)  │  │
                          │ │ 44d → 64d │  │ 5 context │  │ 2-class  │  │
                          └─────────────────────────────────────────────┘
```

**Stage 1 (Optional): Anomaly Detection** — An LSTM autoencoder trained on normal (expected-load) metrics identifies anomalous time windows via reconstruction error. Only anomalous windows proceed to Stage 2. Enable with the `--anomaly-detector` flag.

**Stage 2: Anomaly Attribution** — The core CAAA model classifies detected anomalies:

| Component | Description |
|-----------|-------------|
| **Feature Encoder** | 2-layer MLP with LayerNorm and GELU, projecting 44-dim feature vectors into 64-dim representations |
| **Context Integration Module** | FiLM (Feature-wise Linear Modulation) conditioning with enriched confidence gating (3-feature gate) |
| **Classification Head** | 2-class output (FAULT vs EXPECTED_LOAD) with post-hoc UNKNOWN assignment via confidence thresholding |

**Training methodology:**
- 3-way stratified split (60% train / 20% val / 20% test) — validation for early stopping and calibration, test for final metrics only
- AdamW optimizer with ReduceLROnPlateau scheduler and gradient clipping (max_norm=1.0)
- StandardScaler normalization for neural models (tree baselines use raw features)
- GPU auto-detection (`--device auto|cpu|cuda`)

**Context Consistency Loss** combines three terms:
- Cross-entropy classification loss (with label smoothing 0.1)
- Context consistency penalty — penalizes predictions that contradict available context signals, weighted by confidence² for softer penalty on ambiguous cases
- Confidence calibration loss — entropy regularization guided by context confidence scores

## Feature Vector

The system extracts a 44-dimensional feature vector organized into 6 groups, defined centrally in `src/features/feature_schema.py`:

| Group | Dims | Features | Purpose |
|-------|------|----------|---------|
| **Workload** | 0–5 | `global_load_ratio`, `cpu_request_correlation`, `cross_service_sync`, `error_rate_delta`, `latency_cpu_correlation`, `memory_trend_uniformity` | Characterize whether metric changes correlate with workload |
| **Behavioral** | 6–11 | `onset_gradient`, `peak_duration`, `cascade_score`, `recovery_indicator`, `affected_service_ratio`, `variance_change_ratio` | Capture fault propagation signatures vs. smooth load ramps |
| **Context** | 12–16 | `event_active`, `event_expected_impact`, `time_seasonality`, `recent_deployment`, `context_confidence` | External context signals (the key innovation) |
| **Statistical** | 17–29 | Mean/std of CPU, memory, requests, errors, latency, network; `max_error_rate` | Standard metric statistics |
| **Service-Level** | 30–35 | `n_services`, `max_cpu_service_ratio`, `max_error_service_ratio`, `cpu_spread`, `error_spread`, `latency_spread` | Cross-service aggregation patterns |
| **Extended** | 36–43 | `spectral_entropy`, `high_freq_energy_ratio`, `dominant_frequency`, `network_asymmetry`, `graph_anomaly_centrality`, `anomaly_spread`, `max_cpu_derivative`, `error_rate_slope` | Frequency-domain, graph-structural, and rate-of-change features |

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| False Positive Reduction | >40% | Reduction in false positive rate compared to naive (no-context) baseline |
| Fault Recall | >90% | Proportion of actual faults correctly identified |
| Overall Accuracy | >80% | Classification accuracy across all classes |

Run `python scripts/ablation.py` to reproduce systematic evaluations across model variants and hyperparameters.

## Installation

```bash
# Clone the repository
git clone https://github.com/Xhadou/CAAA.git
cd CAAA

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.6+, scikit-learn 1.3+, NumPy, Pandas, SciPy, XGBoost, LightGBM, CatBoost, SHAP, ruptures, Matplotlib, Seaborn, PyYAML.

## Quick Start

```bash
# Quick demo (small dataset, fast iteration)
python scripts/demo.py --n-fault 20 --n-load 20 --epochs 30

# Full training with baseline comparison
python scripts/train.py --n-fault 100 --n-load 100 --epochs 50 --baseline

# Training with config file
python scripts/train.py --config configs/config.yaml --baseline

# Or use the convenience script
bash run_experiment.sh
```

### Model Selection

CAAA supports multiple model backends for comparison:

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

```bash
# Systematic evaluation across model variants
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5
```

### Using RCAEval Real-World Data

CAAA can be evaluated on [RCAEval](https://zenodo.org/records/14590730) benchmark data — real-world microservice failure traces from Online Boutique (12 services), Sock Shop (15 services), and Train Ticket (64 services):

```bash
# Download dataset (one-time, requires network)
python -m src.main --download-data --dataset RE1 --system online-boutique

# Train with real fault data + synthetic expected-load cases
python -m src.main --data rcaeval --dataset RE1 --system online-boutique --model caaa

# Full pipeline with anomaly detection pre-stage
python -m src.main --data rcaeval --dataset RE1 --system online-boutique \
    --model caaa --anomaly-detector --ad-epochs 50

# Ablation on real data
python scripts/ablation.py --data rcaeval --dataset RE1 --system online-boutique \
    --epochs 50 --n-runs 10
```

## Data

CAAA supports two data modes:

**Synthetic Data** (default) — Generated on-the-fly with no downloads required. Includes:
- Normal microservice metric generation with realistic temporal patterns
- 11 fault injection types: CPU spike, memory leak, latency injection, error burst, cascading failure, network partition, disk I/O saturation, thread pool exhaustion, connection pool leak, garbage collection storm, and downstream timeout
- Load spike generation with configurable intensity and correlation patterns
- Adversarial and hard-negative scenarios for robust evaluation

**RCAEval Benchmark Data** — Real-world microservice failure traces from Zenodo covering three systems and 735 fault cases across 11 fault types. Download with `--download-data` and specify the dataset (RE1, RE2, RE3) and target system.

## Project Structure

```
CAAA/
├── configs/
│   └── config.yaml                  # Model, training, and data configuration
├── scripts/
│   ├── demo.py                      # Quick demonstration (small dataset)
│   ├── train.py                     # Full training pipeline with baselines
│   └── ablation.py                  # Ablation study framework
├── src/
│   ├── main.py                      # Unified pipeline entry point
│   ├── data_loader/
│   │   ├── data_types.py            # ServiceMetrics, AnomalyCase dataclasses
│   │   ├── synthetic_generator.py   # Normal & load-spike metric generation
│   │   ├── fault_generator.py       # 11-type fault injection engine
│   │   ├── dataset.py              # Combined & research dataset generation
│   │   ├── download_data.py        # RCAEval dataset downloader
│   │   └── rcaeval_loader.py       # RCAEval dataset parser
│   ├── features/
│   │   ├── feature_schema.py       # Single source of truth for 44-dim layout
│   │   ├── extractors.py           # Feature extraction from raw metrics
│   │   └── context_features.py     # Context feature computation
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
│   └── evaluation/
│       ├── metrics.py             # Evaluation metrics & FP reduction measurement
│       └── visualization.py       # Confusion matrices, feature importance plots
├── tests/
│   ├── test_data_loader.py        # Data generation tests
│   ├── test_features.py           # Feature extraction tests
│   ├── test_models.py             # Model component tests
│   ├── test_integration.py        # End-to-end pipeline tests
│   ├── test_plan_modules.py       # Sklearn classifier tests
│   └── test_rcaeval_pipeline.py   # RCAEval integration tests
├── requirements.txt
├── pyproject.toml
├── run_experiment.sh
└── README.md
```

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
```

## Related Work

This project builds on and addresses gaps identified in the following research areas. A full literature review covering 150+ papers (2020–2025) is available in `CAAA Literature Review.md`.

- **Anomaly Detection**: Anomaly Transformer (ICLR 2022), DCdetector (KDD 2023), USAD (KDD 2020), OmniAnomaly (KDD 2019)
- **Root Cause Analysis**: RCAEval benchmark (WWW 2025), BARO (FSE 2024), CIRCA (KDD 2022), RCD (NeurIPS 2022), DynaCausal (2025)
- **GNN-based Fault Localization**: MicroRCA (NOMS 2020), MicroIRC (JSS 2024), CHASE (2024), DiagFusion (IEEE TSC 2023)
- **Multi-modal Fusion**: AnoFusion (KDD 2023), DeepTraLog (ICSE 2022)
- **LLM for AIOps**: RCACopilot (EuroSys 2024), RCAgent (CIKM 2024)

**Key gap addressed**: No prior work dynamically adjusts anomaly classification based on workload context (time-of-day patterns, known events, deployment changes). CAAA is the first to integrate these signals into a unified attribution framework.

## Configuration

All model and training parameters are configurable via `configs/config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.caaa_model.input_dim` | 44 | Feature vector dimensionality |
| `model.caaa_model.hidden_dim` | 64 | Hidden layer size |
| `model.caaa_model.context_dim` | 5 | Number of context features |
| `training.epochs` | 50 | Training epochs |
| `training.learning_rate` | 0.001 | Learning rate |
| `training.early_stopping_patience` | 10 | Early stopping patience |
| `evaluation.fp_reduction_target` | 0.40 | Target false positive reduction |
| `evaluation.fault_recall_target` | 0.90 | Target fault recall |

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
