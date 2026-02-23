# Getting Started with CAAA

This guide walks you through setting up your environment, running experiments, and analyzing results with the CAAA (Context-Aware Anomaly Attribution) framework.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Verify Installation](#3-verify-installation)
4. [Run Your First Experiment](#4-run-your-first-experiment)
5. [Understanding the Output](#5-understanding-the-output)
6. [Running the Full Training Pipeline](#6-running-the-full-training-pipeline)
7. [Running Ablation Studies](#7-running-ablation-studies)
8. [Working with Real-World Data (RCAEval)](#8-working-with-real-world-data-rcaeval)
9. [Using the Anomaly Detector Pre-Stage](#9-using-the-anomaly-detector-pre-stage)
10. [Customizing Experiments via Configuration](#10-customizing-experiments-via-configuration)
11. [Analyzing and Interpreting Results](#11-analyzing-and-interpreting-results)
12. [Running Tests](#12-running-tests)
13. [Troubleshooting](#13-troubleshooting)
14. [Suggested Experiment Workflow](#14-suggested-experiment-workflow)

---

## 1. Prerequisites

Before you begin, make sure you have the following installed on your machine:

| Requirement | Minimum Version | Check Command |
|-------------|----------------|---------------|
| **Python** | 3.9+ | `python --version` or `python3 --version` |
| **pip** | 21.0+ | `pip --version` |
| **Git** | any | `git --version` |

**Operating System**: Linux, macOS, or Windows (with WSL recommended for Windows users).

**Hardware**:
- **CPU-only** is fine for all experiments — no GPU required.
- The synthetic data experiments run in minutes on a modern laptop.
- Ablation studies with many runs (`--n-runs 10+`) benefit from more CPU cores, but a single core works.

---

## 2. Environment Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Xhadou/CAAA.git
cd CAAA
```

### Step 2: Create a virtual environment (recommended)

Using a virtual environment avoids conflicts with system-level Python packages.

**Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **Tip:** If you use Conda, you can create a Conda environment instead:
> ```bash
> conda create -n caaa python=3.10 -y
> conda activate caaa
> ```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages: PyTorch, scikit-learn, XGBoost, SHAP, NumPy, Pandas, Matplotlib, and more. See `requirements.txt` for the full list.

---

## 3. Verify Installation

Run the following to confirm everything is installed correctly:

```bash
python -c "import torch; import sklearn; import xgboost; import shap; print('All dependencies OK')"
```

You can also run the test suite to verify the codebase works:

```bash
python -m pytest tests/ -v --tb=short
```

All tests should pass. If any fail, see the [Troubleshooting](#13-troubleshooting) section.

---

## 4. Run Your First Experiment

The quickest way to see CAAA in action is the demo script:

```bash
python scripts/demo.py --n-fault 20 --n-load 20 --epochs 30
```

**What this does:**
1. Generates 20 synthetic fault cases and 20 expected-load cases
2. Extracts 36-dimensional feature vectors from each case
3. Trains the CAAA neural model for 30 epochs
4. Evaluates and prints classification metrics

**Expected runtime:** ~1–2 minutes on a modern laptop.

**Expected output** (values will vary by run):

```
=== CAAA Demo Results ===
Accuracy:  0.875
F1 Score:  0.889
FP Rate:   0.100
Fault Recall: 0.950
FP Reduction: 52.3%
```

Alternatively, use the one-command convenience script:

```bash
bash run_experiment.sh
```

This installs dependencies (if needed), runs the CAAA pipeline with 50 fault/load cases for 50 epochs, and saves results to `outputs/results/`.

---

## 5. Understanding the Output

Every experiment reports these key metrics:

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Accuracy** | Overall classification correctness | >80% |
| **F1 Score** | Balance of precision and recall | Higher is better |
| **FP Rate** | Fraction of load cases wrongly classified as faults | Lower is better |
| **Fault Recall** | Fraction of actual faults correctly identified | >90% |
| **FP Reduction** | Improvement in FP rate vs. a naive (no-context) baseline | >40% |

**Interpreting the results:**
- **High fault recall + low FP rate** = the model correctly identifies faults while not raising false alarms on legitimate load events.
- **FP reduction >40%** means CAAA reduces false positives by at least 40% compared to a classifier that ignores context — this is the central research claim.

---

## 6. Running the Full Training Pipeline

For a more thorough experiment with baseline comparison:

```bash
python scripts/train.py --n-fault 100 --n-load 100 --epochs 50 --baseline
```

**What this does:**
1. Generates 100 fault + 100 load cases (larger dataset for better estimates)
2. Trains the CAAA model for 50 epochs with early stopping
3. Also trains a Random Forest baseline (`--baseline` flag)
4. Compares both models side-by-side
5. Saves the trained CAAA model to `models/final/caaa_model.pt`

**Expected runtime:** ~5–10 minutes.

### Command-Line Options for `train.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--n-fault` | 50 | Number of fault cases to generate |
| `--n-load` | 50 | Number of expected-load cases |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 32 | Mini-batch size |
| `--lr` | 0.001 | Learning rate |
| `--seed` | 42 | Random seed for reproducibility |
| `--baseline` | off | Also train a Random Forest baseline |
| `--shap` | off | Generate SHAP feature importance plots |
| `--config` | — | Path to a YAML config file |

### Using a Config File

Instead of passing flags, you can use the provided config:

```bash
python scripts/train.py --config configs/config.yaml --baseline
```

See `configs/config.yaml` for all tunable parameters.

---

## 7. Running Ablation Studies

The ablation study is the most important experiment — it systematically evaluates 12 model variants to answer the research questions.

### Basic ablation

```bash
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5
```

This trains each of the 12 variants 5 times with different random seeds and reports mean ± standard deviation.

### With SHAP feature importance

```bash
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5 --shap
```

Generates SHAP beeswarm plots showing which features matter most for each model variant. Plots are saved to `outputs/figures/shap/`.

### With calibration analysis

```bash
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5 --calibration
```

Generates reliability diagrams before and after temperature scaling. Saved to `outputs/figures/calibration/`.

### With cross-validation

```bash
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --cv-folds 5
```

Uses 5-fold cross-validation instead of repeated train/test splits.

### Including hard scenarios

```bash
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5 --include-hard
```

Adds adversarial and hard-negative cases to test robustness.

### Full ablation (recommended for paper-quality results)

```bash
python scripts/ablation.py \
    --n-fault 100 --n-load 100 \
    --epochs 50 --n-runs 10 \
    --shap --calibration --include-hard
```

**Expected runtime:** 30–90 minutes depending on your hardware.

### Ablation Variants

The ablation evaluates these 12 variants:

| # | Variant | What It Tests |
|---|---------|--------------|
| 1 | Full CAAA | Complete proposed model (all features + context loss) |
| 2 | CAAA + Contrastive | Adds contrastive learning objective |
| 3 | No Context Features | Removes the 5 context features (dims 12–16) |
| 4 | No Context Loss | Uses standard cross-entropy instead of Context Consistency Loss |
| 5 | No Behavioral Features | Removes behavioral features (dims 6–11) |
| 6 | Context Only | Uses only context features |
| 7 | Statistical Only | Uses only statistical features |
| 8 | Stat + Service-Level | Uses statistical + service-level features |
| 9 | Baseline RF | Random Forest on all features |
| 10 | XGBoost | XGBoost on all features |
| 11 | Rule-Based | Hand-crafted heuristic rules |
| 12 | Naive | No-context baseline (always predicts FAULT) |

---

## 8. Working with Real-World Data (RCAEval)

CAAA supports evaluation on the [RCAEval benchmark](https://zenodo.org/records/14590730) — real microservice failure traces.

### Step 1: Download the dataset

```bash
python -m src.main --download-data --dataset RE1 --system online-boutique
```

Available datasets and systems:

| Dataset | Description |
|---------|-------------|
| `RE1` | Primary benchmark dataset |
| `RE2` | Extended dataset |
| `RE3` | Additional fault scenarios |

| System | Services | Description |
|--------|----------|-------------|
| `online-boutique` | 12 | Google's microservice demo |
| `sock-shop` | 15 | Weaveworks' microservice demo |
| `train-ticket` | 64 | Large-scale train ticketing system |

### Step 2: Train on real data

```bash
python -m src.main --data rcaeval --dataset RE1 --system online-boutique --model caaa
```

This uses real fault traces combined with synthetic expected-load cases (since the benchmark only contains faults).

### Step 3: Run ablation on real data

```bash
python scripts/ablation.py --data rcaeval --dataset RE1 --system online-boutique \
    --epochs 50 --n-runs 10
```

---

## 9. Using the Anomaly Detector Pre-Stage

CAAA has an optional Stage 1 — an LSTM autoencoder that filters normal windows before classification:

```bash
# Synthetic data with anomaly detector
python -m src.main --n-fault 50 --n-load 50 --model caaa \
    --anomaly-detector --ad-epochs 50

# Real data with anomaly detector
python -m src.main --data rcaeval --dataset RE1 --system online-boutique \
    --model caaa --anomaly-detector --ad-epochs 50 --ad-threshold 95
```

| Flag | Default | Description |
|------|---------|-------------|
| `--anomaly-detector` | off | Enable the LSTM-AE pre-stage |
| `--ad-epochs` | 50 | Epochs to train the anomaly detector |
| `--ad-threshold` | 95 | Percentile threshold for anomaly detection |

---

## 10. Customizing Experiments via Configuration

All parameters can be set in `configs/config.yaml`. Here are the most commonly adjusted settings:

### Data settings

```yaml
data:
  n_services: 12          # Number of simulated microservices
  sequence_length: 60     # Time-series window length
```

### Model settings

```yaml
model:
  caaa_model:
    input_dim: 36         # Feature vector size (don't change unless modifying features)
    hidden_dim: 64        # Hidden layer size (try 32 or 128)
    context_dim: 5        # Context features count
    dropout: 0.1          # Dropout rate (try 0.2 for more regularization)
```

### Training settings

```yaml
training:
  epochs: 50              # Increase for better convergence
  batch_size: 32          # Decrease if running out of memory
  learning_rate: 0.001    # Try 0.0005 or 0.002
  early_stopping_patience: 10  # Increase for more patience
  test_split: 0.2         # 80/20 train/test split
```

### Evaluation targets

```yaml
evaluation:
  fp_reduction_target: 0.40  # >40% FP reduction goal
  fault_recall_target: 0.90  # >90% fault recall goal
```

To use a config file:

```bash
python scripts/train.py --config configs/config.yaml
python -m src.main --config configs/config.yaml
```

Command-line flags override config file values when both are specified.

---

## 11. Analyzing and Interpreting Results

### Output file locations

After running experiments, results are saved under the `outputs/` directory:

```
outputs/
├── results/
│   ├── caaa_model.pt              # Trained model checkpoint
│   └── ablation_results.csv       # Ablation metrics table
└── figures/
    ├── shap/
    │   ├── shap_full_caaa.png     # Feature importance (CAAA)
    │   ├── shap_baseline_rf.png   # Feature importance (RF)
    │   └── shap_*_by_fault_type.png
    └── calibration/
        ├── reliability_uncalibrated.png
        └── reliability_calibrated.png
```

### Reading ablation results

Open `outputs/results/ablation_results.csv` in a spreadsheet or with Python:

```python
import pandas as pd
df = pd.read_csv("outputs/results/ablation_results.csv")
print(df.to_string())
```

Each row is a model variant. Compare columns to answer the research questions:

| Research Question | What to Compare |
|-------------------|----------------|
| **RQ1** (Does context help?) | "Full CAAA" vs. "No Context Features" and "Naive" — look at FP Reduction and Fault Recall |
| **RQ2** (Does Context Loss help?) | "Full CAAA" vs. "No Context Loss" — look at FP Reduction |
| **RQ3** (Which features matter?) | Feature-group ablation rows (Context Only, Statistical Only, etc.) and SHAP plots |

### Interpreting SHAP plots

SHAP (SHapley Additive exPlanations) plots show how much each feature contributes to predictions:

- **Red dots on the right** = high feature values push toward FAULT prediction
- **Blue dots on the right** = low feature values push toward FAULT prediction
- Features at the top are most influential

Look for context features (`event_active`, `time_seasonality`, etc.) appearing high in the ranking — this confirms context integration is valuable.

### Interpreting calibration plots

Reliability diagrams show whether predicted probabilities match actual outcomes:

- A well-calibrated model follows the diagonal line
- Points above the diagonal = under-confident, below = over-confident
- Compare "uncalibrated" vs. "calibrated" (after temperature scaling) to see improvement

---

## 12. Running Tests

The test suite validates all components of the framework:

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run with coverage reporting
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test modules

| Test File | What It Tests |
|-----------|--------------|
| `tests/test_data_loader.py` | Synthetic data generation, fault injection |
| `tests/test_features.py` | Feature extraction, 36-dim schema validation |
| `tests/test_models.py` | CAAA model, context module, feature encoder |
| `tests/test_integration.py` | End-to-end training and evaluation pipeline |
| `tests/test_plan_modules.py` | Sklearn classifier wrappers |
| `tests/test_rcaeval_pipeline.py` | RCAEval data loading and processing |

Run individual modules for faster feedback:

```bash
python -m pytest tests/test_models.py -v        # ~10 seconds
python -m pytest tests/test_features.py -v       # ~5 seconds
python -m pytest tests/test_integration.py -v    # ~30 seconds
```

---

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

PyTorch was not installed. Run:

```bash
pip install torch>=2.0.0
```

If you need a specific PyTorch build (CPU-only, CUDA version), see [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### `ModuleNotFoundError: No module named 'src'`

Make sure you are running commands from the repository root directory (`CAAA/`). The `src` package is resolved relative to the working directory.

```bash
cd /path/to/CAAA
python -m src.main --help
```

### `ImportError: No module named 'xgboost'` or `'shap'`

Install missing dependencies:

```bash
pip install -r requirements.txt
```

### Tests fail with import errors

Ensure you have activated your virtual environment:

```bash
source venv/bin/activate  # Linux/macOS
```

### Out of memory during training

Reduce batch size:

```bash
python scripts/train.py --batch-size 16 --n-fault 50 --n-load 50
```

### Experiments are slow

- Reduce `--n-fault` and `--n-load` for faster iterations
- Reduce `--epochs` (30 is often enough for convergence)
- Reduce `--n-runs` in ablation (3 instead of 10)
- Use `scripts/demo.py` for quick sanity checks

### RCAEval download fails

The dataset is hosted on Zenodo. Ensure you have network access and try again:

```bash
python -m src.main --download-data --dataset RE1 --system online-boutique
```

If behind a proxy, configure `HTTP_PROXY` and `HTTPS_PROXY` environment variables.

---

## 14. Suggested Experiment Workflow

Here is a recommended workflow from initial exploration to paper-ready results:

### Phase 1: Familiarization (~5 min)

```bash
# 1. Install and verify
pip install -r requirements.txt
python -m pytest tests/ -v --tb=short

# 2. Run a quick demo
python scripts/demo.py --n-fault 20 --n-load 20 --epochs 30
```

### Phase 2: Initial Training (~10 min)

```bash
# 3. Train CAAA with baseline comparison
python scripts/train.py --n-fault 100 --n-load 100 --epochs 50 --baseline

# 4. Try different model backends
python -m src.main --n-fault 50 --n-load 50 --model xgboost
python -m src.main --n-fault 50 --n-load 50 --model random_forest
python -m src.main --n-fault 50 --n-load 50 --model rule_based
```

### Phase 3: Ablation Study (~30–60 min)

```bash
# 5. Run the full ablation with SHAP analysis
python scripts/ablation.py \
    --n-fault 100 --n-load 100 \
    --epochs 50 --n-runs 5 \
    --shap --calibration

# 6. Review results
cat outputs/results/ablation_results.csv
```

### Phase 4: Real-World Validation (~20 min)

```bash
# 7. Download and evaluate on RCAEval
python -m src.main --download-data --dataset RE1 --system online-boutique
python -m src.main --data rcaeval --dataset RE1 --system online-boutique --model caaa

# 8. Ablation on real data
python scripts/ablation.py --data rcaeval --dataset RE1 --system online-boutique \
    --epochs 50 --n-runs 5
```

### Phase 5: Paper-Quality Results (~60–90 min)

```bash
# 9. High-quality ablation with all analyses
python scripts/ablation.py \
    --n-fault 200 --n-load 200 \
    --epochs 100 --n-runs 10 \
    --shap --calibration --include-hard

# 10. Full pipeline with anomaly detector
python -m src.main --data rcaeval --dataset RE1 --system online-boutique \
    --model caaa --anomaly-detector --ad-epochs 50
```

---

For more details on the research motivation, architecture, and related work, see the main [README.md](README.md).
