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
15. [Extending CAAA](#15-extending-caaa)

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

### Step 3: Install PyTorch

PyTorch must be installed **before** the rest of the requirements, using the correct index for your hardware.

**GPU (NVIDIA, CUDA 12.x — recommended):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> This works with any NVIDIA driver that supports CUDA 12.x (driver ≥ 525). The `cu126` build is compatible with drivers up to 12.8+.

**CPU-only:**

```bash
pip install torch torchvision torchaudio
```

### Step 4: Install remaining dependencies

```bash
pip install -r requirements.txt
```

This installs scikit-learn, XGBoost, SHAP, ruptures, NumPy, Pandas, Matplotlib, and more. PyTorch (already installed above) will be skipped. See `requirements.txt` for the full list.

Alternatively, install as an editable package using `pyproject.toml` (includes all runtime dependencies):

```bash
pip install -e ".[test]"
```

---

## 3. Verify Installation

Run the following to confirm everything is installed correctly:

```bash
python -c "import torch; import sklearn; import xgboost; import shap; import ruptures; print('All dependencies OK')"
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
2. Extracts 44-dimensional feature vectors from each case
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
| **Accuracy** | Overall classification correctness (UNKNOWN counted as incorrect) | >80% |
| **Known Accuracy** | Accuracy over definitive (non-UNKNOWN) predictions only | >80% |
| **F1 Score** | Balance of precision and recall (on known predictions only) | Higher is better |
| **Coverage-Adjusted F1** | F1 × (1 − unknown_rate) — rewards both correctness and decisiveness | Higher is better |
| **FP Rate** | Fraction of load cases wrongly classified as faults (UNKNOWN excluded) | Lower is better |
| **Known FP Rate** | FP rate computed only over samples with a definitive prediction | Lower is better |
| **Fault Recall** | Fraction of actual faults correctly identified | >90% |
| **FP Reduction** | Improvement in FP rate vs. a naive (no-context) baseline | >40% |
| **Unknown Rate** | Fraction of predictions deferred as UNKNOWN | Context-dependent |

**Interpreting the results:**
- **High fault recall + low FP rate** = the model correctly identifies faults while not raising false alarms on legitimate load events.
- **FP reduction >40%** means CAAA reduces false positives by at least 40% compared to a classifier that ignores context — this is the central research claim.
- **Known FP Rate vs FP Rate**: When the model defers many predictions as UNKNOWN, the overall FP rate can appear artificially low. Use `known_fp_rate` for a coverage-adjusted view that only counts definitive predictions.
- **Known Accuracy vs Accuracy**: Similarly, `known_accuracy` excludes UNKNOWN predictions from the accuracy computation, giving a clearer picture of correctness on committed predictions.
- **Coverage-Adjusted F1**: This composite metric (`f1 × (1 - unknown_rate)`) rewards models that are both correct and decisive. A model that classifies everything as UNKNOWN would score 0, while a perfect model with no UNKNOWN predictions would score 1.0.

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

### Training Features

The CAAA trainer includes several features for stable, reproducible training:

- **Gradient clipping** (default `max_grad_norm=1.0`): Prevents exploding gradients during training.
- **ReduceLROnPlateau scheduler**: Automatically halves the learning rate when validation loss stops improving.
- **Temperature calibration**: Post-hoc temperature scaling for calibrated confidence scores.
- **NaNSafeScaler**: Wraps StandardScaler to handle zero-variance features (replaces NaN/inf with 0.0).
- **Piecewise Linear Embeddings (PLE)**: Each feature is transformed into quantile-based bin encodings, giving the MLP tree-like threshold capability.
- **TADAM-style FiLM conditioning**: Gamma constrained to 1 + delta with L2 penalty, preventing multiplicative noise amplification.
- **Feature-group dropout**: Context features (slots 12-16) zeroed with 30% probability during training, forcing the model to learn from metrics alone.
- **Counterfactual baselines**: Reference baselines generated with the same RNG seed but no fault/load injection, providing genuine "what would normal look like?" comparisons.
- **Reproducible RNG**: All data generators use instance-level `np.random.default_rng(seed)` — multiple generators can coexist in the same process without corrupting each other's random state. Per-case seeds further ensure that generation order does not create statistical dependence between data splits.
- **Unscaled tree-based baselines**: When the `--baseline` flag is used, tree-based models (RandomForest, XGBoost) are trained on unscaled features, consistent with their scale-invariant nature. Only neural models (CAAA) receive StandardScaler-transformed features.

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
| `--loss-variant` | gated | Loss penalty variant: `gated`, `clamp`, or `full` |
| `--film-mode` | tadam | FiLM conditioning: `tadam`, `additive`, or `multiplicative` |
| `--context-dropout` | 0.3 | Probability of zeroing context features during training |
| `--unknown-weight` | 0.2 | Weight for unknown-context penalty |

### Using a Config File

Instead of passing flags, you can use the provided config:

```bash
python scripts/train.py --config configs/config.yaml --baseline
```

See `configs/config.yaml` for all tunable parameters.

---

## 7. Running Ablation Studies

The ablation study is the most important experiment — it systematically evaluates 20 model variants to answer the research questions. Two are optional: `CAAA (pretrained)` requires `--pretrain` on RCAEval runs, and `CAAA (temporal)` requires `--temporal`.

### Basic ablation

```bash
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5
```

This trains each of the 20 variants 5 times with different random seeds and reports mean ± standard deviation.

### With SHAP feature importance

```bash
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5 --shap
```

Generates SHAP beeswarm plots showing which features matter most for each model variant. Plots are saved under `outputs/results/shap_*` directories.

### With calibration analysis

```bash
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5 --calibration
```

Generates reliability diagrams before and after temperature scaling. Saved under `outputs/results/calibration_*` directories.

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

### Scaling study

To see how CAAA and tree baselines perform as training data size increases:

```bash
# Default difficulty
python scripts/scaling_study.py

# Moderate difficulty (old hard profile)
python scripts/scaling_study.py --difficulty moderate

# Hard difficulty (highest-difficulty profile)
python scripts/scaling_study.py --difficulty hard

# Dump per-seed arrays for Wilcoxon/Cliff's delta tests
python scripts/scaling_study.py --difficulty hard --log-per-seed
```

This runs the full CAAA pipeline and key baselines across a range of dataset sizes and plots performance curves.

Scaling outputs are written to `outputs/results/` with difficulty suffixes:
- default: `scaling_study.csv`, `scaling_curve.png`, `context_contribution.png`, `scaling_per_seed.json`
- moderate: `scaling_study_moderate.csv`, `scaling_curve_moderate.png`, `context_contribution_moderate.png`, `scaling_per_seed_moderate.json`
- hard: `scaling_study_hard.csv`, `scaling_curve_hard.png`, `context_contribution_hard.png`, `scaling_per_seed_hard.json`

### Ablation Variants

The ablation evaluates these 20 variants:

| # | Variant | What It Tests |
|---|---------|--------------|
| 1 | Full CAAA | Complete proposed model (all features + context loss) |
| 2 | CAAA + Contrastive | Adds contrastive learning objective |
| 3 | CAAA (clamp loss) | Context consistency with clamp-only penalty |
| 4 | CAAA (full penalty) | Aggressive unknown-context penalty |
| 5 | No Context Features | Removes the 5 context features (dims 12–16) |
| 6 | No Context Loss | Uses standard cross-entropy instead of Context Consistency Loss |
| 7 | No Behavioral | Removes behavioral features (dims 6–11) |
| 8 | Context Only | Uses only context features |
| 9 | Statistical Only | Uses only statistical features |
| 10 | Stat + Service-Level | Uses statistical + service-level features |
| 11 | Baseline RF | Random Forest with context columns removed |
| 12 | XGBoost | XGBoost with context columns removed |
| 13 | LightGBM | LightGBM with context columns removed |
| 14 | CatBoost | CatBoost with context columns removed |
| 15 | CatBoost (with context) | Context-enabled tree upper-bound reference |
| 16 | CAAA+CatBoost Hybrid | CAAA embeddings + raw features into CatBoost |
| 17 | CAAA (pretrained) | Synthetic pretrain then RCAEval fine-tune (`--pretrain`) |
| 18 | CAAA (temporal) | Temporal encoder branch (`--temporal`) |
| 19 | Rule-Based | Hand-crafted heuristic rules |
| 20 | Naive | No-context baseline (always predicts FAULT) |

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

CAAA has an optional Stage 1 — an LSTM autoencoder that filters normal windows before classification. The pre-filter checks **all services** in each case and takes the maximum anomaly score, so a fault in any service triggers detection:

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
    input_dim: 44         # Feature vector size (don't change unless modifying features)
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

### Reading ablation results

Open `outputs/results/ablation_results_synthetic.csv` in a spreadsheet or with Python:

```python
import pandas as pd
df = pd.read_csv("outputs/results/ablation_results_synthetic.csv")
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

Look for context-related features (the context feature block at dims 12-16) appearing high in the ranking — this confirms context integration is valuable.

### Interpreting calibration plots

Reliability diagrams show whether predicted probabilities match actual outcomes:

- A well-calibrated model follows the diagonal line
- Points above the diagonal = under-confident, below = over-confident
- Compare "uncalibrated" vs. "calibrated" (after temperature scaling) to see improvement

### Interpreting the FP-vs-threshold curve

The Coverage vs FP Rate trade-off plot (`plot_fp_vs_threshold`) shows two curves as a function of the confidence threshold:

- **Coverage** (blue): Fraction of samples that receive a definitive (non-UNKNOWN) prediction
- **FP Rate (known)** (red, dashed): False positive rate computed only over definitive predictions

This is the key diagnostic for comparing fixed and adaptive threshold strategies. A good model maintains low FP rate even at high coverage levels.

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
| `tests/test_features.py` | Feature extraction, 44-dim schema validation |
| `tests/test_models.py` | CAAA model, context module, feature encoder |
| `tests/test_integration.py` | End-to-end training and evaluation pipeline |
| `tests/test_plan_modules.py` | Sklearn classifier wrappers |
| `tests/test_rcaeval_pipeline.py` | RCAEval data loading and processing |
| `tests/test_review_fixes.py` | Code review fix verification (device detection, vectorization, type hints, exports, performance) |

Run individual modules for faster feedback:

```bash
python -m pytest tests/test_models.py -v        # ~10 seconds
python -m pytest tests/test_features.py -v       # ~5 seconds
python -m pytest tests/test_integration.py -v    # ~30 seconds
python -m pytest tests/test_review_fixes.py -v   # ~5 seconds
```

---

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'torch'` or PyTorch DLL error on Windows

PyTorch was not installed, or was installed with the wrong CUDA build. Install it explicitly with the right index URL **before** running `pip install -r requirements.txt`:

**GPU (CUDA 12.x):**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**CPU-only:**

```powershell
pip install torch torchvision torchaudio
```

> The `cu121` index does not have wheels for Python 3.11+. Use `cu126` for Python 3.11–3.13.

Verify GPU is detected after installing:

```python
python -c "import torch; print(torch.cuda.is_available())"
```

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

### `ImportError: ruptures is required for change-point detection features`

The `ruptures` library is required for the PELT change-point detection used in feature extraction. Install it with:

```bash
pip install ruptures>=1.1.0
```

Or reinstall all dependencies from `requirements.txt`.

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

All three dataset suites (RE1, RE2, RE3) are downloaded directly from Zenodo with no additional packages required.

If behind a proxy, configure `HTTP_PROXY` and `HTTPS_PROXY` environment variables.

---

## 14. Suggested Experiment Workflow

Here is a recommended workflow from initial exploration to paper-ready results:

### Phase 1: Familiarization (~5 min)

```bash
# 1. Install PyTorch (GPU with CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. Install remaining dependencies
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
    --shap --calibration --include-hard

# 6. Review results
cat outputs/results/ablation_results_synthetic.csv
```

### Phase 4: Real-World Validation (~20 min)

```bash
# 7. Download and evaluate on RCAEval (all datasets/systems)
python -m src.main --download-data --dataset all --system all

# 8. Ablation on all real data (per-system + combined summary)
python scripts/ablation.py --data rcaeval --dataset all --system all \
    --epochs 50 --n-runs 10
```

### Phase 5: Paper-Quality Results (~2–3 hours)

```bash
# 9. High-quality synthetic ablation with all analyses
python scripts/ablation.py \
    --n-fault 200 --n-load 200 \
    --epochs 100 --n-runs 10 \
    --shap --calibration --include-hard

# 10. Full RCAEval ablation across all datasets and systems
python scripts/ablation.py --data rcaeval --dataset all --system all \
    --epochs 50 --n-runs 10
```

---

## 15. Extending CAAA

### Adding a new feature

1. Add the feature name to `src/features/feature_schema.py` (in the appropriate group)
2. Implement extraction in `src/features/extractors.py`
3. Update `N_FEATURES` and `ALL_FEATURE_NAMES` in the schema
4. Update `input_dim` in `configs/config.yaml`
5. Update tests with the new feature count

### Adding a new baseline

1. Add a class to `src/models/baseline.py` following the existing pattern (lazy import, `fit`/`predict`/`predict_proba` interface)
2. Export it in `src/models/__init__.py`
3. Add a runner function in `scripts/ablation.py`

### Adding a new fault type

1. Add a method to `src/data_loader/fault_generator.py`
2. Register it in the `FAULT_TYPES` list
3. Add test coverage in `tests/test_data_loader.py`

---

For more details on the research motivation, architecture, and related work, see the main [README.md](README.md).
