#!/usr/bin/env python3
"""CAAA Ablation Study - Systematic evaluation of model variants."""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold

# Fallback for running without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_combined_dataset, generate_rcaeval_dataset
from src.features import FeatureExtractor
from src.features.feature_schema import (
    CONTEXT_RANGE,
    BEHAVIORAL_RANGE,
    WORKLOAD_RANGE,
    STATISTICAL_RANGE,
    SERVICE_LEVEL_RANGE,
    N_FEATURES,
)
from src.models import CAAAModel, BaselineClassifier, NaiveBaseline, RuleBasedBaseline, XGBoostBaseline, LightGBMBaseline, CatBoostBaseline
from src.training.trainer import CAAATrainer
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_false_positive_rate,
)
from src.utils import set_seed, NaNSafeScaler


def run_caaa_variant(
    X_train, y_train, X_test, y_test, naive_fp, epochs, batch_size, lr,
    use_context_loss=True, loss_type="context_consistency", seed=42,
    X_val=None, y_val=None, unknown_weight=0.2, loss_variant="gated",
    film_mode="tadam", context_dropout=0.3,
    use_temporal=False, raw_train=None, mask_train=None,
    raw_test=None, mask_test=None, raw_val=None, mask_val=None,
):
    """Train and evaluate a CAAA model variant.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        naive_fp: Naive baseline FP rate.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        use_context_loss: Whether to use ContextConsistencyLoss.
        loss_type: Loss function variant (context_consistency, contrastive,
            or cross_entropy).
        seed: Random seed.
        X_val: Optional validation features for early stopping.
        y_val: Optional validation labels for early stopping.
        film_mode: FiLM conditioning mode (multiplicative, additive, tadam).
        context_dropout: Probability of zeroing out context features during
            training.
        use_temporal: Whether to enable the temporal encoder branch.
        raw_train: Optional raw time-series tensors for training.
        mask_train: Optional service masks for training.
        raw_test: Optional raw time-series tensors for test.
        mask_test: Optional service masks for test.
        raw_val: Optional raw time-series tensors for validation.
        mask_val: Optional service masks for validation.

    Returns:
        Dictionary of evaluation metrics.
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CAAAModel(input_dim=N_FEATURES, hidden_dim=64, n_classes=2,
                      film_mode=film_mode, use_temporal=use_temporal)
    trainer = CAAATrainer(
        model, learning_rate=lr, device=device,
        use_context_loss=use_context_loss,
        loss_type=loss_type,
        unknown_weight=unknown_weight,
        loss_variant=loss_variant,
        context_dropout_p=context_dropout,
    )
    early_stopping_patience = 10 if X_val is not None else epochs
    trainer.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size, early_stopping_patience=early_stopping_patience,
        raw_train=raw_train, mask_train=mask_train,
        raw_val=raw_val, mask_val=mask_val,
    )
    y_pred = trainer.predict(X_test, raw=raw_test, mask=mask_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def run_baseline_rf(X_train, y_train, X_test, y_test, naive_fp, seed=42):
    """Train and evaluate RandomForest baseline.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        naive_fp: Naive baseline FP rate.
        seed: Random seed.

    Returns:
        Dictionary of evaluation metrics.
    """
    clf = BaselineClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def run_naive(X_test, y_test, naive_fp):
    """Evaluate the naive baseline (always predicts FAULT).

    Args:
        X_test: Test features.
        y_test: Test labels.
        naive_fp: Naive baseline FP rate.

    Returns:
        Dictionary of evaluation metrics.
    """
    nb = NaiveBaseline()
    y_pred = nb.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def run_rule_based(X_train, y_train, X_test, y_test, naive_fp):
    """Train and evaluate the rule-based baseline.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        naive_fp: Naive baseline FP rate.

    Returns:
        Dictionary of evaluation metrics.
    """
    clf = RuleBasedBaseline()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def run_xgboost(X_train, y_train, X_test, y_test, naive_fp, seed=42):
    """Train and evaluate the XGBoost baseline.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        naive_fp: Naive baseline FP rate.
        seed: Random seed.

    Returns:
        Dictionary of evaluation metrics.
    """
    clf = XGBoostBaseline(random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def run_lightgbm(X_train, y_train, X_test, y_test, naive_fp, seed=42):
    """Train and evaluate the LightGBM baseline."""
    clf = LightGBMBaseline(random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def run_catboost(X_train, y_train, X_test, y_test, naive_fp, seed=42):
    """Train and evaluate the CatBoost baseline."""
    clf = CatBoostBaseline(random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def pretrain_on_synthetic(
    n_fault=500, n_load=500, seed=42, epochs=50, batch_size=32, lr=0.001,
    film_mode="tadam", unknown_weight=0.2, context_dropout=0.3,
):
    """Pre-train CAAA on synthetic data and return checkpoint path.

    Generates synthetic data with genuine external context and trains a CAAA
    model. The checkpoint can then be loaded for fine-tuning on real data.
    """
    save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "pretrained")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"caaa_pretrained_synthetic_{n_fault}f_{n_load}l_seed{seed}.pt",
    )

    fault_cases, load_cases = generate_combined_dataset(
        n_fault=n_fault, n_load=n_load,
        systems=["online-boutique", "sock-shop", "train-ticket"],
        seed=seed, include_hard=True,
    )
    all_cases = fault_cases + load_cases
    labels = np.array([0 if c.label == "FAULT" else 1 for c in all_cases])

    extractor = FeatureExtractor(context_mode="external")
    X = extractor.extract_batch(all_cases).astype(np.float32)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, labels, test_size=0.15, random_state=seed, stratify=labels,
    )
    scaler = NaNSafeScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CAAAModel(
        input_dim=N_FEATURES, hidden_dim=64, n_classes=2, film_mode=film_mode,
    )
    trainer = CAAATrainer(
        model, learning_rate=lr, device=device,
        use_context_loss=True, loss_type="context_consistency",
        unknown_weight=unknown_weight, loss_variant="gated",
        context_dropout_p=context_dropout,
    )
    trainer.train(
        X_train, y_train, X_val=X_val, y_val=y_val,
        epochs=epochs, batch_size=batch_size, early_stopping_patience=15,
    )
    trainer.save_model(save_path)
    print(f"  Pre-trained model saved to {save_path}")
    return save_path


def run_caaa_pretrained(
    X_train, y_train, X_test, y_test, naive_fp,
    pretrain_path, epochs=30, batch_size=32, lr=0.0003,
    seed=42, X_val=None, y_val=None,
    unknown_weight=0.2, loss_variant="clamp",
    film_mode="tadam", context_dropout=0.3,
):
    """Fine-tune a pre-trained CAAA model on real data.

    Loads only model weights from the checkpoint (discards optimizer state)
    and trains with a lower learning rate to preserve representations.
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CAAAModel(
        input_dim=N_FEATURES, hidden_dim=64, n_classes=2, film_mode=film_mode,
    )
    checkpoint = torch.load(pretrain_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    trainer = CAAATrainer(
        model, learning_rate=lr, device=device,
        use_context_loss=True, loss_type="context_consistency",
        unknown_weight=unknown_weight, loss_variant=loss_variant,
        context_dropout_p=context_dropout,
    )
    trainer.train(
        X_train, y_train, X_val=X_val, y_val=y_val,
        epochs=epochs, batch_size=batch_size, early_stopping_patience=10,
    )
    y_pred = trainer.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


_SYSTEM_ABBREV = {
    "online-boutique": "OB",
    "sock-shop": "SS",
    "train-ticket": "TT",
}


def merge_results(csv_paths):
    """Merge per-system ablation CSVs into a combined summary table.

    Args:
        csv_paths: List of (dataset, system, csv_path) tuples.
    """
    import csv as _csv

    # Read each CSV
    per_system = {}
    variants = None
    for ds, sys_name, path in csv_paths:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        abbrev = f"{ds}_{_SYSTEM_ABBREV.get(sys_name, sys_name)}"
        with open(path) as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        if variants is None:
            variants = [r["variant"] for r in rows]
        per_system[abbrev] = {r["variant"]: r for r in rows}

    if not per_system or not variants:
        return

    # Key metrics to show in combined table
    key_metrics = ["f1_mean", "mcc_mean", "fp_rate_mean", "fault_recall_mean", "fp_reduction_mean"]

    # Write combined CSV
    combined_dir = "outputs/results"
    combined_path = os.path.join(combined_dir, "ablation_results_combined.csv")
    sys_keys = sorted(per_system.keys())

    with open(combined_path, "w", newline="") as f:
        writer = _csv.writer(f)
        header = ["variant"]
        for sk in sys_keys:
            for m in key_metrics:
                header.append(f"{sk}_{m}")
        for m in key_metrics:
            header.append(f"Avg_{m}")
        writer.writerow(header)

        for v in variants:
            row = [v]
            for sk in sys_keys:
                data = per_system[sk].get(v, {})
                for m in key_metrics:
                    row.append(data.get(m, ""))
            # Macro-average
            for m in key_metrics:
                vals = []
                for sk in sys_keys:
                    data = per_system[sk].get(v, {})
                    val = data.get(m, "")
                    if val:
                        vals.append(float(val))
                row.append(f"{np.mean(vals):.4f}" if vals else "")
            writer.writerow(row)

    print(f"\nCombined results saved to {combined_path}")

    # Print summary console table
    print()
    print("=" * 100)
    print("COMBINED ABLATION RESULTS (macro-averaged across systems)")
    print("=" * 100)
    col_w = 10
    header_str = f"{'Variant':<25s}"
    for sk in sys_keys:
        header_str += f"{sk + ' F1':>{col_w}s}"
    header_str += f"{'Avg F1':>{col_w}s}{'Avg MCC':>{col_w}s}{'Avg FPR':>{col_w}s}"
    print(header_str)
    print("-" * 100)
    for v in variants:
        line = f"{v:<25s}"
        for sk in sys_keys:
            data = per_system[sk].get(v, {})
            val = data.get("f1_mean", "")
            line += f"{float(val):>{col_w}.3f}" if val else f"{'N/A':>{col_w}s}"
        # Averages
        for m in ["f1_mean", "mcc_mean", "fp_rate_mean"]:
            vals = [float(per_system[sk].get(v, {}).get(m, 0)) for sk in sys_keys if per_system[sk].get(v, {}).get(m)]
            avg = f"{np.mean(vals):.3f}" if vals else "N/A"
            line += f"{avg:>{col_w}s}"
        print(line)
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="CAAA Ablation Study")
    parser.add_argument("--n-fault", type=int, default=50)
    parser.add_argument("--n-load", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--systems", nargs="+", default=["online-boutique"])

    # Data source
    parser.add_argument(
        "--data", type=str, default="synthetic",
        choices=["synthetic", "rcaeval"],
        help="Data source: synthetic (default) or rcaeval (real faults)",
    )
    parser.add_argument("--include-hard", action="store_true",
                        help="Include hard/adversarial scenarios in dataset")
    parser.add_argument("--dataset", type=str, default="RE1",
                        choices=["RE1", "RE2", "RE3", "all"], help="RCAEval dataset")
    parser.add_argument("--system", type=str, default="online-boutique",
                        choices=["online-boutique", "sock-shop", "train-ticket", "all"],
                        help="Microservice system (for rcaeval)")
    parser.add_argument("--load-ratio", type=int, default=1,
                        help="Synthetic loads per RCAEval fault")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="RCAEval data directory")
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="Number of CV folds (0 = use n-runs with train/test split)")
    parser.add_argument("--shap", action="store_true",
                        help="Generate SHAP plots for Full CAAA and Baseline RF")
    parser.add_argument("--calibration", action="store_true",
                        help="Generate reliability diagrams comparing ECE before/after temperature scaling")
    parser.add_argument("--unknown-weight", type=float, default=0.2,
                        help="Weight for unknown-context penalty in ContextConsistencyLoss")
    parser.add_argument("--loss-variant", type=str, default="gated",
                        choices=["clamp", "gated", "full"],
                        help="ContextConsistencyLoss penalty variant")
    parser.add_argument("--film-mode", type=str, default="tadam",
                        choices=["multiplicative", "additive", "tadam"],
                        help="FiLM conditioning mode")
    parser.add_argument("--context-dropout", type=float, default=0.3,
                        help="Probability of zeroing out context features during training")
    parser.add_argument("--temporal", action="store_true",
                        help="Enable temporal encoder branch (processes raw time-series)")
    parser.add_argument("--pool", type=str, default="none",
                        choices=["none", "systems", "all"],
                        help="Pooling scope: none (per-system), systems (pool 3 systems within dataset), all (all datasets x all systems)")
    parser.add_argument("--pretrain", action="store_true",
                        help="Pre-train on synthetic data before fine-tuning on real")

    args = parser.parse_args()

    # Auto-clamp context loss on real data where context is noise
    if args.data == "rcaeval" and args.loss_variant == "gated":
        args.loss_variant = "clamp"
        print("  [Auto] Using loss_variant='clamp' for RCAEval (context is derived, not external)")

    # Handle --dataset all / --system all by recursing into per-combo runs.
    # Pooling modes change the recursion:
    #   --pool all:     no recursion — single run loads everything
    #   --pool systems: recurse per-dataset only (each run pools 3 systems)
    #   --pool none:    original per-(dataset, system) recursion
    if args.data == "rcaeval" and (args.dataset == "all" or args.system == "all"):
        # --pool all: skip recursion entirely, load all data in one run
        if args.pool == "all":
            print("  [Pool=all] Loading all datasets x all systems in one run")
            args.dataset = "RE1"  # placeholder — overridden by pool=all below
            args.system = "online-boutique"
            # Fall through to main loop (pooling override happens later)
        else:
            datasets = ["RE1", "RE2", "RE3"] if args.dataset == "all" else [args.dataset]
            if args.pool == "systems":
                # Pool all systems within each dataset — one run per dataset
                systems = ["online-boutique"]  # placeholder — overridden by pool=systems
            else:
                systems = (
                    ["online-boutique", "sock-shop", "train-ticket"]
                    if args.system == "all" else [args.system]
                )
            csv_paths = []
            import subprocess
            for ds in datasets:
                for sys_name in systems:
                    label = f"{ds} / all-systems-pooled" if args.pool == "systems" else f"{ds} / {sys_name}"
                    print(f"\n{'='*60}")
                    print(f"  ABLATION: {label}")
                    print(f"{'='*60}")
                    cmd = [sys.executable, __file__]
                    skip_next = False
                    for j, a in enumerate(sys.argv[1:]):
                        if skip_next:
                            skip_next = False
                            continue
                        if a == "--dataset":
                            cmd.extend(["--dataset", ds])
                            skip_next = True
                        elif a == "--system":
                            cmd.extend(["--system", sys_name])
                            skip_next = True
                        else:
                            cmd.append(a)
                    if "--dataset" not in cmd:
                        cmd.extend(["--dataset", ds])
                    if "--system" not in cmd:
                        cmd.extend(["--system", sys_name])
                    result = subprocess.run(cmd, check=False)
                    if result.returncode != 0:
                        print(f"  Warning: ablation for {label} failed (exit {result.returncode})")
                    if args.pool == "systems":
                        suffix = f"{ds}_pooled-systems"
                        csv_paths.append((ds, "pooled", f"outputs/results/ablation_results_{suffix}.csv"))
                    else:
                        suffix = f"{ds}_{sys_name}"
                        csv_paths.append((ds, sys_name, f"outputs/results/ablation_results_{suffix}.csv"))
            merge_results(csv_paths)
            return

    metrics_to_track = ["accuracy", "f1", "f1_macro", "mcc", "fp_rate", "fault_recall", "fp_reduction"]

    # Variant definitions
    variants = [
        "Full CAAA",
        "CAAA + Contrastive",
        "CAAA (clamp loss)",
        "CAAA (full penalty)",
        "No Context Features",
        "No Context Loss",
        "No Behavioral",
        "Context Only",
        "Statistical Only",
        "Stat + Service-Level",
        "Baseline RF",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "CatBoost (with context)",
        "CAAA+CatBoost Hybrid",
        "CAAA (pretrained)",
        "CAAA (temporal)",
        "Rule-Based",
        "Naive",
    ]

    # Collect results: {variant: {metric: [values across runs]}}
    all_results = {v: {m: [] for m in metrics_to_track} for v in variants}

    # Print configuration summary
    print()
    print("=" * 70)
    print("  CAAA ABLATION STUDY")
    print("=" * 70)
    data_label = args.data.upper()
    if args.data == "rcaeval":
        if args.pool == "all":
            data_label += " (all datasets x all systems, pooled)"
        elif args.pool == "systems":
            data_label += f" ({args.dataset} x all systems, pooled)"
        else:
            data_label += f" ({args.dataset} / {args.system})"
    print(f"  Data:       {data_label}")
    print(f"  Runs:       {args.n_runs}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  FiLM mode:  {args.film_mode}")
    if args.pretrain and args.data == "rcaeval":
        print(f"  Pre-train:  Yes (500+500 synthetic)")
    if args.temporal:
        print(f"  Temporal:   Yes (LITE 1D-CNN + cross-service attention)")
    print("=" * 70)

    # Pre-train on synthetic data once if requested (reused across all runs)
    pretrain_path = None
    if args.pretrain and args.data == "rcaeval":
        print("\n  Pre-training on synthetic data (500 fault + 500 load)...")
        pretrain_path = pretrain_on_synthetic(
            n_fault=500, n_load=500, seed=args.base_seed,
            epochs=50, batch_size=32, lr=0.001,
            film_mode=args.film_mode,
            unknown_weight=args.unknown_weight,
            context_dropout=args.context_dropout,
        )

    use_cv = args.cv_folds > 0
    n_iterations = args.cv_folds if use_cv else args.n_runs

    for run_idx in range(n_iterations if not use_cv else 1):
        run_seed = args.base_seed + run_idx
        if not use_cv:
            bar_len = 20
            filled = int(bar_len * run_idx / args.n_runs)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\n  [{bar}] Run {run_idx + 1}/{args.n_runs} (seed={run_seed})")

        set_seed(run_seed if not use_cv else args.base_seed)

        # Generate dataset
        if args.data == "rcaeval":
            # Apply pooling overrides
            rcaeval_dataset = args.dataset
            rcaeval_system = args.system
            if args.pool == "systems":
                rcaeval_system = ["online-boutique", "sock-shop", "train-ticket"]
            elif args.pool == "all":
                rcaeval_dataset = ["RE1", "RE2", "RE3"]
                rcaeval_system = ["online-boutique", "sock-shop", "train-ticket"]
            fault_cases, load_cases = generate_rcaeval_dataset(
                dataset=rcaeval_dataset, system=rcaeval_system,
                n_load_per_fault=args.load_ratio,
                data_dir=args.data_dir, seed=run_seed if not use_cv else args.base_seed,
            )
        else:
            fault_cases, load_cases = generate_combined_dataset(
                n_fault=args.n_fault, n_load=args.n_load,
                systems=args.systems, seed=run_seed if not use_cv else args.base_seed,
                include_hard=args.include_hard,
            )
        all_cases = fault_cases + load_cases
        labels = np.array([0 if c.label == "FAULT" else 1 for c in all_cases])
        fault_types_all = np.array([c.fault_type for c in all_cases])
        difficulties_all = np.array([c.difficulty or "unknown" for c in all_cases])
        if run_idx == 0:
            print(f"  Dataset: {len(fault_cases)} fault + {len(load_cases)} load = {len(all_cases)} samples")

        # Use external context for synthetic data (where the generator creates
        # genuine operational context alongside the data) and comparison context
        # for real data (where context must be derived from the metrics).
        ctx_mode = "external" if args.data == "synthetic" else "comparison"
        extractor = FeatureExtractor(context_mode=ctx_mode)
        X = extractor.extract_batch(all_cases).astype(np.float32)

        # Extract raw time-series tensors for temporal encoder (if enabled)
        raw_tensors = None
        raw_masks = None
        if args.temporal:
            from src.features.raw_tensor_extractor import RawTensorExtractor
            raw_extractor = RawTensorExtractor(max_services=20, seq_len=120)
            raw_tensors, raw_masks = raw_extractor.extract_batch(all_cases)
            if run_idx == 0:
                print(f"  Raw tensors: {raw_tensors.shape} ({raw_tensors.nbytes / 1e6:.1f} MB)")

        if use_cv:
            # Generate all CV folds once — shared across all variants
            skf = StratifiedKFold(
                n_splits=args.cv_folds, shuffle=True, random_state=args.base_seed,
            )
            folds = list(skf.split(X, labels))
        else:
            # Single train/test split
            train_idx, test_idx = train_test_split(
                np.arange(len(labels)), test_size=0.2,
                random_state=run_seed, stratify=labels,
            )
            folds = [(train_idx, test_idx)]

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            if use_cv:
                print(f"\n--- Fold {fold_idx + 1}/{args.cv_folds} ---")

            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train_all, y_test = labels[train_idx], labels[test_idx]
            fault_types_test = fault_types_all[test_idx]
            difficulties_test = difficulties_all[test_idx]

            # Split a validation set (12.5% of training data) for early
            # stopping and calibration — mirrors the split in train.py.
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                X_train_raw, y_train_all, test_size=0.125,
                random_state=run_seed if not use_cv else args.base_seed,
                stratify=y_train_all,
            )

            # Split raw tensors/masks with same indices (for temporal encoder)
            raw_tr = raw_te = raw_va = mask_tr = mask_te = mask_va = None
            if raw_tensors is not None:
                raw_all_train = raw_tensors[train_idx]
                raw_te = raw_tensors[test_idx]
                mask_all_train = raw_masks[train_idx]
                mask_te = raw_masks[test_idx]
                # Further split train -> train + val (same indices as X split)
                raw_tr, raw_va, mask_tr, mask_va = train_test_split(
                    raw_all_train, mask_all_train, test_size=0.125,
                    random_state=run_seed if not use_cv else args.base_seed,
                    stratify=y_train_all,
                )

            # Scale features (fit on train only) for neural models
            scaler = NaNSafeScaler()
            X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
            X_val = scaler.transform(X_val_raw).astype(np.float32)
            X_test = scaler.transform(X_test_raw).astype(np.float32)

            # Keep unscaled copies for tree-based baselines (scale-invariant)
            X_train_unscaled = X_train_raw
            X_test_unscaled = X_test_raw

            # Context-free arrays for tree baselines (39 features).
            # Delete (not zero) context columns so trees get a clean input
            # without wasting splits on constant-zero placeholder columns.
            # This represents the real-world scenario: traditional anomaly
            # detectors don't have access to operational context.
            _ctx_s, _ctx_e = CONTEXT_RANGE
            X_train_no_ctx = np.delete(X_train_unscaled, range(_ctx_s, _ctx_e), axis=1)
            X_test_no_ctx = np.delete(X_test_unscaled, range(_ctx_s, _ctx_e), axis=1)

            # Naive baseline FP rate
            naive = NaiveBaseline()
            naive_pred = naive.predict(X_test_raw)
            naive_fp = compute_false_positive_rate(y_test, naive_pred)

            # --- Full CAAA ---
            print("  Full CAAA...")
            m = run_caaa_variant(
                X_train, y_train, X_test, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr,
                use_context_loss=True, seed=run_seed,
                X_val=X_val, y_val=y_val,
                unknown_weight=args.unknown_weight,
                loss_variant=args.loss_variant,
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["Full CAAA"][k].append(m.get(k, 0.0))

            # --- CAAA + Contrastive ---
            print("  CAAA + Contrastive...")
            m = run_caaa_variant(
                X_train, y_train, X_test, y_test, naive_fp,
                args.epochs, max(args.batch_size, 16), args.lr,
                loss_type="contrastive", seed=run_seed,
                X_val=X_val, y_val=y_val,
                unknown_weight=args.unknown_weight,
                loss_variant=args.loss_variant,
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["CAAA + Contrastive"][k].append(m.get(k, 0.0))

            # --- CAAA (clamp loss) ---
            print("  CAAA (clamp loss)...")
            m = run_caaa_variant(
                X_train, y_train, X_test, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr, seed=run_seed,
                X_val=X_val, y_val=y_val,
                unknown_weight=args.unknown_weight,
                loss_variant="clamp",
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["CAAA (clamp loss)"][k].append(m.get(k, 0.0))

            # --- CAAA (full penalty) ---
            print("  CAAA (full penalty)...")
            m = run_caaa_variant(
                X_train, y_train, X_test, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr, seed=run_seed,
                X_val=X_val, y_val=y_val,
                unknown_weight=0.5,
                loss_variant="full",
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["CAAA (full penalty)"][k].append(m.get(k, 0.0))

            # --- No Context Features ---
            print("  No Context Features...")
            ctx_s, ctx_e = CONTEXT_RANGE
            X_train_nc = X_train.copy()
            X_test_nc = X_test.copy()
            X_val_nc = X_val.copy()
            X_train_nc[:, ctx_s:ctx_e] = 0.0
            X_test_nc[:, ctx_s:ctx_e] = 0.0
            X_val_nc[:, ctx_s:ctx_e] = 0.0
            m = run_caaa_variant(
                X_train_nc, y_train, X_test_nc, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr,
                use_context_loss=True, seed=run_seed,
                X_val=X_val_nc, y_val=y_val,
                unknown_weight=args.unknown_weight,
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["No Context Features"][k].append(m.get(k, 0.0))

            # --- No Context Loss ---
            print("  No Context Loss...")
            m = run_caaa_variant(
                X_train, y_train, X_test, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr,
                use_context_loss=False, seed=run_seed,
                X_val=X_val, y_val=y_val,
                unknown_weight=args.unknown_weight,
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["No Context Loss"][k].append(m.get(k, 0.0))

            # --- No Behavioral Features ---
            print("  No Behavioral...")
            beh_s, beh_e = BEHAVIORAL_RANGE
            X_train_nb = X_train.copy()
            X_test_nb = X_test.copy()
            X_val_nb = X_val.copy()
            X_train_nb[:, beh_s:beh_e] = 0.0
            X_test_nb[:, beh_s:beh_e] = 0.0
            X_val_nb[:, beh_s:beh_e] = 0.0
            m = run_caaa_variant(
                X_train_nb, y_train, X_test_nb, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr,
                use_context_loss=True, seed=run_seed,
                X_val=X_val_nb, y_val=y_val,
                unknown_weight=args.unknown_weight,
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["No Behavioral"][k].append(m.get(k, 0.0))

            # --- Context Features Only ---
            print("  Context Only...")
            wkl_s, wkl_e = WORKLOAD_RANGE
            stat_s, stat_e = STATISTICAL_RANGE
            svc_s, svc_e = SERVICE_LEVEL_RANGE
            X_train_co = X_train.copy()
            X_test_co = X_test.copy()
            X_val_co = X_val.copy()
            X_train_co[:, wkl_s:beh_e] = 0.0  # zero workload + behavioral
            X_train_co[:, stat_s:svc_e] = 0.0  # zero statistical + service-level
            X_train_co[:, 36:] = 0.0           # zero extended
            X_test_co[:, wkl_s:beh_e] = 0.0
            X_test_co[:, stat_s:svc_e] = 0.0
            X_test_co[:, 36:] = 0.0
            X_val_co[:, wkl_s:beh_e] = 0.0
            X_val_co[:, stat_s:svc_e] = 0.0
            X_val_co[:, 36:] = 0.0
            m = run_caaa_variant(
                X_train_co, y_train, X_test_co, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr,
                use_context_loss=True, seed=run_seed,
                X_val=X_val_co, y_val=y_val,
                unknown_weight=args.unknown_weight,
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["Context Only"][k].append(m.get(k, 0.0))

            # --- Statistical Features Only ---
            print("  Statistical Only...")
            X_train_so = X_train.copy()
            X_test_so = X_test.copy()
            X_val_so = X_val.copy()
            # Zero out workload, behavioral, context, and service-level;
            # keep only statistical
            X_train_so[:, :stat_s] = 0.0
            X_train_so[:, stat_e:] = 0.0
            X_test_so[:, :stat_s] = 0.0
            X_test_so[:, stat_e:] = 0.0
            X_val_so[:, :stat_s] = 0.0
            X_val_so[:, stat_e:] = 0.0
            m = run_caaa_variant(
                X_train_so, y_train, X_test_so, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr,
                use_context_loss=False, seed=run_seed,
                X_val=X_val_so, y_val=y_val,
                unknown_weight=args.unknown_weight,
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["Statistical Only"][k].append(m.get(k, 0.0))

            # --- Statistical + Service-Level ---
            print("  Stat + Service-Level...")
            X_train_ssl = X_train.copy()
            X_test_ssl = X_test.copy()
            X_val_ssl = X_val.copy()
            # Zero out workload, behavioral, context;
            # keep statistical and service-level
            X_train_ssl[:, :stat_s] = 0.0
            X_test_ssl[:, :stat_s] = 0.0
            X_val_ssl[:, :stat_s] = 0.0
            m = run_caaa_variant(
                X_train_ssl, y_train, X_test_ssl, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr,
                use_context_loss=False, seed=run_seed,
                X_val=X_val_ssl, y_val=y_val,
                unknown_weight=args.unknown_weight,
                film_mode=args.film_mode,
                context_dropout=args.context_dropout,
            )
            for k in metrics_to_track:
                all_results["Stat + Service-Level"][k].append(m.get(k, 0.0))

            # --- Baseline RF (no context — fair comparison) ---
            print("  Baseline RF...")
            m = run_baseline_rf(X_train_no_ctx, y_train, X_test_no_ctx, y_test, naive_fp, seed=run_seed)
            for k in metrics_to_track:
                all_results["Baseline RF"][k].append(m.get(k, 0.0))

            # --- XGBoost (no context — fair comparison) ---
            print("  XGBoost...")
            m = run_xgboost(X_train_no_ctx, y_train, X_test_no_ctx, y_test, naive_fp, seed=run_seed)
            for k in metrics_to_track:
                all_results["XGBoost"][k].append(m.get(k, 0.0))

            # --- LightGBM (no context — fair comparison) ---
            print("  LightGBM...")
            m = run_lightgbm(X_train_no_ctx, y_train, X_test_no_ctx, y_test, naive_fp, seed=run_seed)
            for k in metrics_to_track:
                all_results["LightGBM"][k].append(m.get(k, 0.0))

            # --- CatBoost (no context — fair comparison) ---
            print("  CatBoost...")
            m = run_catboost(X_train_no_ctx, y_train, X_test_no_ctx, y_test, naive_fp, seed=run_seed)
            for k in metrics_to_track:
                all_results["CatBoost"][k].append(m.get(k, 0.0))

            # --- CatBoost (with context) — upper-bound reference ---
            print("  CatBoost (with context)...")
            m = run_catboost(X_train_unscaled, y_train, X_test_unscaled, y_test, naive_fp, seed=run_seed)
            for k in metrics_to_track:
                all_results["CatBoost (with context)"][k].append(m.get(k, 0.0))

            # --- CAAA+CatBoost Hybrid ---
            print("  CAAA+CatBoost Hybrid...")
            # Train a CAAA model first to get embeddings
            torch.manual_seed(run_seed)
            _hybrid_device = "cuda" if torch.cuda.is_available() else "cpu"
            _hybrid_model = CAAAModel(input_dim=N_FEATURES, hidden_dim=64, n_classes=2,
                                      film_mode=args.film_mode)
            _hybrid_trainer = CAAATrainer(
                _hybrid_model, learning_rate=args.lr, device=_hybrid_device,
                use_context_loss=True,
                loss_variant=args.loss_variant,
                unknown_weight=args.unknown_weight,
                context_dropout_p=args.context_dropout,
            )
            _hybrid_trainer.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                early_stopping_patience=10,
            )
            # Extract embeddings using CAAAModel.get_embeddings (returns torch.Tensor)
            _hybrid_device_obj = next(_hybrid_model.parameters()).device
            _hybrid_model.eval()
            with torch.no_grad():
                _X_train_t = torch.tensor(X_train, dtype=torch.float32, device=_hybrid_device_obj)
                _X_test_t = torch.tensor(X_test, dtype=torch.float32, device=_hybrid_device_obj)
                emb_train = _hybrid_model.get_embeddings(_X_train_t).cpu().numpy()
                emb_test = _hybrid_model.get_embeddings(_X_test_t).cpu().numpy()
            # Concatenate embeddings + raw (unscaled) features
            X_hybrid_train = np.concatenate([emb_train, X_train_unscaled], axis=1)
            X_hybrid_test = np.concatenate([emb_test, X_test_unscaled], axis=1)
            # Train CatBoost on hybrid features
            _hybrid_cb = CatBoostBaseline(random_state=run_seed)
            _hybrid_cb.fit(X_hybrid_train, y_train)
            y_pred_hybrid = _hybrid_cb.predict(X_hybrid_test)
            m = compute_all_metrics(y_test, y_pred_hybrid, baseline_fp_rate=naive_fp)
            for k in metrics_to_track:
                all_results["CAAA+CatBoost Hybrid"][k].append(m.get(k, 0.0))

            # --- CAAA (pretrained) — fine-tune pre-trained model on real data ---
            if pretrain_path is not None:
                print("  CAAA (pretrained)...")
                m = run_caaa_pretrained(
                    X_train, y_train, X_test, y_test, naive_fp,
                    pretrain_path=pretrain_path,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=0.0003, seed=run_seed,
                    X_val=X_val, y_val=y_val,
                    unknown_weight=args.unknown_weight,
                    loss_variant=args.loss_variant,
                    film_mode=args.film_mode,
                    context_dropout=args.context_dropout,
                )
                for k in metrics_to_track:
                    all_results["CAAA (pretrained)"][k].append(m.get(k, 0.0))
            else:
                for k in metrics_to_track:
                    all_results["CAAA (pretrained)"][k].append(0.0)

            # --- CAAA (temporal) — temporal encoder + cross-service attention ---
            if args.temporal and raw_tr is not None:
                print("  CAAA (temporal)...")
                m = run_caaa_variant(
                    X_train, y_train, X_test, y_test, naive_fp,
                    args.epochs, args.batch_size, args.lr,
                    use_context_loss=True, seed=run_seed,
                    X_val=X_val, y_val=y_val,
                    unknown_weight=args.unknown_weight,
                    loss_variant=args.loss_variant,
                    film_mode=args.film_mode,
                    context_dropout=args.context_dropout,
                    use_temporal=True,
                    raw_train=raw_tr, mask_train=mask_tr,
                    raw_test=raw_te, mask_test=mask_te,
                    raw_val=raw_va, mask_val=mask_va,
                )
                for k in metrics_to_track:
                    all_results["CAAA (temporal)"][k].append(m.get(k, 0.0))
            else:
                for k in metrics_to_track:
                    all_results["CAAA (temporal)"][k].append(0.0)

            # --- Rule-Based (uses context — tests simple context rules) ---
            print("  Rule-Based...")
            m = run_rule_based(X_train_unscaled, y_train, X_test_unscaled, y_test, naive_fp)
            for k in metrics_to_track:
                all_results["Rule-Based"][k].append(m.get(k, 0.0))

            # --- Naive (uses unscaled features) ---
            print("  Naive...")
            m = run_naive(X_test_unscaled, y_test, naive_fp)
            for k in metrics_to_track:
                all_results["Naive"][k].append(m.get(k, 0.0))

    # Compute mean ± std
    summary = {}
    for v in variants:
        summary[v] = {}
        for m in metrics_to_track:
            vals = all_results[v][m]
            summary[v][m + "_mean"] = np.mean(vals)
            summary[v][m + "_std"] = np.std(vals)

    # Print table
    n_evals = args.cv_folds if use_cv else args.n_runs
    eval_label = f"{args.cv_folds}-fold CV" if use_cv else f"{args.n_runs} runs"
    print()
    print()
    print("=" * 106)
    print(f"  ABLATION STUDY RESULTS (mean ± std over {eval_label})")
    print("=" * 106)
    header = f"  {'Variant':<24s}{'Accuracy':>12s}{'F1':>12s}{'F1 Macro':>12s}{'MCC':>12s}{'FP Rate':>12s}{'Recall':>12s}{'FP Red.':>12s}"
    print(header)
    print("  " + "-" * 102)

    # Group variants for visual clarity
    _ablation_variants = {"No Context Features", "No Context Loss", "No Behavioral",
                          "Context Only", "Statistical Only", "Stat + Service-Level"}
    _tree_variants = {"Baseline RF", "XGBoost", "LightGBM", "CatBoost",
                      "CatBoost (with context)", "CAAA+CatBoost Hybrid"}
    _printed_sep = set()

    for v in variants:
        s = summary[v]
        # Skip variants that were not run (all zeros)
        if s["accuracy_mean"] == 0.0 and s["accuracy_std"] == 0.0:
            continue

        # Add separator between groups
        if v in _ablation_variants and "ablation" not in _printed_sep:
            print("  " + "·" * 102)
            _printed_sep.add("ablation")
        elif v in _tree_variants and "tree" not in _printed_sep:
            print("  " + "·" * 102)
            _printed_sep.add("tree")
        elif v in {"Rule-Based", "Naive"} and "baseline" not in _printed_sep:
            print("  " + "·" * 102)
            _printed_sep.add("baseline")

        acc = f"{s['accuracy_mean']:.2f}±{s['accuracy_std']:.2f}"
        f1 = f"{s['f1_mean']:.2f}±{s['f1_std']:.2f}"
        f1m = f"{s['f1_macro_mean']:.2f}±{s['f1_macro_std']:.2f}"
        mcc = f"{s['mcc_mean']:.2f}±{s['mcc_std']:.2f}"
        fpr = f"{s['fp_rate_mean']:.2f}±{s['fp_rate_std']:.2f}"
        fr = f"{s['fault_recall_mean']:.2f}±{s['fault_recall_std']:.2f}"
        fpred = f"{s['fp_reduction_mean']*100:.1f}±{s['fp_reduction_std']*100:.1f}%"

        # Highlight key variants
        marker = ">>>" if v == "Full CAAA" else "   "
        if v == "CAAA (pretrained)" or v == "CAAA (temporal)":
            marker = " * "
        print(f"{marker} {v:<23s}{acc:>12s}{f1:>12s}{f1m:>12s}{mcc:>12s}{fpr:>12s}{fr:>12s}{fpred:>12s}")
    print("=" * 106)

    # Print key comparison summary
    caaa_f1 = summary["Full CAAA"]["f1_mean"]
    best_tree_name = max(
        [v for v in ["Baseline RF", "XGBoost", "LightGBM", "CatBoost"] if summary[v]["accuracy_mean"] > 0],
        key=lambda v: summary[v]["f1_mean"],
    )
    best_tree_f1 = summary[best_tree_name]["f1_mean"]
    delta = (caaa_f1 - best_tree_f1) * 100
    sign = "+" if delta >= 0 else ""
    print(f"\n  Key comparison: Full CAAA ({caaa_f1:.1%}) vs {best_tree_name} ({best_tree_f1:.1%}) = {sign}{delta:.1f}pp")
    if "CAAA (pretrained)" in summary and summary["CAAA (pretrained)"]["accuracy_mean"] > 0:
        pt_f1 = summary["CAAA (pretrained)"]["f1_mean"]
        delta_pt = (pt_f1 - best_tree_f1) * 100
        sign_pt = "+" if delta_pt >= 0 else ""
        print(f"  Key comparison: CAAA pretrained ({pt_f1:.1%}) vs {best_tree_name} ({best_tree_f1:.1%}) = {sign_pt}{delta_pt:.1f}pp")
    if "CAAA (temporal)" in summary and summary["CAAA (temporal)"]["accuracy_mean"] > 0:
        temp_f1 = summary["CAAA (temporal)"]["f1_mean"]
        delta_temp = (temp_f1 - best_tree_f1) * 100
        sign_temp = "+" if delta_temp >= 0 else ""
        print(f"  Key comparison: CAAA temporal ({temp_f1:.1%}) vs {best_tree_name} ({best_tree_f1:.1%}) = {sign_temp}{delta_temp:.1f}pp")

    # Per-fault-type breakdown for Full CAAA on last fold
    print()
    print("=" * 80)
    print("PER-FAULT-TYPE BREAKDOWN (Full CAAA, last fold)")
    print("=" * 80)
    print(f"{'Fault Type':<30s}{'Count':>8s}{'Accuracy':>12s}{'Recall':>12s}{'Misclass→LOAD':>15s}")
    print("-" * 80)

    # Re-run Full CAAA on last fold to get predictions for breakdown
    torch.manual_seed(args.base_seed)
    _ft_device = "cuda" if torch.cuda.is_available() else "cpu"
    _ft_model = CAAAModel(input_dim=N_FEATURES, hidden_dim=64, n_classes=2,
                          film_mode=args.film_mode)
    _ft_trainer = CAAATrainer(
        _ft_model, learning_rate=args.lr, device=_ft_device,
        use_context_loss=True,
        unknown_weight=args.unknown_weight,
        loss_variant=args.loss_variant,
        context_dropout_p=args.context_dropout,
    )
    _ft_trainer.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size, early_stopping_patience=10,
    )
    _ft_pred = _ft_trainer.predict(X_test)

    unique_ft = sorted(set(ft for ft in fault_types_test if ft is not None))
    for ft in unique_ft:
        ft_mask = fault_types_test == ft
        if ft_mask.sum() == 0:
            continue
        y_true_ft = y_test[ft_mask]
        y_pred_ft = _ft_pred[ft_mask]
        count = int(ft_mask.sum())
        acc = float((y_true_ft == y_pred_ft).mean())
        # Recall: fraction of true-FAULT correctly predicted as FAULT
        fault_mask = y_true_ft == 0
        recall = float((y_pred_ft[fault_mask] == 0).mean()) if fault_mask.sum() > 0 else 0.0
        # Misclassification as EXPECTED_LOAD
        misclass = float((y_pred_ft[fault_mask] == 1).mean()) if fault_mask.sum() > 0 else 0.0
        print(f"{ft:<30s}{count:>8d}{acc:>12.3f}{recall:>12.3f}{misclass:>15.3f}")
    print("=" * 80)

    # Per-difficulty breakdown (Full CAAA, last fold)
    unique_diffs = sorted(set(d for d in difficulties_test if d != "unknown"))
    if unique_diffs:
        print()
        print("=" * 80)
        print("PER-DIFFICULTY BREAKDOWN (Full CAAA, last fold)")
        print("=" * 80)
        print(f"{'Difficulty':<15s}{'Count':>8s}{'Accuracy':>12s}{'FP Rate':>12s}{'Recall':>12s}")
        print("-" * 80)
        for diff in unique_diffs:
            diff_mask = difficulties_test == diff
            if diff_mask.sum() == 0:
                continue
            y_true_d = y_test[diff_mask]
            y_pred_d = _ft_pred[diff_mask]
            count = int(diff_mask.sum())
            acc = float((y_true_d == y_pred_d).mean())
            # FP rate: load cases misclassified as fault
            load_mask = y_true_d == 1
            fpr = float((y_pred_d[load_mask] == 0).mean()) if load_mask.sum() > 0 else 0.0
            # Fault recall
            fault_mask = y_true_d == 0
            recall = float((y_pred_d[fault_mask] == 0).mean()) if fault_mask.sum() > 0 else 0.0
            print(f"{diff:<15s}{count:>8d}{acc:>12.3f}{fpr:>12.3f}{recall:>12.3f}")
        print("=" * 80)

    # Save CSV — suffix encodes data source for unique filenames
    csv_dir = "outputs/results"
    os.makedirs(csv_dir, exist_ok=True)
    if args.data == "rcaeval":
        if args.pool == "all":
            suffix = "all_pooled"
        elif args.pool == "systems":
            suffix = f"{args.dataset}_pooled-systems"
        else:
            suffix = f"{args.dataset}_{args.system}"
    else:
        suffix = "synthetic"
    csv_path = os.path.join(csv_dir, f"ablation_results_{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = ["variant"]
        for m in metrics_to_track:
            header_row.extend([m + "_mean", m + "_std"])
        writer.writerow(header_row)
        for v in variants:
            row = [v]
            for m in metrics_to_track:
                row.append(f"{summary[v][m + '_mean']:.4f}")
                row.append(f"{summary[v][m + '_std']:.4f}")
            writer.writerow(row)
    print(f"\nResults saved to {csv_path}")

    # SHAP analysis (optional, on last fold/run's data)
    if args.shap:
        from src.features.feature_schema import ALL_FEATURE_NAMES
        from src.evaluation.visualization import (
            plot_shap_summary, plot_shap_by_class, plot_shap_by_fault_type,
        )

        shap_dir = os.path.join(csv_dir, f"shap_{suffix}")
        os.makedirs(shap_dir, exist_ok=True)
        print("\nGenerating SHAP plots on last fold data...")

        # Feature names for context-free baselines (39 features)
        ctx_s, ctx_e = CONTEXT_RANGE
        no_ctx_feature_names = [n for i, n in enumerate(ALL_FEATURE_NAMES) if i < ctx_s or i >= ctx_e]

        # Baseline RF (no context, 39 features)
        print("  Baseline RF SHAP...")
        rf = BaselineClassifier(random_state=args.base_seed)
        rf.fit(X_train_no_ctx, y_train)
        plot_shap_summary(
            rf, X_test_no_ctx, no_ctx_feature_names,
            save_path=os.path.join(shap_dir, "shap_baseline_rf.png"),
        )
        plot_shap_by_class(
            rf, X_test_no_ctx, y_test, no_ctx_feature_names,
            save_path=os.path.join(shap_dir, "shap_baseline_rf_by_class.png"),
        )
        plot_shap_by_fault_type(
            rf, X_test_no_ctx, list(fault_types_test), no_ctx_feature_names,
            save_path=os.path.join(shap_dir, "shap_baseline_rf_by_fault_type.png"),
        )
        print(f"  Saved to {shap_dir}/shap_baseline_rf*.png")

        # Full CAAA (uses KernelExplainer, slower)
        print("  Full CAAA SHAP (KernelExplainer, may take a moment)...")
        torch.manual_seed(args.base_seed)
        _shap_device = "cuda" if torch.cuda.is_available() else "cpu"
        caaa_model = CAAAModel(input_dim=N_FEATURES, hidden_dim=64, n_classes=2,
                               film_mode=args.film_mode)
        caaa_trainer = CAAATrainer(
            caaa_model, learning_rate=args.lr, device=_shap_device,
            use_context_loss=True,
            unknown_weight=args.unknown_weight,
            loss_variant=args.loss_variant,
            context_dropout_p=args.context_dropout,
        )
        caaa_trainer.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size, early_stopping_patience=10,
        )
        plot_shap_summary(
            caaa_trainer, X_test, ALL_FEATURE_NAMES,
            save_path=os.path.join(shap_dir, "shap_full_caaa.png"),
            X_background=X_train[:50],
        )
        plot_shap_by_fault_type(
            caaa_trainer, X_test, list(fault_types_test), ALL_FEATURE_NAMES,
            save_path=os.path.join(shap_dir, "shap_full_caaa_by_fault_type.png"),
            X_background=X_train[:50],
        )
        print(f"  Saved to {shap_dir}/shap_full_caaa*.png")

    # Calibration evaluation (optional, on last fold/run's data)
    if args.calibration:
        from src.evaluation.metrics import compute_expected_calibration_error
        from src.evaluation.visualization import plot_reliability_diagram

        cal_dir = os.path.join(csv_dir, f"calibration_{suffix}")
        os.makedirs(cal_dir, exist_ok=True)
        print("\nCalibration evaluation on last fold data...")

        # Train a fresh Full CAAA model for calibration analysis
        torch.manual_seed(args.base_seed)
        _cal_device = "cuda" if torch.cuda.is_available() else "cpu"
        cal_model = CAAAModel(input_dim=N_FEATURES, hidden_dim=64, n_classes=2,
                              film_mode=args.film_mode)
        cal_trainer = CAAATrainer(
            cal_model, learning_rate=args.lr, device=_cal_device,
            use_context_loss=True,
            unknown_weight=args.unknown_weight,
            loss_variant=args.loss_variant,
            context_dropout_p=args.context_dropout,
        )
        cal_trainer.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size, early_stopping_patience=10,
        )

        # Before temperature scaling
        proba_uncal = cal_trainer.predict_proba(X_test)
        ece_uncal, _, _, _ = compute_expected_calibration_error(y_test, proba_uncal)
        plot_reliability_diagram(
            y_test, proba_uncal, n_bins=10,
            save_path=os.path.join(cal_dir, "reliability_uncalibrated.png"),
            title="Before Temperature Scaling",
        )
        print(f"  ECE (uncalibrated): {ece_uncal:.4f}")

        # Calibrate temperature on held-out validation set (NOT test set)
        T = cal_trainer.calibrate_temperature(X_val, y_val)
        proba_cal = cal_trainer.predict_proba(X_test)
        ece_cal, _, _, _ = compute_expected_calibration_error(y_test, proba_cal)
        plot_reliability_diagram(
            y_test, proba_cal, n_bins=10,
            save_path=os.path.join(cal_dir, "reliability_calibrated.png"),
            title=f"After Temperature Scaling (T={T:.3f})",
        )
        print(f"  ECE (calibrated):   {ece_cal:.4f}")
        print(f"  Temperature:        {T:.4f}")
        print(f"  Saved to {cal_dir}/")

    # Clean up pre-training checkpoint
    if pretrain_path and os.path.exists(pretrain_path):
        os.remove(pretrain_path)


if __name__ == "__main__":
    main()
