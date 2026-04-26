#!/usr/bin/env python3
"""Scaling study: how does CAAA performance change vs tree baselines as
synthetic training data size increases?

Scientific motivation
---------------------
Grinsztajn et al. (NeurIPS 2022) showed that trees dominate neural networks
on small tabular datasets but the gap narrows as data grows. This study
maps the neural-tree crossover point for the CAAA task.

Expected pattern
----------------
At low sample counts (100-500), tree baselines (CatBoost) beat CAAA.
As samples grow (1000+), CAAA's richer architecture begins extracting
more signal than tree splits can. The "crossover point" is where CAAA's
accuracy equals or exceeds CatBoost.

Output
------
- outputs/results/scaling_study.csv: size, variant, f1_mean, f1_std
- outputs/results/scaling_curve.png: F1 vs training size for each variant
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_combined_dataset
from src.features import FeatureExtractor
from src.features.feature_schema import CONTEXT_RANGE, N_FEATURES
from src.models import CAAAModel, CatBoostBaseline, XGBoostBaseline, LightGBMBaseline
from src.models.baseline import BaselineClassifier, NaiveBaseline
from src.training.trainer import CAAATrainer
from src.evaluation.metrics import compute_all_metrics, compute_false_positive_rate
from src.utils import set_seed, NaNSafeScaler


def run_size_point(
    n_per_class: int,
    n_runs: int = 3,
    base_seed: int = 42,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    difficulty: str = "default",
):
    """Run ablation at a single dataset size.

    Args:
        n_per_class: Number of fault and load cases each (total = 2 * n_per_class).
        n_runs: Number of runs for variance estimation.
        base_seed: Base random seed.
        epochs: Training epochs for CAAA.
        batch_size: Training batch size.
        lr: Learning rate.
        difficulty: Task difficulty ("default", "moderate", or "hard").
            "moderate" lowers the Bayes-optimal ceiling by using more
            disguised faults and smaller severity factors; "hard" pushes
            further to expose architectural differences still masked under
            moderate saturation.

    Returns:
        Dictionary of ``{variant: {metric: [values]}}``.
    """
    # Variants include both "no context" (fair traditional baseline) and
    # "with context" (upper bound) versions for every tree model family.
    # This enables a "context contribution by model family" analysis regardless
    # of which thesis framing the final results support.
    variants = [
        "Full CAAA",
        "No Context",
        "CatBoost (no context)",
        "CatBoost (with context)",
        "XGBoost (no context)",
        "XGBoost (with context)",
        "LightGBM (no context)",
        "LightGBM (with context)",
        "RandomForest (no context)",
        "RandomForest (with context)",
    ]
    results = {v: {"accuracy": [], "f1": []} for v in variants}

    for run_idx in range(n_runs):
        run_seed = base_seed + run_idx
        set_seed(run_seed)

        # Generate dataset
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=n_per_class, n_load=n_per_class,
            systems=["online-boutique"], seed=run_seed,
            include_hard=True, difficulty=difficulty,
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0 if c.label == "FAULT" else 1 for c in all_cases])

        extractor = FeatureExtractor(context_mode="external")
        X = extractor.extract_batch(all_cases).astype(np.float32)

        # Train/test split (80/20)
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)), test_size=0.2,
            random_state=run_seed, stratify=labels,
        )
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train_all = labels[train_idx]
        y_test = labels[test_idx]

        # Val split
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_train_raw, y_train_all, test_size=0.125,
            random_state=run_seed, stratify=y_train_all,
        )

        scaler = NaNSafeScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_val = scaler.transform(X_val_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)

        # Context-free arrays for trees
        ctx_s, ctx_e = CONTEXT_RANGE
        X_train_no_ctx = np.delete(X_train_raw, range(ctx_s, ctx_e), axis=1)
        X_test_no_ctx = np.delete(X_test_raw, range(ctx_s, ctx_e), axis=1)

        naive = NaiveBaseline()
        naive_fp = compute_false_positive_rate(
            y_test, naive.predict(X_test_raw),
        )

        # Full CAAA
        torch.manual_seed(run_seed)
        model = CAAAModel(input_dim=N_FEATURES, hidden_dim=64, n_classes=2,
                          film_mode="tadam")
        trainer = CAAATrainer(
            model, learning_rate=lr, device="cpu",
            use_context_loss=True, loss_variant="gated",
            context_dropout_p=0.3,
        )
        trainer.train(
            X_train, y_train, X_val=X_val, y_val=y_val,
            epochs=epochs, batch_size=batch_size,
            early_stopping_patience=10,
        )
        y_pred = trainer.predict(X_test)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["Full CAAA"]["accuracy"].append(m["accuracy"])
        results["Full CAAA"]["f1"].append(m["f1"])

        # No Context CAAA (ablation)
        X_train_nc = X_train.copy()
        X_test_nc = X_test.copy()
        X_val_nc = X_val.copy()
        X_train_nc[:, ctx_s:ctx_e] = 0.0
        X_test_nc[:, ctx_s:ctx_e] = 0.0
        X_val_nc[:, ctx_s:ctx_e] = 0.0
        torch.manual_seed(run_seed)
        model_nc = CAAAModel(input_dim=N_FEATURES, hidden_dim=64, n_classes=2,
                             film_mode="tadam")
        trainer_nc = CAAATrainer(
            model_nc, learning_rate=lr, device="cpu",
            use_context_loss=True, loss_variant="gated",
            context_dropout_p=0.3,
        )
        trainer_nc.train(
            X_train_nc, y_train, X_val=X_val_nc, y_val=y_val,
            epochs=epochs, batch_size=batch_size,
            early_stopping_patience=10,
        )
        y_pred = trainer_nc.predict(X_test_nc)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["No Context"]["accuracy"].append(m["accuracy"])
        results["No Context"]["f1"].append(m["f1"])

        # CatBoost (no context)
        cb_no = CatBoostBaseline(random_state=run_seed)
        cb_no.fit(X_train_no_ctx, y_train)
        y_pred = cb_no.predict(X_test_no_ctx)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["CatBoost (no context)"]["accuracy"].append(m["accuracy"])
        results["CatBoost (no context)"]["f1"].append(m["f1"])

        # CatBoost (with context)
        cb_ctx = CatBoostBaseline(random_state=run_seed)
        cb_ctx.fit(X_train_raw, y_train)
        y_pred = cb_ctx.predict(X_test_raw)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["CatBoost (with context)"]["accuracy"].append(m["accuracy"])
        results["CatBoost (with context)"]["f1"].append(m["f1"])

        # XGBoost (no context)
        xgb = XGBoostBaseline(random_state=run_seed)
        xgb.fit(X_train_no_ctx, y_train)
        y_pred = xgb.predict(X_test_no_ctx)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["XGBoost (no context)"]["accuracy"].append(m["accuracy"])
        results["XGBoost (no context)"]["f1"].append(m["f1"])

        # XGBoost (with context) — upper-bound reference for Path B story
        xgb_ctx = XGBoostBaseline(random_state=run_seed)
        xgb_ctx.fit(X_train_raw, y_train)
        y_pred = xgb_ctx.predict(X_test_raw)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["XGBoost (with context)"]["accuracy"].append(m["accuracy"])
        results["XGBoost (with context)"]["f1"].append(m["f1"])

        # LightGBM (no context)
        lgb = LightGBMBaseline(random_state=run_seed)
        lgb.fit(X_train_no_ctx, y_train)
        y_pred = lgb.predict(X_test_no_ctx)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["LightGBM (no context)"]["accuracy"].append(m["accuracy"])
        results["LightGBM (no context)"]["f1"].append(m["f1"])

        # LightGBM (with context)
        lgb_ctx = LightGBMBaseline(random_state=run_seed)
        lgb_ctx.fit(X_train_raw, y_train)
        y_pred = lgb_ctx.predict(X_test_raw)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["LightGBM (with context)"]["accuracy"].append(m["accuracy"])
        results["LightGBM (with context)"]["f1"].append(m["f1"])

        # RandomForest (no context)
        rf = BaselineClassifier(random_state=run_seed)
        rf.fit(X_train_no_ctx, y_train)
        y_pred = rf.predict(X_test_no_ctx)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["RandomForest (no context)"]["accuracy"].append(m["accuracy"])
        results["RandomForest (no context)"]["f1"].append(m["f1"])

        # RandomForest (with context)
        rf_ctx = BaselineClassifier(random_state=run_seed)
        rf_ctx.fit(X_train_raw, y_train)
        y_pred = rf_ctx.predict(X_test_raw)
        m = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)
        results["RandomForest (with context)"]["accuracy"].append(m["accuracy"])
        results["RandomForest (with context)"]["f1"].append(m["f1"])

    return results


def main():
    parser = argparse.ArgumentParser(description="CAAA Scaling Study")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[100, 250, 500, 1000, 2500],
                        help="List of per-class sample counts")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Runs per size point")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--difficulty", choices=["default", "moderate", "hard"],
                        default="default",
                        help="Task difficulty. 'moderate' lowers the ceiling to "
                             "~95%; 'hard' to ~88-92% to expose "
                             "architectural differences that moderate 40K "
                             "saturation still masks.")
    parser.add_argument("--log-per-seed", action="store_true",
                        help="In addition to the aggregate CSV, dump per-seed "
                             "F1 arrays to JSON to enable Wilcoxon signed-rank "
                             "and Cliff's delta statistics.")
    args = parser.parse_args()

    # Print config
    print("=" * 70)
    print("  CAAA SCALING STUDY — Neural vs Tree Crossover Point")
    print("=" * 70)
    print(f"  Difficulty:        {args.difficulty}")
    print(f"  Sizes (per class): {args.sizes}")
    print(f"  Total samples:     {[2*s for s in args.sizes]}")
    print(f"  Runs per size:     {args.n_runs}")
    print(f"  Epochs:            {args.epochs}")
    print("=" * 70)

    # Run scaling study
    all_results = {}
    for size in args.sizes:
        total = 2 * size
        print(f"\n{'='*70}")
        print(f"  Size: {size}+{size} = {total} total samples ({args.difficulty})")
        print(f"{'='*70}")
        results = run_size_point(
            n_per_class=size, n_runs=args.n_runs,
            base_seed=args.base_seed, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            difficulty=args.difficulty,
        )
        all_results[total] = results

        # Print summary for this size
        print(f"\n  Results @ {total} samples:")
        for variant, metrics in results.items():
            f1s = metrics["f1"]
            print(f"    {variant:<30s} F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # Save CSV — merge with existing results if present (keep new for duplicate sizes).
    # Separate files for each difficulty so results don't overwrite.
    csv_dir = "outputs/results"
    os.makedirs(csv_dir, exist_ok=True)
    _DIFF_SUFFIX = {"default": "", "moderate": "_moderate", "hard": "_hard"}[args.difficulty]
    csv_name = f"scaling_study{_DIFF_SUFFIX}.csv"
    csv_path = os.path.join(csv_dir, csv_name)

    existing_rows = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows[(int(row["total_samples"]), row["variant"])] = row

    # Preserve raw per-seed values for significance testing (gets overwritten
    # by the 2-point reconstruction below which loses the sample count).
    raw_results = {
        total: {variant: list(metrics["f1"])
                for variant, metrics in results.items()}
        for total, results in all_results.items()
    }

    if args.log_per_seed:
        import json
        json_name = f"scaling_per_seed{_DIFF_SUFFIX}.json"
        json_path = os.path.join(csv_dir, json_name)
        existing_per_seed = {}
        if os.path.exists(json_path):
            with open(json_path) as jf:
                existing_per_seed = json.load(jf)
        for total, per_variant in raw_results.items():
            existing_per_seed.setdefault(str(total), {})
            for variant, f1_list in per_variant.items():
                existing_per_seed[str(total)][variant] = f1_list
        with open(json_path, "w") as jf:
            json.dump(existing_per_seed, jf, indent=2)
        print(f"  Per-seed F1 arrays saved to {json_path}")

    # Overlay new results on top of existing (new results win for duplicate keys)
    for total, results in all_results.items():
        for variant, metrics in results.items():
            existing_rows[(total, variant)] = {
                "total_samples": total,
                "variant": variant,
                "accuracy_mean": f"{np.mean(metrics['accuracy']):.4f}",
                "accuracy_std": f"{np.std(metrics['accuracy']):.4f}",
                "f1_mean": f"{np.mean(metrics['f1']):.4f}",
                "f1_std": f"{np.std(metrics['f1']):.4f}",
            }

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["total_samples", "variant", "accuracy_mean",
                         "accuracy_std", "f1_mean", "f1_std"])
        for key in sorted(existing_rows.keys()):
            r = existing_rows[key]
            writer.writerow([r["total_samples"], r["variant"],
                             r["accuracy_mean"], r["accuracy_std"],
                             r["f1_mean"], r["f1_std"]])

    # Also reconstruct all_results from combined rows for the plot, preserving
    # the original std (reconstruct a synthetic 2-point list that has the
    # correct mean and std for np.mean/np.std to recover them).
    combined = {}
    for (total, variant), row in existing_rows.items():
        mean = float(row["f1_mean"])
        std = float(row["f1_std"])
        # Two-point list with correct mean and std (n=2): [mean-std, mean+std]
        combined.setdefault(total, {}).setdefault(variant, {"f1": [], "accuracy": []})
        combined[total][variant]["f1"] = [mean - std, mean + std]
        combined[total][variant]["accuracy"] = [
            float(row["accuracy_mean"]) - float(row["accuracy_std"]),
            float(row["accuracy_mean"]) + float(row["accuracy_std"]),
        ]
    all_results = combined
    print(f"\n  Results saved to {csv_path}")

    # Generate scaling curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 7))
        sizes = sorted(all_results.keys())
        # Only show the headline variants on the scaling curve — the full
        # per-family breakdown lives in the context-contribution bar chart.
        variants_to_plot = [
            ("Full CAAA", "o-", "#2E86AB", 3),
            ("No Context", "o--", "#A5A5A5", 2),
            ("CatBoost (no context)", "s-", "#E63946", 2),
            ("CatBoost (with context)", "s--", "#F4A261", 3),
            ("XGBoost (no context)", "^-", "#8D5A97", 2),
            ("LightGBM (no context)", "v-", "#8DB580", 2),
        ]

        for variant, style, color, lw in variants_to_plot:
            if variant not in all_results[sizes[0]]:
                continue
            means = [np.mean(all_results[s][variant]["f1"]) for s in sizes]
            stds = [np.std(all_results[s][variant]["f1"]) for s in sizes]
            ax.errorbar(sizes, means, yerr=stds, fmt=style,
                        label=variant, color=color, capsize=4, linewidth=lw,
                        markersize=9, alpha=0.95)

        ax.set_xlabel("Total training samples", fontsize=13)
        ax.set_ylabel("F1 Score", fontsize=13)
        title_suffix = {"default": "Default Task", "moderate": "Moderate Task", "hard": "Hard Task"}[args.difficulty]
        ax.set_title(
            f"Scaling Study ({title_suffix}): CAAA vs Trees",
            fontsize=14, fontweight="bold",
        )
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=11)
        # Wider y-range for non-default modes since ceilings are lower.
        ax.set_ylim([0.5, 1.0] if args.difficulty in ("moderate", "hard") else [0.86, 1.0])

        # Annotate crossover point if CAAA exceeds CatBoost (with context)
        for s in sizes:
            caaa = np.mean(all_results[s]["Full CAAA"]["f1"])
            cb_ctx = np.mean(all_results[s]["CatBoost (with context)"]["f1"])
            if caaa > cb_ctx and s >= 1000:
                ax.annotate("CAAA > CatBoost (+ctx)",
                            xy=(s, caaa), xytext=(s*0.3, caaa+0.03),
                            fontsize=11, fontweight="bold", color="darkgreen",
                            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2))
                break

        plt.tight_layout()
        plot_name = f"scaling_curve{_DIFF_SUFFIX}.png"
        plot_path = os.path.join(csv_dir, plot_name)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {plot_path}")

        # Context contribution bar chart (Path B hero plot)
        # Uses the largest size to get tightest estimates.
        largest = max(sizes)
        families = [
            ("CAAA", "Full CAAA", "No Context"),
            ("CatBoost", "CatBoost (with context)", "CatBoost (no context)"),
            ("XGBoost", "XGBoost (with context)", "XGBoost (no context)"),
            ("LightGBM", "LightGBM (with context)", "LightGBM (no context)"),
            ("RandomForest", "RandomForest (with context)", "RandomForest (no context)"),
        ]
        deltas = []
        family_labels = []
        for label, with_ctx, no_ctx in families:
            if with_ctx not in all_results[largest] or no_ctx not in all_results[largest]:
                continue
            with_f1 = np.mean(all_results[largest][with_ctx]["f1"])
            no_f1 = np.mean(all_results[largest][no_ctx]["f1"])
            delta_pp = (with_f1 - no_f1) * 100
            deltas.append(delta_pp)
            family_labels.append(label)

        if deltas:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#2E86AB", "#F4A261", "#8D5A97", "#8DB580", "#E63946"][:len(deltas)]
            bars = ax.bar(family_labels, deltas, color=colors, edgecolor="black", linewidth=1.5)
            ax.set_ylabel("F1 gain from adding context (pp)", fontsize=12)
            ax.set_title(
                f"Context Contribution by Model Family ({title_suffix}, n={largest:,})",
                fontsize=13, fontweight="bold",
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(True, alpha=0.3, axis="y")
            # Annotate bar heights
            for bar, delta in zip(bars, deltas):
                height = bar.get_height()
                y_offset = 0.05 if height >= 0 else -0.15
                ax.text(bar.get_x() + bar.get_width() / 2, height + y_offset,
                        f"+{delta:.2f}pp" if delta >= 0 else f"{delta:.2f}pp",
                        ha="center", fontsize=11, fontweight="bold")
            plt.tight_layout()
            ctx_name = f"context_contribution{_DIFF_SUFFIX}.png"
            ctx_plot_path = os.path.join(csv_dir, ctx_name)
            plt.savefig(ctx_plot_path, dpi=150, bbox_inches="tight")
            print(f"  Context contribution plot saved to {ctx_plot_path}")
    except ImportError:
        print("  matplotlib not available, skipping plot")

    # Statistical significance test: CAAA vs CatBoost+ctx at largest size.
    # Uses raw_results (not reconstructed 2-point lists from all_results).
    try:
        from scipy import stats
        largest_total = max(raw_results.keys())
        caaa_f1s = raw_results[largest_total].get("Full CAAA", [])
        cb_f1s = raw_results[largest_total].get("CatBoost (with context)", [])
        if len(caaa_f1s) >= 2 and len(cb_f1s) >= 2:
            t, p = stats.ttest_ind(caaa_f1s, cb_f1s, equal_var=False)
            delta = (np.mean(caaa_f1s) - np.mean(cb_f1s)) * 100
            sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"
            print()
            print("=" * 70)
            print(f"  STATISTICAL TEST @ n={largest_total} ({args.difficulty} task)")
            print("=" * 70)
            print(f"  Full CAAA F1:            {np.mean(caaa_f1s):.4f} (n={len(caaa_f1s)})")
            print(f"  CatBoost+ctx F1:         {np.mean(cb_f1s):.4f} (n={len(cb_f1s)})")
            print(f"  Delta (CAAA - CatBoost): {delta:+.3f}pp")
            print(f"  Welch's t-test:          t={t:.3f}, p={p:.4f}")
            print(f"  Result: {sig} (alpha=0.05)")
            print("=" * 70)
    except ImportError:
        print("  scipy not available, skipping significance test")


if __name__ == "__main__":
    main()
