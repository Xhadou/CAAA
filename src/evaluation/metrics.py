"""Evaluation metrics for the CAAA system."""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def compute_false_positive_rate(
    y_true: np.ndarray, y_pred: np.ndarray, positive_class: int = 0
) -> float:
    """Computes false positive rate for the given positive class.

    A false positive occurs when predicting FAULT (0) when the true label
    is EXPECTED_LOAD (1).

    UNKNOWN handling:
        Predictions of class 2 (UNKNOWN) are **not** counted as false
        positives because the model explicitly deferred a decision.
        This means a model that produces many UNKNOWN predictions will
        report a lower FP rate than one that commits to a class for
        every sample.  Use ``unknown_rate`` from
        :func:`compute_all_metrics` alongside this metric to assess
        coverage, or compute ``known_fp_rate`` (FP rate only over
        samples with a definitive prediction) for a coverage-adjusted
        view.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        positive_class: The class considered positive (default: 0 = FAULT).

    Returns:
        False positive rate: FP / (FP + TN).
    """
    negatives = y_true != positive_class
    n_negatives = negatives.sum()
    if n_negatives == 0:
        return 0.0
    false_positives = ((y_pred == positive_class) & negatives).sum()
    return float(false_positives / n_negatives)


def compute_false_positive_reduction(
    baseline_fp_rate: float, model_fp_rate: float
) -> float:
    """Computes false positive reduction percentage.

    This is the KEY metric: FP reduction = (baseline_fp - model_fp) / baseline_fp.

    Args:
        baseline_fp_rate: False positive rate of the baseline model.
        model_fp_rate: False positive rate of the CAAA model.

    Returns:
        False positive reduction as a fraction (0.0 to 1.0).
    """
    if baseline_fp_rate == 0.0:
        return 0.0
    return (baseline_fp_rate - model_fp_rate) / baseline_fp_rate


def compute_fault_recall(
    y_true: np.ndarray, y_pred: np.ndarray, fault_class: int = 0
) -> float:
    """Computes recall for the fault class.

    Recall = TP_fault / (TP_fault + FN_fault). Target: >90%.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        fault_class: The class representing faults (default: 0).

    Returns:
        Fault recall value.
    """
    fault_mask = y_true == fault_class
    n_faults = fault_mask.sum()
    if n_faults == 0:
        return 0.0
    true_positives = ((y_pred == fault_class) & fault_mask).sum()
    return float(true_positives / n_faults)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_fp_rate: Optional[float] = None,
) -> Dict[str, float]:
    """Computes all evaluation metrics.

    Handles 3 possible prediction classes: 0=FAULT, 1=EXPECTED_LOAD,
    2=UNKNOWN. UNKNOWN predictions are excluded from precision/recall/F1
    calculations. The unknown_rate metric tracks the fraction of UNKNOWN
    predictions.

    Args:
        y_true: Ground truth labels (values in {0, 1}).
        y_pred: Predicted labels (values in {0, 1, 2}).
        baseline_fp_rate: Optional baseline false positive rate for
            computing FP reduction.

    Returns:
        Dictionary with:
            - accuracy: Overall accuracy (UNKNOWN counted as incorrect).
            - precision: Weighted precision (on known predictions only).
            - recall: Weighted recall (on known predictions only).
            - f1: Weighted F1 score (on known predictions only).
            - fp_rate: False positive rate for FAULT class.
            - fault_recall: Recall for the fault class.
            - fp_reduction: FP reduction vs baseline (if baseline_fp_rate provided).
            - attribution_accuracy: Same as accuracy.
            - unknown_rate: Fraction of predictions that are UNKNOWN.
    """
    unknown_mask = y_pred == 2
    unknown_rate = float(unknown_mask.sum()) / len(y_pred) if len(y_pred) > 0 else 0.0
    known_mask = ~unknown_mask

    if known_mask.sum() > 0:
        y_true_known = y_true[known_mask]
        y_pred_known = y_pred[known_mask]
        precision = precision_score(
            y_true_known, y_pred_known, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_true_known, y_pred_known, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_true_known, y_pred_known, average="weighted", zero_division=0
        )
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": compute_false_positive_rate(y_true, y_pred),
        "fault_recall": compute_fault_recall(y_true, y_pred),
        "unknown_rate": unknown_rate,
    }

    # FP rate computed only over samples with a definitive prediction
    if known_mask.sum() > 0:
        metrics["known_fp_rate"] = compute_false_positive_rate(
            y_true[known_mask], y_pred[known_mask]
        )
    else:
        metrics["known_fp_rate"] = 0.0

    if baseline_fp_rate is not None:
        metrics["fp_reduction"] = compute_false_positive_reduction(
            baseline_fp_rate, metrics["fp_rate"]
        )

    metrics["attribution_accuracy"] = metrics["accuracy"]

    return metrics


def print_evaluation_summary(metrics: Dict[str, float]) -> None:
    """Prints a formatted evaluation summary.

    Args:
        metrics: Dictionary of metric names to values, as returned
            by compute_all_metrics.
    """
    summary_lines = [
        "=" * 50,
        "Evaluation Summary",
        "=" * 50,
        f"  Accuracy:            {metrics.get('accuracy', 0.0):.4f}",
        f"  Precision (weighted): {metrics.get('precision', 0.0):.4f}",
        f"  Recall (weighted):    {metrics.get('recall', 0.0):.4f}",
        f"  F1 (weighted):        {metrics.get('f1', 0.0):.4f}",
        "-" * 50,
        f"  False Positive Rate:  {metrics.get('fp_rate', 0.0):.4f}",
        f"  Fault Recall:         {metrics.get('fault_recall', 0.0):.4f}",
    ]

    if "fp_reduction" in metrics:
        summary_lines.append(
            f"  FP Reduction:         {metrics['fp_reduction']:.4f}"
        )

    summary_lines.append("=" * 50)

    summary = "\n".join(summary_lines)
    logger.info("\n%s", summary)
    print(summary)


def cross_validate_model(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Run stratified k-fold CV and return per-fold metrics.

    Args:
        model_factory: Callable that returns a fresh (unfitted) model with
            ``fit(X, y)`` and ``predict(X)`` methods.
        X: Full feature matrix of shape (n_samples, n_features).
        y: Full labels of shape (n_samples,).
        n_splits: Number of folds.
        seed: Random seed for fold generation.

    Returns:
        Dict mapping metric names to lists of per-fold values.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics: Dict[str, List[float]] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Compute naive FP rate on this fold's test set
        naive_fp = compute_false_positive_rate(
            y_test, np.zeros(len(y_test), dtype=int)
        )

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)

        for key, value in metrics.items():
            fold_metrics.setdefault(key, []).append(value)

        logger.debug("Fold %d: accuracy=%.3f, f1=%.3f", fold_idx, metrics["accuracy"], metrics["f1"])

    return fold_metrics


def compute_expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by max-class probability and measures the gap between
    mean confidence and accuracy in each bin.

    Reference: Guo et al., "On Calibration of Modern Neural Networks",
    ICML 2017.

    Args:
        y_true: Ground truth labels of shape (n_samples,).
        proba: Predicted probabilities of shape (n_samples, n_classes).
        n_bins: Number of equally-spaced confidence bins.

    Returns:
        Tuple of (ece, bin_accuracies, bin_confidences, bin_counts) where
        *ece* is the scalar ECE value and the arrays have length *n_bins*.
    """
    confidences = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        bin_counts[b] = count
        if count > 0:
            bin_accuracies[b] = correct[mask].mean()
            bin_confidences[b] = confidences[mask].mean()

    # Weighted average of |accuracy - confidence| per bin
    total = len(y_true)
    ece = float(np.sum(bin_counts / total * np.abs(bin_accuracies - bin_confidences)))

    return ece, bin_accuracies, bin_confidences, bin_counts
