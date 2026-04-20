"""Shared utility functions for the CAAA pipeline."""

import random
from typing import Dict, List

import numpy as np
import torch

# ── Label constants ───────────────────────────────────────────────────

LABEL_FAULT = "FAULT"
LABEL_EXPECTED_LOAD = "EXPECTED_LOAD"
LABEL_UNKNOWN = "UNKNOWN"

LABEL_TO_INT: Dict[str, int] = {
    LABEL_FAULT: 0,
    LABEL_EXPECTED_LOAD: 1,
    LABEL_UNKNOWN: 2,
}

INT_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}


def resolve_device(device: str = "auto") -> str:
    """Resolve device string to a concrete device.

    Args:
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        ``"cuda"`` if available and requested, else ``"cpu"``.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_to_int(labels, label_map=None) -> List[int]:
    """Convert string labels to integer encoding.

    Args:
        labels: Iterable of string labels ("FAULT", "EXPECTED_LOAD").
        label_map: Optional custom mapping. Defaults to LABEL_TO_INT.

    Returns:
        List of integer labels.
    """
    if label_map is None:
        label_map = LABEL_TO_INT
    return [label_map[label] for label in labels]


def int_to_labels(ints, label_map=None) -> List[str]:
    """Convert integer labels back to strings.

    Args:
        ints: Iterable of integer labels (0, 1, 2).
        label_map: Optional custom mapping. Defaults to INT_TO_LABEL.

    Returns:
        List of string labels.
    """
    if label_map is None:
        label_map = INT_TO_LABEL
    return [label_map[int_label] for int_label in ints]


class NaNSafeScaler:
    """StandardScaler wrapper that replaces NaN/inf from zero-variance columns.

    When a feature column has zero variance (e.g. all-zero metrics from services
    with no traffic), StandardScaler divides by zero producing NaN.  This wrapper
    detects those columns and replaces the resulting NaN/inf values with 0.0.
    """

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()

    def fit(self, X):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self._scaler.fit(X)
        return self

    def transform(self, X):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = self._scaler.transform(X)
        np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
