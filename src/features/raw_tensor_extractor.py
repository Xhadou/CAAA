"""Extract raw time-series tensors from AnomalyCase for temporal encoding.

Produces a fixed-size (max_services, seq_len, 7) tensor from variable-length
per-service metric DataFrames, along with a boolean mask indicating which
service slots contain real data (vs zero-padding).
"""

from typing import List, Tuple

import numpy as np

from src.data_loader.data_types import AnomalyCase

METRIC_COLS = [
    "cpu_usage",
    "memory_usage",
    "request_rate",
    "error_rate",
    "latency",
    "network_in",
    "network_out",
]


class RawTensorExtractor:
    """Extract raw metric tensors from AnomalyCase objects.

    Handles variable service counts (padding/truncation to ``max_services``)
    and variable timestep counts (linear interpolation to ``seq_len``).
    Applies per-case z-score normalization so metrics are on comparable scales.

    Args:
        max_services: Fixed number of service slots. Cases with more services
            keep only the ``max_services`` with highest metric variance.
        seq_len: Fixed number of timesteps after resampling.
    """

    def __init__(self, max_services: int = 20, seq_len: int = 120) -> None:
        self.max_services = max_services
        self.seq_len = seq_len

    def extract(self, case: AnomalyCase) -> Tuple[np.ndarray, np.ndarray]:
        """Extract tensor and mask from a single case.

        Returns:
            tensor: Shape ``(max_services, seq_len, 7)`` float32.
            mask:   Shape ``(max_services,)`` float32, 1.0 for real services.
        """
        tensor = np.zeros(
            (self.max_services, self.seq_len, len(METRIC_COLS)),
            dtype=np.float32,
        )
        mask = np.zeros(self.max_services, dtype=np.float32)

        services = case.services
        if not services:
            return tensor, mask

        # If more services than slots, keep the ones with highest variance
        if len(services) > self.max_services:
            variances = []
            for svc in services:
                df = svc.metrics
                cols = [c for c in METRIC_COLS if c in df.columns]
                v = df[cols].var().sum() if cols else 0.0
                variances.append(v)
            top_idx = np.argsort(variances)[-self.max_services :]
            services = [services[i] for i in sorted(top_idx)]

        for i, svc in enumerate(services):
            if i >= self.max_services:
                break
            df = svc.metrics
            vals = np.zeros((len(df), len(METRIC_COLS)), dtype=np.float32)
            for j, col in enumerate(METRIC_COLS):
                if col in df.columns:
                    raw = df[col].values.astype(np.float32)
                    np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    vals[:, j] = raw

            # Resample to seq_len via linear interpolation
            if len(vals) != self.seq_len and len(vals) > 1:
                x_old = np.linspace(0, 1, len(vals))
                x_new = np.linspace(0, 1, self.seq_len)
                resampled = np.zeros(
                    (self.seq_len, len(METRIC_COLS)), dtype=np.float32,
                )
                for j in range(len(METRIC_COLS)):
                    resampled[:, j] = np.interp(x_new, x_old, vals[:, j])
                vals = resampled
            elif len(vals) == 1:
                vals = np.tile(vals, (self.seq_len, 1))

            tensor[i, : len(vals)] = vals[: self.seq_len]
            mask[i] = 1.0

        # Per-case z-score normalization per metric column
        for j in range(len(METRIC_COLS)):
            col_data = tensor[:, :, j]  # (max_services, seq_len)
            real_mask = mask[:, None]  # (max_services, 1)
            masked = col_data * real_mask
            n_real = real_mask.sum() * self.seq_len
            if n_real > 0:
                mean = masked.sum() / n_real
                var = ((masked - mean * real_mask) ** 2).sum() / n_real
                std = np.sqrt(var + 1e-8)
                tensor[:, :, j] = (col_data - mean) / std * real_mask

        return tensor, mask

    def extract_batch(
        self, cases: List[AnomalyCase],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract tensors and masks for a batch of cases.

        Returns:
            tensors: Shape ``(n_cases, max_services, seq_len, 7)`` float32.
            masks:   Shape ``(n_cases, max_services)`` float32.
        """
        n = len(cases)
        tensors = np.zeros(
            (n, self.max_services, self.seq_len, len(METRIC_COLS)),
            dtype=np.float32,
        )
        masks = np.zeros((n, self.max_services), dtype=np.float32)
        for i, case in enumerate(cases):
            tensors[i], masks[i] = self.extract(case)
        return tensors, masks
