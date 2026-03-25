"""Feature extraction for anomaly cases.

Extracts a fixed-size feature vector (44 features) from an AnomalyCase,
organized into workload, behavioral, context, statistical, service-level,
and extended feature groups.

The ``onset_gradient`` feature uses PELT change point detection (Killick 2012)
via the ``ruptures`` library, inspired by BARO (FSE 2024) which showed that
Bayesian change point detection before RCA improves results by 58-189%.
"""

from datetime import datetime as _datetime, timezone as _timezone
import functools
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:
    import ruptures as rpt
except ImportError:
    raise ImportError(
        "ruptures is required for change-point detection features. "
        "Install with: pip install ruptures"
    )

from src.data_loader.data_types import AnomalyCase, ServiceMetrics
from src.features.feature_schema import (
    WORKLOAD_NAMES as _WORKLOAD_NAMES,
    BEHAVIORAL_NAMES as _BEHAVIORAL_NAMES,
    CONTEXT_NAMES as _CONTEXT_NAMES,
    STAT_METRIC_COLS as _STAT_METRIC_COLS,
    STATISTICAL_NAMES as _STATISTICAL_NAMES,
    SERVICE_LEVEL_NAMES as _SERVICE_LEVEL_NAMES,
    EXTENDED_NAMES as _EXTENDED_NAMES,
    N_FEATURES,
)

logger = logging.getLogger(__name__)


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning 0.0 for constant arrays."""
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    try:
        r, _ = pearsonr(x, y)
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


def _linear_slope(arr: np.ndarray) -> float:
    """Compute slope of a linear fit to *arr*."""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    with np.errstate(invalid="ignore"):
        coeffs = np.polyfit(x, arr, 1)
    return float(coeffs[0]) if np.isfinite(coeffs[0]) else 0.0


@functools.lru_cache(maxsize=4096)
def _detect_change_point_cached(
    series_tuple: Tuple[float, ...], penalty: float = 10,
) -> Tuple[int, float, float]:
    """Wrapper around :func:`_detect_change_point`."""
    return _detect_change_point(np.array(series_tuple), penalty)


def _detect_change_point(
    series: np.ndarray, penalty: float = 10,
) -> Tuple[int, float, float]:
    """Detect the most significant change point in a time series.

    Uses the PELT algorithm (Killick 2012) for O(n) change point detection
    with an RBF cost model, inspired by BARO (FSE 2024) which showed that
    change point detection before RCA improves results significantly.

    The RBF (Radial Basis Function) kernel is chosen because it can detect
    changes in both mean and variance simultaneously — important for fault
    signals that may shift the distribution shape, not just the mean.

    Args:
        series: 1-D array of metric values.
        penalty: Penalty value for PELT.  Higher values produce fewer change
            points.

    Returns:
        Tuple of ``(change_idx, magnitude, abruptness)`` where:
            - *change_idx*: index of the most significant change point, or
              ``-1`` if none found.
            - *magnitude*: ``|mean_after - mean_before| / std`` — measures
              the size of the regime change.
            - *abruptness*: ``|gradient[cp]| / std`` — measures how sudden
              the change is.
    """
    if len(series) < 10:
        return -1, 0.0, 0.0

    algo = rpt.Pelt(model="rbf").fit(series.reshape(-1, 1))
    change_points = algo.predict(pen=penalty)
    # ruptures includes len(series) as the last element
    change_points = [cp for cp in change_points if cp < len(series)]

    if not change_points:
        return -1, 0.0, 0.0

    # Find the most significant change point.  We skip change points within
    # 3 indices of the edges to ensure stable mean/gradient estimates on
    # both sides of the split.
    best_cp, best_magnitude = -1, 0.0
    for cp in change_points:
        if cp < 3 or cp > len(series) - 3:
            continue
        mean_before = np.mean(series[:cp])
        mean_after = np.mean(series[cp:])
        std = np.std(series) + 1e-10
        magnitude = abs(mean_after - mean_before) / std
        if magnitude > best_magnitude:
            best_cp, best_magnitude = cp, magnitude

    if best_cp == -1:
        return -1, 0.0, 0.0

    # Compute abruptness (gradient at change point)
    grad = np.gradient(series)
    abruptness = abs(grad[best_cp]) / (np.std(series) + 1e-10)

    return best_cp, best_magnitude, abruptness


class FeatureExtractor:
    """Extracts a 44-dimensional feature vector from an ``AnomalyCase``.

    Feature groups:
        - Workload features (6)
        - Behavioral features (6)
        - Context features (5)
        - Statistical features (13)
        - Service-level features (6)
        - Extended features (8)

    Args:
        seed: Random seed for reproducible noise injection in context
            features.  Using a fixed seed ensures that extracting the
            same case twice produces identical features.
    """

    def __init__(self, seed: int = 42) -> None:
        self._names: List[str] = (
            _WORKLOAD_NAMES
            + _BEHAVIORAL_NAMES
            + _CONTEXT_NAMES
            + _STATISTICAL_NAMES
            + _SERVICE_LEVEL_NAMES
            + _EXTENDED_NAMES
        )
        if len(self._names) != N_FEATURES:
            raise RuntimeError(
                f"Feature name count {len(self._names)} != N_FEATURES {N_FEATURES}"
            )
        self._seed = seed
        self._case_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, case: AnomalyCase) -> np.ndarray:
        """Extract a feature vector from a single anomaly case.

        Args:
            case: An ``AnomalyCase`` instance.

        Returns:
            A 1-D numpy array of shape ``(44,)``.
        """
        # Derive a per-case RNG so that extraction order does not affect
        # results.  The seed is based on case_id when available, otherwise
        # on a monotonic counter (preserving batch-determinism).
        if case.case_id is not None:
            case_seed = (self._seed + hash(case.case_id)) % (2**32)
        else:
            case_seed = self._seed + self._case_counter
        self._case_counter += 1
        case_rng = np.random.default_rng(case_seed)

        feats = np.concatenate([
            self._workload_features(case.services),
            self._behavioral_features(case.services),
            self._context_features(case.context, case.services, rng=case_rng),
            self._statistical_features(case.services),
            self._service_level_features(case.services),
            self._extended_features(case),
        ])
        if feats.shape != (N_FEATURES,):
            raise RuntimeError(f"Expected {N_FEATURES} features, got {feats.shape}")
        return feats

    def extract_batch(self, cases: List[AnomalyCase]) -> np.ndarray:
        """Extract features for multiple cases.

        Args:
            cases: List of ``AnomalyCase`` instances.

        Returns:
            A 2-D numpy array of shape ``(len(cases), 44)``.
        """
        out = np.empty((len(cases), N_FEATURES), dtype=np.float64)
        for i, c in enumerate(cases):
            out[i] = self.extract(c)
        return out

    def feature_names(self) -> List[str]:
        """Return ordered list of 44 feature names."""
        return list(self._names)

    # ------------------------------------------------------------------
    # Workload features (6)
    # ------------------------------------------------------------------

    def _workload_features(self, services: List[ServiceMetrics]) -> np.ndarray:
        n = len(services)
        if n == 0:
            return np.zeros(6)

        # Pre-extract arrays once per service to avoid repeated .values calls
        cpu_arrays = [svc.metrics["cpu_usage"].values for svc in services]
        req_arrays = [svc.metrics["request_rate"].values for svc in services]
        err_arrays = [svc.metrics["error_rate"].values for svc in services]
        lat_arrays = [svc.metrics["latency"].values for svc in services]

        # 1. global_load_ratio
        increased = 0
        for cpu in cpu_arrays:
            mid = len(cpu) // 2
            if mid == 0:
                continue
            if np.mean(cpu[mid:]) > np.mean(cpu[:mid]) * 1.10:
                increased += 1
        global_load_ratio = increased / n

        # 2. cpu_request_correlation
        corrs = []
        for cpu, req in zip(cpu_arrays, req_arrays):
            corrs.append(_safe_pearsonr(cpu, req))
        cpu_request_correlation = float(np.mean(corrs))

        # 3. cross_service_sync — mean pairwise CPU correlation
        if n < 2:
            cross_service_sync = 0.0
        else:
            # Pad/truncate to uniform length for vectorized correlation
            min_len = min(len(s) for s in cpu_arrays)
            if min_len < 2:
                cross_service_sync = 0.0
            else:
                cpu_matrix = np.array([s[:min_len] for s in cpu_arrays])
                # Filter out constant series (zero std) which produce NaN
                # correlations — treat them as uncorrelated (0.0).
                stds = np.std(cpu_matrix, axis=1)
                non_const = stds > 0
                if non_const.sum() < 2:
                    cross_service_sync = 0.0
                else:
                    cpu_filtered = cpu_matrix[non_const]
                    corr_matrix = np.corrcoef(cpu_filtered)
                    n_filt = len(cpu_filtered)
                    upper = corr_matrix[np.triu_indices(n_filt, k=1)]
                    cross_service_sync = float(np.mean(upper)) if len(upper) > 0 else 0.0

        # 4. error_rate_delta
        deltas = []
        for err in err_arrays:
            mid = len(err) // 2
            if mid == 0:
                continue
            deltas.append(float(np.mean(err[mid:]) - np.mean(err[:mid])))
        error_rate_delta = float(np.mean(deltas)) if deltas else 0.0

        # 5. latency_cpu_correlation
        lat_corrs = []
        for lat, cpu in zip(lat_arrays, cpu_arrays):
            lat_corrs.append(_safe_pearsonr(lat, cpu))
        latency_cpu_correlation = float(np.mean(lat_corrs))

        # 6. change_point_magnitude (replaces memory_trend_uniformity)
        magnitudes = []
        for cpu in cpu_arrays:
            _, mag, _ = _detect_change_point_cached(tuple(cpu))
            magnitudes.append(mag)
        change_point_magnitude = float(np.mean(magnitudes))

        return np.array([
            global_load_ratio,
            cpu_request_correlation,
            cross_service_sync,
            error_rate_delta,
            latency_cpu_correlation,
            change_point_magnitude,
        ])

    # ------------------------------------------------------------------
    # Behavioral features (6)
    # ------------------------------------------------------------------

    def _behavioral_features(self, services: List[ServiceMetrics]) -> np.ndarray:
        n = len(services)
        if n == 0:
            return np.zeros(6)

        # Pre-extract CPU arrays once per service
        cpu_arrays = [svc.metrics["cpu_usage"].values for svc in services]

        # Pre-compute per-array stats once (reused across features 8-12)
        cpu_means = [np.mean(cpu) for cpu in cpu_arrays]
        cpu_stds = [np.std(cpu) for cpu in cpu_arrays]
        cpu_zscores = [
            (cpu - mu) / std if std > 0 else np.zeros_like(cpu)
            for cpu, mu, std in zip(cpu_arrays, cpu_means, cpu_stds)
        ]

        # 7. onset_gradient (change-point-based abruptness via PELT)
        abruptness_values = []
        for cpu in cpu_arrays:
            _, _, abruptness = _detect_change_point_cached(tuple(cpu))
            abruptness_values.append(abruptness)
        onset_gradient = float(np.mean(abruptness_values))

        # 8. peak_duration
        durations = []
        for i, cpu in enumerate(cpu_arrays):
            seq_len = len(cpu)
            if seq_len == 0 or cpu_stds[i] == 0.0:
                durations.append(0.0)
                continue
            anom_count = int(np.sum(cpu_zscores[i] > 2.0))
            durations.append(anom_count / seq_len)
        peak_duration = float(np.mean(durations))

        # 9. cascade_score
        onset_times: List[float] = []
        for i, cpu in enumerate(cpu_arrays):
            if cpu_stds[i] == 0.0 or len(cpu) < 2:
                continue
            anom_idx = np.where(cpu_zscores[i] > 2.0)[0]
            if len(anom_idx) > 0:
                onset_times.append(float(anom_idx[0]))
        if len(onset_times) < 2:
            cascade_score = 0.0
        else:
            seq_len = max(len(c) for c in cpu_arrays)
            cascade_score = float(np.std(onset_times) / (seq_len + 1e-10))

        # 10. recovery_indicator
        recoveries = []
        for i, cpu in enumerate(cpu_arrays):
            if len(cpu) < 4:
                recoveries.append(0.0)
                continue
            peak = np.max(cpu)
            baseline = cpu_means[i]
            if peak <= baseline:
                recoveries.append(0.0)
                continue
            tail = np.mean(cpu[-max(1, len(cpu) // 5):])
            recoveries.append(float((peak - tail) / (peak - baseline + 1e-10)))
        recovery_indicator = float(np.clip(np.mean(recoveries), 0.0, 1.0))

        # 11. affected_service_ratio
        affected = 0
        for i in range(n):
            if cpu_stds[i] == 0.0:
                continue
            z = cpu_zscores[i]
            if np.any(z > 2.0):
                affected += 1
        affected_service_ratio = affected / n

        # 12. variance_change_ratio
        var_ratios = []
        for cpu in cpu_arrays:
            mid = len(cpu) // 2
            if mid == 0:
                var_ratios.append(0.0)
                continue
            var1 = np.var(cpu[:mid]) + 1e-10
            var2 = np.var(cpu[mid:]) + 1e-10
            var_ratios.append(float(np.log(var2 / var1)))
        variance_change_ratio = float(np.mean(var_ratios)) if var_ratios else 0.0

        return np.array([
            onset_gradient,
            peak_duration,
            cascade_score,
            recovery_indicator,
            affected_service_ratio,
            variance_change_ratio,
        ])

    # ------------------------------------------------------------------
    # Context features (5)
    #
    # NOTE: Context features are intentionally noisy to prevent label
    # leakage.  Without noise, ``event_active`` would be a perfect proxy
    # for the label (1.0 for all EXPECTED_LOAD, 0.0 for all FAULT),
    # allowing the model to achieve near-perfect accuracy without
    # learning from metric patterns.  Gaussian noise on
    # ``context_confidence`` and ``event_expected_impact``, plus a
    # label-independent ``recent_deployment`` base rate, force the model
    # to rely on workload / behavioral / statistical features.
    # ------------------------------------------------------------------

    def _context_features(
        self,
        context: Optional[Dict],
        services: Optional[List[ServiceMetrics]] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Extract context features from an anomaly case.

        Context features are intentionally noisy to prevent label leakage
        and force the model to learn from metric patterns rather than
        relying on a deterministic context signal.

        Noise applied:
            - ``context_confidence``: Gaussian noise (std=0.1)
            - ``event_expected_impact``: Gaussian noise (std=0.05)
            - ``recent_deployment``: sampled from a base rate of 0.15
              for all cases, independent of the label

        Args:
            rng: Optional seeded Generator for deterministic noise.
                 Falls back to a new default_rng if not provided.
        """
        ctx = context or {}
        if rng is None:
            rng = np.random.default_rng(self._seed)

        # 13. event_active
        event_active = 1.0 if "event_type" in ctx else 0.0

        # 14. event_expected_impact (with Gaussian noise, std=0.05)
        if "load_multiplier" in ctx:
            event_expected_impact = min(float(ctx["load_multiplier"]) / 5.0, 1.0)
        else:
            event_expected_impact = 0.0
        event_expected_impact = float(
            np.clip(event_expected_impact + rng.normal(0, 0.05), 0.0, 1.0)
        )

        # 15. time_seasonality – derive from mean service timestamp
        # Synthetic timestamps are np.arange(n) (integers 0–59), producing
        # a constant ~0.306 for all cases.  We set 0.5 (neutral) for
        # synthetic data; this feature only activates on real-world data
        # with epoch-based timestamps.
        # Epoch timestamps for dates after 2001 are > 1 billion; synthetic
        # sequential integers are typically < 1000.
        _SYNTHETIC_TS_THRESHOLD = 1_000_000
        if services:
            mean_ts = np.mean(
                [svc.metrics["timestamp"].mean() for svc in services]
            )
            # Detect synthetic timestamps: if mean_ts is below the threshold,
            # the timestamps are sequential integers, not epoch seconds.
            if mean_ts < _SYNTHETIC_TS_THRESHOLD:
                time_seasonality = 0.5
            else:
                hour = _datetime.fromtimestamp(
                    mean_ts, tz=_timezone.utc
                ).hour
                if 9 <= hour <= 20:
                    time_seasonality = 0.7 + 0.3 * (hour - 9) / 11.0
                else:
                    h = hour if hour < 9 else hour - 20
                    time_seasonality = 0.1 + 0.3 * h / 8.0
                time_seasonality = float(
                    np.clip(time_seasonality + rng.uniform(-0.05, 0.05), 0.0, 1.0)
                )
        else:
            time_seasonality = 0.5

        # 16. recent_deployment – label-independent base rate of 0.15
        if rng.random() < 0.15:
            recent_deployment = 0.3 * rng.random()
        else:
            recent_deployment = 0.0

        # 17. context_confidence (with Gaussian noise, std=0.1)
        conf = 0.0
        if "event_type" in ctx:
            conf += 0.3
        if "load_multiplier" in ctx:
            conf += 0.2
        if "event_name" in ctx:
            conf += 0.1
        context_confidence = float(np.clip(min(conf, 1.0) + rng.normal(0, 0.1), 0.0, 1.0))

        return np.array([
            event_active,
            event_expected_impact,
            time_seasonality,
            recent_deployment,
            context_confidence,
        ])

    # ------------------------------------------------------------------
    # Statistical features (13)
    # ------------------------------------------------------------------

    def _statistical_features(self, services: List[ServiceMetrics]) -> np.ndarray:
        if not services:
            return np.zeros(13)

        # Use numpy directly instead of pd.concat() to avoid DataFrame
        # allocation overhead per case.
        arrays = [svc.metrics[_STAT_METRIC_COLS].values for svc in services]
        combined = np.concatenate(arrays, axis=0)  # (total_rows, n_cols)

        means = np.mean(combined, axis=0)   # one mean per metric column
        stds = np.std(combined, axis=0, ddof=1)  # one std per metric column; ddof=1 matches pandas
        stds = np.nan_to_num(stds, nan=0.0)

        # max error_rate across all services
        max_error = float(
            max(svc.metrics["error_rate"].max() for svc in services)
        )

        return np.concatenate([means, stds, [max_error]])  # 13

    # ------------------------------------------------------------------
    # Service-level features (6)
    # ------------------------------------------------------------------

    def _service_level_features(self, services: List[ServiceMetrics]) -> np.ndarray:
        n = len(services)
        if n == 0:
            return np.zeros(6)

        cpu_means = np.array([svc.metrics["cpu_usage"].mean() for svc in services])
        err_means = np.array([svc.metrics["error_rate"].mean() for svc in services])
        lat_means = np.array([svc.metrics["latency"].mean() for svc in services])

        overall_cpu = float(np.mean(cpu_means))
        overall_err = float(np.mean(err_means))

        n_services = float(n)
        max_cpu_service_ratio = float(np.max(cpu_means) / (overall_cpu + 1e-10))
        max_error_service_ratio = float(np.max(err_means) / (overall_err + 1e-10))
        cpu_spread = float(np.std(cpu_means))
        error_spread = float(np.std(err_means))
        latency_spread = float(np.std(lat_means))

        return np.array([
            n_services,
            max_cpu_service_ratio,
            max_error_service_ratio,
            cpu_spread,
            error_spread,
            latency_spread,
        ])

    # ------------------------------------------------------------------
    # Extended features (8)
    # ------------------------------------------------------------------

    def _extended_features(self, case: AnomalyCase) -> np.ndarray:
        """Extract 8 extended features: frequency-domain, network, graph, rate-of-change."""
        services = case.services
        n = len(services)
        if n == 0:
            return np.zeros(8)

        # 1-3: Frequency-domain features (spectral_entropy, high_freq_energy_ratio, dominant_frequency)
        from scipy.signal import welch

        # Batch services with equal-length CPU arrays for a single welch() call
        cpu_by_len: Dict[int, List[np.ndarray]] = {}
        short_count = 0
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            if len(cpu) < 8:
                short_count += 1
                continue
            cpu_by_len.setdefault(len(cpu), []).append(cpu)

        entropies, hf_ratios, dom_freqs = (
            [0.0] * short_count,
            [0.0] * short_count,
            [0.0] * short_count,
        )
        for length, arrays in cpu_by_len.items():
            stacked = np.array(arrays)  # (n_svcs_with_len, length)
            freqs, psd_batch = welch(stacked, nperseg=min(length, 256), axis=-1)
            # psd_batch shape: (n_svcs, n_freqs)
            total = psd_batch.sum(axis=1, keepdims=True) + 1e-10
            psd_norm = psd_batch / total
            ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)
            nyquist_half = len(freqs) // 2
            hf = psd_batch[:, nyquist_half:].sum(axis=1) / total.squeeze()
            dom = freqs[np.argmax(psd_batch, axis=1)]
            entropies.extend(ent.tolist())
            hf_ratios.extend(hf.tolist())
            dom_freqs.extend(dom.tolist())

        spectral_entropy = float(np.mean(entropies)) if entropies else 0.0
        high_freq_energy_ratio = float(np.mean(hf_ratios)) if hf_ratios else 0.0
        dominant_frequency = float(np.mean(dom_freqs)) if dom_freqs else 0.0

        # 4: Network asymmetry
        asymmetries = []
        for svc in services:
            net_in = svc.metrics["network_in"].values
            net_out = svc.metrics["network_out"].values
            eps = 1e-10
            asym = np.mean(np.abs(net_in - net_out) / (net_in + net_out + eps))
            asymmetries.append(float(asym))
        network_asymmetry = float(np.mean(asymmetries))

        # 5-6: Graph structural features
        from src.data_loader.dataset import _SERVICE_ADJACENCY

        svc_map = {s.service_name: s for s in services}

        # Pre-compute cpu arrays and z-scores for graph features
        ext_cpu_arrays = [svc.metrics["cpu_usage"].values for svc in services]
        ext_zscores = [
            (cpu - np.mean(cpu)) / (np.std(cpu) + 1e-10) if np.std(cpu) > 0 else np.zeros_like(cpu)
            for cpu in ext_cpu_arrays
        ]

        # graph_anomaly_centrality: degree-weighted anomaly score
        centrality_scores = []
        for i, svc in enumerate(services):
            is_anomalous = float(np.any(ext_zscores[i] > 2.0))
            degree = len(_SERVICE_ADJACENCY.get(svc.service_name, []))
            centrality_scores.append(is_anomalous * degree)
        graph_anomaly_centrality = float(np.mean(centrality_scores)) if centrality_scores else 0.0

        # anomaly_spread: std of pairwise correlations between anomalous neighbors
        anomalous_corrs = []
        for i, svc in enumerate(services):
            if not np.any(ext_zscores[i] > 2.0):
                continue
            cpu = ext_cpu_arrays[i]
            neighbors = _SERVICE_ADJACENCY.get(svc.service_name, [])
            for nb_name in neighbors:
                if nb_name in svc_map:
                    nb_cpu = svc_map[nb_name].metrics["cpu_usage"].values
                    min_len = min(len(cpu), len(nb_cpu))
                    if min_len > 1:
                        anomalous_corrs.append(_safe_pearsonr(cpu[:min_len], nb_cpu[:min_len]))
        anomaly_spread = float(np.std(anomalous_corrs)) if len(anomalous_corrs) > 1 else 0.0

        # 7: max_cpu_derivative
        max_derivs = []
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            if len(cpu) < 2:
                max_derivs.append(0.0)
                continue
            deriv = np.diff(cpu)
            max_derivs.append(float(np.max(np.abs(deriv))))
        max_cpu_derivative = float(np.max(max_derivs)) if max_derivs else 0.0

        # 8: error_rate_slope
        slopes = []
        for svc in services:
            err = svc.metrics["error_rate"].values
            slopes.append(_linear_slope(err))
        error_rate_slope = float(np.mean(slopes))

        return np.array([
            spectral_entropy,
            high_freq_energy_ratio,
            dominant_frequency,
            network_asymmetry,
            graph_anomaly_centrality,
            anomaly_spread,
            max_cpu_derivative,
            error_rate_slope,
        ])
