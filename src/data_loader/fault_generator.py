"""Fault scenario metrics generator for microservice anomaly cases."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader.data_types import ServiceMetrics
from src.data_loader.synthetic_generator import SERVICE_NAMES
from src.data_loader.utils import generate_base_metrics

logger = logging.getLogger(__name__)

# All 11 fault types matching RCAEval benchmark specification.
FAULT_TYPES: List[str] = [
    "cpu_hog",
    "memory_leak",
    "network_delay",
    "packet_loss",
    "disk_io",
    "pod_failure",
    "dns_failure",
    "connection_pool_exhaustion",
    "thread_leak",
    "config_error",
    "dependency_failure",
]

# Mapping from synthetic fault type names to RCAEval short names.
# RCAEval uses: cpu, mem, disk, delay, loss, socket
# Synthetic types without a direct RCAEval equivalent map to the closest match.
FAULT_TYPE_TO_RCAEVAL = {
    "cpu_hog": "cpu",
    "memory_leak": "mem",
    "network_delay": "delay",
    "packet_loss": "loss",
    "disk_io": "disk",
    "pod_failure": "socket",
    "dns_failure": "socket",
    "connection_pool_exhaustion": "socket",
    "thread_leak": "cpu",
    "config_error": "cpu",
    "dependency_failure": "delay",
}

RCAEVAL_TO_FAULT_TYPES = {}
for synthetic, rcaeval in FAULT_TYPE_TO_RCAEVAL.items():
    RCAEVAL_TO_FAULT_TYPES.setdefault(rcaeval, []).append(synthetic)


class FaultGenerator:
    """Generates metrics that simulate fault injection in a microservice system.

    Supports all 11 fault types from the RCAEval benchmark:
        cpu_hog, memory_leak, network_delay, packet_loss, disk_io,
        pod_failure, dns_failure, connection_pool_exhaustion, thread_leak,
        config_error, dependency_failure.

    Attributes:
        n_services: Number of services to generate metrics for.
        sequence_length: Number of time steps per service.
    """

    SERVICE_NAMES = SERVICE_NAMES
    FAULT_TYPES = FAULT_TYPES

    def __init__(
        self,
        n_services: int = 12,
        sequence_length: int = 60,
        seed: int = 42,
    ) -> None:
        """Initialize the fault generator.

        Args:
            n_services: Number of microservices.
            sequence_length: Number of timesteps in each sequence.
            seed: Random seed for reproducibility.
        """
        self.n_services = n_services
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ar1_signal(self, length: int, lo: float, hi: float, phi: float = 0.9) -> np.ndarray:
        """Generate an AR(1) autocorrelated signal for realistic fault patterns.

        Uses ``scipy.signal.lfilter`` for a vectorised IIR filter implementation
        (runs in C) instead of a Python loop.

        Args:
            length: Number of timesteps.
            lo: Lower bound for random innovations.
            hi: Upper bound for random innovations.
            phi: Autoregression coefficient (0=white noise, 1=random walk).

        Returns:
            1-D array of length *length* with temporally correlated values.
        """
        from scipy.signal import lfilter

        innovations = self.rng.uniform(lo, hi, size=length)
        # IIR filter: y[t] = (1-phi)*x[t] + phi*y[t-1]
        # Transfer function: b=[1-phi], a=[1, -phi]
        return lfilter([1 - phi], [1, -phi], innovations)

    def _scale_mult(self, factor: float, sev: float) -> float:
        """Scale a multiplicative degradation factor by severity.
        Interpolates between 1.0 (no change) and the original factor.
        At sev=1.0: returns factor. At sev=0.0: returns 1.0.
        """
        return 1.0 - (1.0 - factor) * sev

    def _base_metrics(self, service_name: str) -> pd.DataFrame:
        """Generate a DataFrame of normal-operation metrics for one service.

        Args:
            service_name: Name of the service.

        Returns:
            DataFrame with normal baseline metrics.
        """
        return generate_base_metrics(self.sequence_length, service_name, rng=self.rng)

    # Severity factors scale the error_rate injection magnitude.
    # "low" creates subtle faults that overlap with load-induced errors,
    # forcing the model to rely on context for correct classification.
    SEVERITY_FACTORS = {"low": 0.05, "medium": 0.3, "high": 1.0}

    def _inject_fault(
        self, df: pd.DataFrame, fault_type: str, fault_start: int,
        severity: str = "high",
    ) -> pd.DataFrame:
        """Inject a fault into the metrics starting at fault_start.

        Args:
            df: Baseline metrics DataFrame.
            fault_type: One of the 11 supported fault types.
            fault_start: Index at which the fault begins.
            severity: Fault severity — "low", "medium", or "high".
                Controls the magnitude of error_rate injection.

        Returns:
            Modified DataFrame with the fault injected.
        """
        sev = self.SEVERITY_FACTORS[severity]
        df = df.copy()
        n = len(df)
        fault_len = n - fault_start
        fault_slice = slice(fault_start, n)

        if fault_type == "cpu_hog":
            ar_signal = self._ar1_signal(fault_len, 30 * sev, 60 * sev)
            df.loc[fault_slice, "cpu_usage"] = np.clip(
                df.loc[fault_slice, "cpu_usage"].values + ar_signal,
                0, 100,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.1, 0.5) * sev,
                0, 1,
            )

        elif fault_type == "memory_leak":
            # Gradual memory increase (leak pattern)
            leak_ramp = np.linspace(0, self.rng.uniform(30, 55) * sev, fault_len)
            df.loc[fault_slice, "memory_usage"] = np.clip(
                df.loc[fault_slice, "memory_usage"].values + leak_ramp,
                0, 100,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.1, 0.5) * sev,
                0, 1,
            )

        elif fault_type == "network_delay":
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + self._ar1_signal(fault_len, 200 * sev, 800 * sev),
                0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.05, 0.3) * sev,
                0, 1,
            )

        elif fault_type == "packet_loss":
            df.loc[fault_slice, "network_in"] = np.clip(
                df.loc[fault_slice, "network_in"].values * self._scale_mult(0.1, sev), 0, None,
            )
            df.loc[fault_slice, "network_out"] = np.clip(
                df.loc[fault_slice, "network_out"].values * self._scale_mult(0.1, sev), 0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.1, 0.5) * sev,
                0, 1,
            )

        elif fault_type == "disk_io":
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + self._ar1_signal(fault_len, 100 * sev, 500 * sev),
                0, None,
            )
            df.loc[fault_slice, "cpu_usage"] = np.clip(
                df.loc[fault_slice, "cpu_usage"].values
                + self._ar1_signal(fault_len, 10 * sev, 30 * sev),
                0, 100,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.05, 0.2) * sev,
                0, 1,
            )

        elif fault_type == "pod_failure":
            # Service goes completely down — requests drop, errors spike
            df.loc[fault_slice, "request_rate"] = np.clip(
                df.loc[fault_slice, "request_rate"].values * self._scale_mult(0.05, sev), 0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.4, 0.8) * sev,
                0, 1,
            )
            df.loc[fault_slice, "cpu_usage"] = np.clip(
                df.loc[fault_slice, "cpu_usage"].values * self._scale_mult(0.1, sev), 0, 100,
            )

        elif fault_type == "dns_failure":
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + self._ar1_signal(fault_len, 500 * sev, 2000 * sev),
                0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.2, 0.6) * sev,
                0, 1,
            )
            df.loc[fault_slice, "network_in"] = np.clip(
                df.loc[fault_slice, "network_in"].values * self._scale_mult(0.3, sev), 0, None,
            )

        elif fault_type == "connection_pool_exhaustion":
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + self._ar1_signal(fault_len, 300 * sev, 1000 * sev),
                0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.15, 0.5) * sev,
                0, 1,
            )
            # Requests back up
            df.loc[fault_slice, "request_rate"] = np.clip(
                df.loc[fault_slice, "request_rate"].values * self._scale_mult(0.4, sev), 0, None,
            )

        elif fault_type == "thread_leak":
            # CPU climbs gradually, latency grows
            leak_ramp = np.linspace(0, self.rng.uniform(20, 50) * sev, fault_len)
            df.loc[fault_slice, "cpu_usage"] = np.clip(
                df.loc[fault_slice, "cpu_usage"].values + leak_ramp,
                0, 100,
            )
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + self._ar1_signal(fault_len, 50 * sev, 300 * sev),
                0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.05, 0.3) * sev,
                0, 1,
            )

        elif fault_type == "config_error":
            # Immediate error spike, some latency increase
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.2, 0.7) * sev,
                0, 1,
            )
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + self._ar1_signal(fault_len, 50 * sev, 200 * sev),
                0, None,
            )

        elif fault_type == "dependency_failure":
            # Downstream dependency dies — errors spike, latency goes up
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + self.rng.uniform(0.2, 0.6) * sev,
                0, 1,
            )
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + self._ar1_signal(fault_len, 200 * sev, 600 * sev),
                0, None,
            )
            df.loc[fault_slice, "request_rate"] = np.clip(
                df.loc[fault_slice, "request_rate"].values * self._scale_mult(0.5, sev), 0, None,
            )

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject_fault(
        self, df: pd.DataFrame, fault_type: str, fault_start: int
    ) -> pd.DataFrame:
        """Inject a fault into the metrics starting at fault_start.

        Public wrapper around the internal fault injection logic.

        Args:
            df: Baseline metrics DataFrame.
            fault_type: One of the 11 supported fault types.
            fault_start: Index at which the fault begins.

        Returns:
            Modified DataFrame with the fault injected.
        """
        return self._inject_fault(df, fault_type, fault_start)

    def generate_base_metrics(self, service_name: str) -> pd.DataFrame:
        """Generate a DataFrame of normal-operation metrics for one service.

        Public wrapper around the internal base metrics generation.

        Args:
            service_name: Name of the service.

        Returns:
            DataFrame with normal baseline metrics.
        """
        return self._base_metrics(service_name)

    def generate_fault_metrics(
        self,
        system: str = "online-boutique",
        fault_type: Optional[str] = None,
        fault_service: Optional[str] = None,
        case_seed: Optional[int] = None,
        severity: str = "high",
    ) -> Tuple[List[ServiceMetrics], str, str]:
        """Generate metrics that simulate a fault in one microservice.

        Characteristics:
            - Error rate increases on the faulty service.
            - Sudden onset (step change) at a random point in the middle third.
            - Localised to a single service; others remain normal.
            - Fault type determines which metrics are affected.

        Args:
            system: Name of the system.
            fault_type: Type of fault to inject. Random from the 11 supported
                types if not given.
            fault_service: Service to inject the fault into. Random (excluding
                loadgenerator) if not given.
            case_seed: When provided, an independent RNG seeded with this
                value is used for this call, avoiding sequential state
                dependence on ``self.rng``.

        Returns:
            Tuple of (list of ServiceMetrics, fault_service name, fault_type).
        """
        if case_seed is not None:
            original_rng = self.rng
            self.rng = np.random.default_rng(case_seed)

        eligible_services = [s for s in self.SERVICE_NAMES[: self.n_services] if s != "loadgenerator"]

        if fault_type is None:
            fault_type = str(self.rng.choice(self.FAULT_TYPES))
        if fault_service is None:
            fault_service = str(self.rng.choice(eligible_services))

        # Fault starts at a random point in the middle third of the sequence
        mid_start = self.sequence_length // 3
        mid_end = 2 * self.sequence_length // 3
        fault_start = int(self.rng.integers(mid_start, mid_end))

        logger.info(
            "Generating fault metrics: system=%s service=%s type=%s start=%d",
            system,
            fault_service,
            fault_type,
            fault_start,
        )

        # Fork RNG for base metrics so counterfactual can reproduce identical noise
        base_rng_seed = int(self.rng.integers(0, 2**63))
        base_rng = np.random.default_rng(base_rng_seed)

        results: List[ServiceMetrics] = []
        for name in self.SERVICE_NAMES[: self.n_services]:
            df = generate_base_metrics(self.sequence_length, name, rng=base_rng)
            if name == fault_service:
                df = self._inject_fault(df, fault_type, fault_start, severity=severity)
            results.append(ServiceMetrics(service_name=name, metrics=df))

        if case_seed is not None:
            self.rng = original_rng

        return results, fault_service, fault_type

    def generate_disguised_fault(
        self,
        system: str = "online-boutique",
        fault_type: Optional[str] = None,
        fault_service: Optional[str] = None,
        case_seed: Optional[int] = None,
    ) -> Tuple[List[ServiceMetrics], str, str]:
        """Generate a fault that mimics a load spike in its metric pattern.

        All services get a load-like envelope (ramp-up, plateau, ramp-down).
        The faulty service gets a small additional perturbation on top.
        Without context, this is metrically indistinguishable from a load event.

        Args:
            system: Name of the system.
            fault_type: Type of fault. Random if not given.
            fault_service: Service to inject fault into. Random if not given.
            case_seed: Independent RNG seed for this call.

        Returns:
            Tuple of (list of ServiceMetrics, fault_service name, fault_type).
        """
        if case_seed is not None:
            original_rng = self.rng
            self.rng = np.random.default_rng(case_seed)

        eligible_services = [s for s in self.SERVICE_NAMES[: self.n_services] if s != "loadgenerator"]

        if fault_type is None:
            fault_type = str(self.rng.choice(self.FAULT_TYPES))
        if fault_service is None:
            fault_service = str(self.rng.choice(eligible_services))

        n = self.sequence_length

        # Build load-like envelope (identical to SyntheticMetricsGenerator)
        ramp_frac = self.rng.uniform(0.10, 0.20)
        ramp_len = max(1, int(n * ramp_frac))
        spike_start = int(self.rng.integers(int(n * 0.15), int(n * 0.35)))
        spike_end = min(n, spike_start + int(self.rng.integers(int(n * 0.3), int(n * 0.5))))
        ramp_down_start = max(spike_start + ramp_len, spike_end - ramp_len)

        envelope = np.zeros(n)
        up_end = min(spike_start + ramp_len, n)
        envelope[spike_start:up_end] = np.linspace(0, 1, up_end - spike_start)
        envelope[up_end:ramp_down_start] = 1.0
        if ramp_down_start < spike_end:
            envelope[ramp_down_start:spike_end] = np.linspace(
                1, 0, spike_end - ramp_down_start,
            )

        # Fake load multiplier (same range as real load events)
        load_multiplier = float(self.rng.uniform(2.0, 5.0))

        logger.info(
            "Generating disguised fault: system=%s service=%s type=%s mult=%.1f",
            system, fault_service, fault_type, load_multiplier,
        )

        base_rng_seed = int(self.rng.integers(0, 2**63))
        base_rng = np.random.default_rng(base_rng_seed)

        results: List[ServiceMetrics] = []
        for name in self.SERVICE_NAMES[: self.n_services]:
            df = generate_base_metrics(self.sequence_length, name, rng=base_rng)

            # Apply load-like pattern to ALL services (mimics load spike)
            svc_mult = 1.0 + (load_multiplier - 1.0) * envelope * self.rng.uniform(0.7, 1.3)
            df["cpu_usage"] = np.clip(df["cpu_usage"] * svc_mult, 0, 100)
            df["request_rate"] = np.clip(df["request_rate"] * svc_mult, 0, None)
            df["latency"] = np.clip(df["latency"] * svc_mult, 0, None)
            df["network_in"] = np.clip(df["network_in"] * svc_mult, 0, None)
            df["network_out"] = np.clip(df["network_out"] * svc_mult, 0, None)

            # Proportional error increase (same as load spikes)
            err_increase = (load_multiplier - 1.0) * self.rng.uniform(0.002, 0.01) * envelope
            df["error_rate"] = np.clip(df["error_rate"] + err_increase, 0, 1)

            # Fault perturbation on the faulty service only
            if name == fault_service:
                df["cpu_usage"] = np.clip(
                    df["cpu_usage"] + self.rng.uniform(3, 8) * envelope, 0, 100,
                )
                df["error_rate"] = np.clip(
                    df["error_rate"] + self.rng.uniform(0.01, 0.03) * envelope, 0, 1,
                )
                df["latency"] = np.clip(
                    df["latency"] + self.rng.uniform(20, 80) * envelope, 0, None,
                )

            results.append(ServiceMetrics(service_name=name, metrics=df))

        if case_seed is not None:
            self.rng = original_rng

        return results, fault_service, fault_type

    def generate_counterfactual_fault(
        self,
        case_seed: int,
        system: str = "online-boutique",
        severity: str = "high",
    ) -> List[ServiceMetrics]:
        """Generate counterfactual baseline: same seed as generate_fault_metrics, no injection.

        Replays the exact RNG sequence up to the base_rng fork point,
        then generates only base metrics with the same forked RNG.
        """
        rng = np.random.default_rng(case_seed)
        eligible = [s for s in self.SERVICE_NAMES[: self.n_services] if s != "loadgenerator"]

        # Replay pre-loop RNG calls (matching generate_fault_metrics)
        rng.choice(self.FAULT_TYPES)      # fault_type selection
        rng.choice(eligible)               # fault_service selection
        rng.integers(self.sequence_length // 3, 2 * self.sequence_length // 3)  # fault_start

        # Fork base_rng identically
        base_rng_seed = int(rng.integers(0, 2**63))
        base_rng = np.random.default_rng(base_rng_seed)

        results: List[ServiceMetrics] = []
        for name in self.SERVICE_NAMES[: self.n_services]:
            df = generate_base_metrics(self.sequence_length, name, rng=base_rng)
            results.append(ServiceMetrics(service_name=name, metrics=df))
        return results

    def generate_counterfactual_disguised(
        self,
        case_seed: int,
        system: str = "online-boutique",
    ) -> List[ServiceMetrics]:
        """Generate counterfactual baseline: same seed as generate_disguised_fault, no injection."""
        rng = np.random.default_rng(case_seed)
        eligible = [s for s in self.SERVICE_NAMES[: self.n_services] if s != "loadgenerator"]

        # Replay pre-loop RNG calls (matching generate_disguised_fault)
        rng.choice(self.FAULT_TYPES)
        rng.choice(eligible)
        n = self.sequence_length
        rng.uniform(0.10, 0.20)                                    # ramp_frac
        rng.integers(int(n * 0.15), int(n * 0.35))                # spike_start
        rng.integers(int(n * 0.3), int(n * 0.5))                  # spike_end
        rng.uniform(2.0, 5.0)                                      # load_multiplier

        # Fork base_rng identically
        base_rng_seed = int(rng.integers(0, 2**63))
        base_rng = np.random.default_rng(base_rng_seed)

        results: List[ServiceMetrics] = []
        for name in self.SERVICE_NAMES[: self.n_services]:
            df = generate_base_metrics(self.sequence_length, name, rng=base_rng)
            results.append(ServiceMetrics(service_name=name, metrics=df))
        return results
