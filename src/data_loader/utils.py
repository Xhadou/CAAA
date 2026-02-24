"""Shared utilities for data generation modules."""

from typing import Optional

import numpy as np
import pandas as pd


def generate_base_metrics(
    sequence_length: int,
    service_name: str,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Generate a DataFrame of normal-operation metrics for one service.

    This is the shared implementation used by both
    :class:`~src.data_loader.synthetic_generator.SyntheticMetricsGenerator` and
    :class:`~src.data_loader.fault_generator.FaultGenerator`.

    Args:
        sequence_length: Number of timesteps in the sequence.
        service_name: Name of the service (unused currently, but kept for
            future per-service baseline differentiation).
        rng: NumPy random Generator instance.  When *None* a new
            unseeded generator is created (non-deterministic).

    Returns:
        DataFrame with columns: timestamp, cpu_usage, memory_usage,
        request_rate, error_rate, latency, network_in, network_out.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = sequence_length

    def noise(scale):
        return rng.normal(0, scale, n)

    cpu = rng.uniform(10, 30) + noise(2)
    mem = rng.uniform(20, 40) + noise(1.5)
    req = rng.uniform(50, 200) + noise(5)
    err = rng.uniform(0.001, 0.01) + noise(0.001)
    lat = rng.uniform(10, 100) + noise(3)
    net_in = rng.uniform(1000, 5000) + noise(50)
    net_out = rng.uniform(1000, 5000) + noise(50)

    # Clamp to sensible ranges
    cpu = np.clip(cpu, 0, 100)
    mem = np.clip(mem, 0, 100)
    req = np.clip(req, 0, None)
    err = np.clip(err, 0, 1)
    lat = np.clip(lat, 0, None)
    net_in = np.clip(net_in, 0, None)
    net_out = np.clip(net_out, 0, None)

    return pd.DataFrame(
        {
            "timestamp": np.arange(n),
            "cpu_usage": cpu,
            "memory_usage": mem,
            "request_rate": req,
            "error_rate": err,
            "latency": lat,
            "network_in": net_in,
            "network_out": net_out,
        }
    )
