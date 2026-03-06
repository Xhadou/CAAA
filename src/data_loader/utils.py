"""Shared utilities for data generation modules."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Per-service baseline profiles — realistic differentiation between
# microservice roles.  Each service maps to (min, max) ranges for
# cpu_usage, memory_usage, request_rate, error_rate, latency,
# network_in, network_out.
SERVICE_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "frontend": {
        "cpu": (15, 35), "mem": (20, 35), "req": (200, 500),
        "err": (0.001, 0.008), "lat": (20, 80),
        "net_in": (2000, 8000), "net_out": (3000, 10000),
    },
    "cart": {
        "cpu": (8, 20), "mem": (15, 30), "req": (80, 250),
        "err": (0.001, 0.005), "lat": (5, 40),
        "net_in": (500, 2000), "net_out": (500, 2000),
    },
    "checkout": {
        "cpu": (10, 25), "mem": (20, 40), "req": (50, 150),
        "err": (0.002, 0.01), "lat": (30, 120),
        "net_in": (1000, 4000), "net_out": (1000, 4000),
    },
    "payment": {
        "cpu": (5, 15), "mem": (10, 25), "req": (30, 100),
        "err": (0.001, 0.005), "lat": (50, 200),
        "net_in": (500, 2000), "net_out": (500, 2000),
    },
    "shipping": {
        "cpu": (5, 15), "mem": (10, 20), "req": (30, 100),
        "err": (0.001, 0.005), "lat": (20, 80),
        "net_in": (300, 1500), "net_out": (300, 1500),
    },
    "email": {
        "cpu": (3, 10), "mem": (8, 18), "req": (10, 50),
        "err": (0.001, 0.003), "lat": (10, 60),
        "net_in": (200, 1000), "net_out": (500, 2000),
    },
    "currency": {
        "cpu": (5, 12), "mem": (8, 15), "req": (100, 300),
        "err": (0.001, 0.003), "lat": (5, 20),
        "net_in": (300, 1000), "net_out": (300, 1000),
    },
    "productcatalog": {
        "cpu": (10, 25), "mem": (25, 50), "req": (100, 400),
        "err": (0.001, 0.005), "lat": (10, 50),
        "net_in": (1000, 5000), "net_out": (2000, 8000),
    },
    "recommendation": {
        "cpu": (15, 40), "mem": (30, 60), "req": (80, 250),
        "err": (0.001, 0.005), "lat": (15, 60),
        "net_in": (500, 3000), "net_out": (500, 3000),
    },
    "ad": {
        "cpu": (8, 20), "mem": (15, 30), "req": (100, 400),
        "err": (0.002, 0.008), "lat": (10, 40),
        "net_in": (500, 2000), "net_out": (1000, 4000),
    },
    "redis-cart": {
        "cpu": (5, 15), "mem": (40, 70), "req": (100, 300),
        "err": (0.0005, 0.002), "lat": (1, 10),
        "net_in": (500, 3000), "net_out": (500, 3000),
    },
    "loadgenerator": {
        "cpu": (20, 50), "mem": (15, 30), "req": (200, 600),
        "err": (0.001, 0.005), "lat": (5, 30),
        "net_in": (2000, 8000), "net_out": (2000, 8000),
    },
}

# Default profile for services not listed above.
_DEFAULT_PROFILE: Dict[str, Tuple[float, float]] = {
    "cpu": (10, 30), "mem": (20, 40), "req": (50, 200),
    "err": (0.001, 0.01), "lat": (10, 100),
    "net_in": (1000, 5000), "net_out": (1000, 5000),
}


def generate_base_metrics(sequence_length: int, service_name: str) -> pd.DataFrame:
    """Generate a DataFrame of normal-operation metrics for one service.

    Uses per-service baseline profiles for realistic differentiation
    between microservice roles (e.g. redis-cart has low CPU but high
    memory, frontend has high request rates).

    Args:
        sequence_length: Number of timesteps in the sequence.
        service_name: Name of the service.

    Returns:
        DataFrame with columns: timestamp, cpu_usage, memory_usage,
        request_rate, error_rate, latency, network_in, network_out.
    """
    n = sequence_length
    noise = lambda scale: np.random.normal(0, scale, n)
    profile = SERVICE_PROFILES.get(service_name, _DEFAULT_PROFILE)

    cpu = np.random.uniform(*profile["cpu"]) + noise(2)
    mem = np.random.uniform(*profile["mem"]) + noise(1.5)
    req = np.random.uniform(*profile["req"]) + noise(5)
    err = np.random.uniform(*profile["err"]) + noise(0.001)
    lat = np.random.uniform(*profile["lat"]) + noise(3)
    net_in = np.random.uniform(*profile["net_in"]) + noise(50)
    net_out = np.random.uniform(*profile["net_out"]) + noise(50)

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
