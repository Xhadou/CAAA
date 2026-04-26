"""Combined dataset generation for CAAA anomaly attribution."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data_loader.data_types import AnomalyCase, ServiceMetrics
from src.data_loader.fault_generator import FaultGenerator
from src.data_loader.synthetic_generator import SyntheticMetricsGenerator, EVENT_TYPES

logger = logging.getLogger(__name__)

# Systems matching RCAEval benchmark.
RESEARCH_SYSTEMS: List[str] = ["online-boutique", "sock-shop", "train-ticket"]

# Multiplier used to derive per-case seeds from the base seed and case index.
# Must be large enough to avoid overlap between different generation phases
# (e.g., seed * _CASE_SEED_MULT + i vs (seed+1) * _CASE_SEED_MULT + j).
_CASE_SEED_MULT: int = 10000


def generate_combined_dataset(
    n_fault: int = 50,
    n_load: int = 50,
    systems: Optional[List[str]] = None,
    seed: int = 42,
    include_hard: bool = True,
    difficulty: str = "default",
) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
    """Generate a combined dataset of FAULT and EXPECTED_LOAD cases.

    Args:
        n_fault: Number of fault cases to generate.
        n_load: Number of expected-load cases to generate.
        systems: List of system names to cycle through. Defaults to
            ["online-boutique"].
        seed: Random seed for reproducibility.
        include_hard: When True, replace ~35% of cases with
            hard/adversarial scenarios from :func:`generate_hard_dataset`.
        difficulty: Task difficulty level. One of:
            - ``"default"``: Standard 35/35/30 low/medium/high severity mix with
              normal severity factors (0.05/0.3/1.0). Achieves ~97% F1 ceiling.
                        - ``"moderate"``: 60/25/15 severity mix with reduced severity factors
              (0.02/0.15/0.7) and higher context noise. Lowers ceiling to
              ~95% so model architecture differences become visible.
                        - ``"hard"``: 70/20/10 severity mix with further-reduced
              severity factors (0.01/0.10/0.50) and context noise 0.60.
                            Target ceiling ~88-92% to expose differences that "moderate"
              40K saturation still masks.

    Returns:
        Tuple of (fault_cases, load_cases).
    """
    if systems is None:
        systems = ["online-boutique"]

    # Difficulty profile controls the severity distribution, severity factors
    # scaling, and context-noise rate. Harder modes intentionally lower the
    # empirical saturation plateau so that architectural differences (neural vs
    # tree, context vs no context) become visible.
    if difficulty == "hard":
        # Harshest principled setting: maximises task difficulty to test the
        # H2 architectural-crossover hypothesis at its most extreme. Predicted
        # plateau ~82-87% F1. We commit to reporting this outcome regardless
        # of what falls out (no parameter search beyond this).
        _SEVERITY_DIST = [("low", 0.85), ("medium", 0.12), ("high", 0.03)]
        _severity_factors = {"low": 0.003, "medium": 0.03, "high": 0.20}
        _fake_context_rate = 0.75
        _empty_context_rate = 0.75
    elif difficulty == "moderate":
        _SEVERITY_DIST = [("low", 0.60), ("medium", 0.25), ("high", 0.15)]
        _severity_factors = {"low": 0.02, "medium": 0.15, "high": 0.7}
        _fake_context_rate = 0.50
        _empty_context_rate = 0.50
    else:
        _SEVERITY_DIST = [("low", 0.35), ("medium", 0.35), ("high", 0.30)]
        _severity_factors = None  # use class default
        _fake_context_rate = 0.30
        _empty_context_rate = 0.30

    rng = np.random.default_rng(seed)
    fault_gen = FaultGenerator(seed=seed, severity_factors=_severity_factors)
    load_gen = SyntheticMetricsGenerator(seed=seed + 1)

    fault_cases: List[AnomalyCase] = []
    load_cases: List[AnomalyCase] = []

    # Generate fault cases with mixed severity levels.
    # Low-severity faults produce subtle error increases that overlap with
    # load-induced errors, forcing the model to rely on context features.

    logger.info("Generating %d fault cases", n_fault)
    for i in range(n_fault):
        system = systems[i % len(systems)]
        case_seed = seed * _CASE_SEED_MULT + i

        # Assign severity based on distribution
        roll = rng.random()
        if roll < _SEVERITY_DIST[0][1]:
            severity = "low"
            case_difficulty = "hard"
        elif roll < _SEVERITY_DIST[0][1] + _SEVERITY_DIST[1][1]:
            severity = "medium"
            case_difficulty = "medium"
        else:
            severity = "high"
            case_difficulty = "easy"

        if severity == "low":
            # Disguised fault: mimics load spike pattern, empty context
            services, fault_service, fault_type = fault_gen.generate_disguised_fault(
                system=system, case_seed=case_seed,
            )
            fault_context = {}
        else:
            services, fault_service, fault_type = fault_gen.generate_fault_metrics(
                system=system, case_seed=case_seed, severity=severity,
            )
            fault_context = {}
            if rng.random() < 0.3:
                fault_context["recent_deployment"] = True
            # Some fault cases get a fake context with event_type to prevent
            # event_active from being a perfect proxy for the label. Rate is
            # controlled by difficulty mode (50% for moderate, 75% for hard).
            if rng.random() < _fake_context_rate:
                fake_event = str(rng.choice([
                    "flash_sale", "marketing_campaign", "scheduled_batch",
                ]))
                fault_context["event_type"] = fake_event
                fault_context["event_name"] = f"{fake_event}_event"
                fault_context["load_multiplier"] = float(
                    rng.uniform(1.2, 2.5)
                )
        # Counterfactual baseline: same seed, no injection
        if severity == "low":
            reference_services = fault_gen.generate_counterfactual_disguised(
                case_seed=case_seed, system=system,
            )
        else:
            reference_services = fault_gen.generate_counterfactual_fault(
                case_seed=case_seed, system=system, severity=severity,
            )
        fault_cases.append(
            AnomalyCase(
                case_id=f"fault_{i:04d}",
                system=system,
                label="FAULT",
                services=services,
                context=fault_context,
                fault_service=fault_service,
                fault_type=fault_type,
                difficulty=case_difficulty,
                reference_services=reference_services,
            )
        )

    # Generate expected-load cases.
    # Load spikes now produce proportional error increases (via A2),
    # so high-multiplier loads are "hard" (their errors overlap with
    # low-severity faults).
    logger.info("Generating %d expected-load cases", n_load)
    for i in range(n_load):
        system = systems[i % len(systems)]
        case_seed = (seed + 1) * _CASE_SEED_MULT + i
        services, context = load_gen.generate_load_spike_metrics(
            system=system, case_seed=case_seed,
        )
        # Assign difficulty based on load multiplier
        mult = context.get("load_multiplier", 1.0)
        if mult >= 4.0:
            load_difficulty = "hard"
        elif mult >= 2.5:
            load_difficulty = "medium"
        else:
            load_difficulty = "easy"

        # Some load cases get empty context (simulating unscheduled load
        # spikes with no calendar entry) to prevent label leakage. Rate is
        # controlled by difficulty mode (50% for moderate, 75% for hard).
        if rng.random() < _empty_context_rate:
            context = {}
        # Counterfactual baseline: same seed, no load injection
        reference_services = load_gen.generate_counterfactual_load(
            case_seed=case_seed, system=system,
        )
        load_cases.append(
            AnomalyCase(
                case_id=f"load_{i:04d}",
                system=system,
                label="EXPECTED_LOAD",
                services=services,
                context=context,
                difficulty=load_difficulty,
                reference_services=reference_services,
            )
        )

    # Optionally replace ~35% of cases with hard/adversarial scenarios
    if include_hard and (n_fault > 0 or n_load > 0):
        # 3 fault hard types at ~12% each ≈ 36% fault replacement
        n_fault_per_type = max(1, int(n_fault * 0.12)) if n_fault > 0 else 0
        # 2 load hard types at ~15% each ≈ 30% load replacement
        n_load_per_type = max(1, int(n_load * 0.15)) if n_load > 0 else 0
        hard_fault, hard_load = generate_hard_dataset(
            n_fault_per_type=n_fault_per_type,
            n_load_per_type=n_load_per_type,
            systems=systems,
            seed=seed + 100,
        )
        # Replace tail of standard cases with hard cases
        if hard_fault:
            fault_cases = fault_cases[: max(0, len(fault_cases) - len(hard_fault))] + hard_fault
        if hard_load:
            load_cases = load_cases[: max(0, len(load_cases) - len(hard_load))] + hard_load

    logger.info(
        "Dataset complete: %d fault cases, %d load cases",
        len(fault_cases),
        len(load_cases),
    )
    return fault_cases, load_cases


def generate_research_dataset(
    seed: int = 42,
) -> Dict[str, List[AnomalyCase]]:
    """Generate the full research dataset matching the specification.

    Produces 735 FAULT + 600 EXPECTED_LOAD = 1335 total cases across
    3 systems (online-boutique, sock-shop, train-ticket), split into
    train / val / test partitions:

    +---------+-------+---------------+-------+
    | Split   | FAULT | EXPECTED_LOAD | Total |
    +---------+-------+---------------+-------+
    | train   |   500 |           400 |   900 |
    | val     |   100 |           100 |   200 |
    | test    |   135 |           100 |   235 |
    +---------+-------+---------------+-------+
    | Total   |   735 |           600 | 1,335 |
    +---------+-------+---------------+-------+

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``, each
        mapping to a list of ``AnomalyCase`` objects.
    """
    systems = RESEARCH_SYSTEMS

    fault_cases, load_cases = generate_combined_dataset(
        n_fault=735, n_load=600, systems=systems, seed=seed,
    )

    rng = np.random.default_rng(seed)

    # Shuffle within each class
    fault_idx = rng.permutation(len(fault_cases)).tolist()
    load_idx = rng.permutation(len(load_cases)).tolist()

    # Split FAULT: 500 train, 100 val, 135 test
    fault_train = [fault_cases[i] for i in fault_idx[:500]]
    fault_val = [fault_cases[i] for i in fault_idx[500:600]]
    fault_test = [fault_cases[i] for i in fault_idx[600:735]]

    # Split EXPECTED_LOAD: 400 train, 100 val, 100 test
    load_train = [load_cases[i] for i in load_idx[:400]]
    load_val = [load_cases[i] for i in load_idx[400:500]]
    load_test = [load_cases[i] for i in load_idx[500:600]]

    train = fault_train + load_train
    val = fault_val + load_val
    test = fault_test + load_test

    # Shuffle each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    logger.info(
        "Research dataset: train=%d, val=%d, test=%d (total=%d)",
        len(train), len(val), len(test),
        len(train) + len(val) + len(test),
    )

    return {"train": train, "val": val, "test": test}


def generate_rcaeval_dataset(
    dataset: "str | List[str]" = "RE1",
    system: "str | List[str]" = "online-boutique",
    n_load_per_fault: int = 1,
    data_dir: str = "data/raw",
    seed: int = 42,
    split_mode: str = "pre_injection",
) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
    """Load FAULT cases from RCAEval and produce matching EXPECTED_LOAD cases.

    Two modes are supported via *split_mode*:

    * ``"pre_injection"`` (default): calls
      :meth:`RCAEvalLoader.load_dataset_split` which splits each real
      recording at its injection timestamp.  The post-injection window
      becomes a FAULT case and the pre-injection window becomes an
      EXPECTED_LOAD case.  Both halves come from the same real recording,
      avoiding distribution mismatch.  *n_load_per_fault* is ignored in
      this mode.

    * ``"synthetic"``: loads all FAULT cases via
      :meth:`RCAEvalLoader.load_dataset`, then generates
      *n_load_per_fault* synthetic EXPECTED_LOAD cases per fault case
      using :class:`SyntheticMetricsGenerator`.  This was the original
      behaviour.

    Args:
        dataset: RCAEval dataset identifier(s). A single string like ``"RE1"``
            or a list like ``["RE1", "RE2", "RE3"]``.
        system: Microservice system(s). A single string like
            ``"online-boutique"`` or a list like
            ``["online-boutique", "sock-shop", "train-ticket"]``.
        n_load_per_fault: Number of synthetic load cases per fault case.
            Only used when *split_mode* is ``"synthetic"``.
        data_dir: Path to downloaded RCAEval data.
        seed: Random seed (used only in ``"synthetic"`` mode).
        split_mode: One of ``"pre_injection"`` or ``"synthetic"``.

    Returns:
        Tuple of (fault_cases, load_cases) as AnomalyCase lists.

    Raises:
        FileNotFoundError: If none of the requested RCAEval combinations
            contain data.
        ValueError: If *split_mode* is not a recognised value.
    """
    from src.data_loader.rcaeval_loader import RCAEvalLoader

    if split_mode not in ("pre_injection", "synthetic"):
        raise ValueError(
            f"split_mode must be 'pre_injection' or 'synthetic', got {split_mode!r}"
        )

    datasets = [dataset] if isinstance(dataset, str) else list(dataset)
    systems = [system] if isinstance(system, str) else list(system)

    loader = RCAEvalLoader(data_dir=data_dir)
    fault_cases: List[AnomalyCase] = []
    load_cases: List[AnomalyCase] = []

    if split_mode == "pre_injection":
        for ds in datasets:
            for sys_name in systems:
                try:
                    ds_faults, ds_normals = loader.load_dataset_split(
                        dataset=ds, system=sys_name, seed=seed,
                    )
                except FileNotFoundError:
                    logger.warning(
                        "No data at %s/%s/%s — skipping", data_dir, ds, sys_name
                    )
                    continue
                if ds_faults or ds_normals:
                    logger.info(
                        "Loaded %d fault / %d normal cases from RCAEval %s/%s",
                        len(ds_faults), len(ds_normals), ds, sys_name,
                    )
                    fault_cases.extend(ds_faults)
                    load_cases.extend(ds_normals)
                else:
                    logger.warning("Empty dataset for %s/%s — skipping", ds, sys_name)

        if not fault_cases and not load_cases:
            raise FileNotFoundError(
                f"No RCAEval data found for datasets={datasets}, systems={systems} "
                f"in {data_dir}. Download the data first."
            )

        logger.info(
            "RCAEval dataset (pre_injection): %d fault cases + %d normal cases = %d total",
            len(fault_cases), len(load_cases), len(fault_cases) + len(load_cases),
        )

    else:  # split_mode == "synthetic"
        for ds in datasets:
            for sys_name in systems:
                try:
                    cases = loader.load_dataset(dataset=ds, system=sys_name)
                except FileNotFoundError:
                    logger.warning(
                        "No data at %s/%s/%s — skipping", data_dir, ds, sys_name
                    )
                    continue
                if cases:
                    logger.info(
                        "Loaded %d fault cases from RCAEval %s/%s", len(cases), ds, sys_name
                    )
                    fault_cases.extend(cases)
                else:
                    logger.warning("Empty dataset for %s/%s — skipping", ds, sys_name)

        if not fault_cases:
            raise FileNotFoundError(
                f"No RCAEval data found for datasets={datasets}, systems={systems} "
                f"in {data_dir}. Download the data first."
            )

        # Generate synthetic EXPECTED_LOAD cases
        load_gen = SyntheticMetricsGenerator(seed=seed)
        for i, fault_case in enumerate(fault_cases):
            for j in range(n_load_per_fault):
                services, context = load_gen.generate_load_spike_metrics(
                    system=fault_case.system,
                )
                load_cases.append(
                    AnomalyCase(
                        case_id=f"load_rcaeval_{i:04d}_{j:02d}",
                        system=fault_case.system,
                        label="EXPECTED_LOAD",
                        services=services,
                        context=context,
                    )
                )

        logger.info(
            "RCAEval dataset (synthetic): %d real faults + %d synthetic loads = %d total",
            len(fault_cases), len(load_cases), len(fault_cases) + len(load_cases),
        )

    return fault_cases, load_cases


# ── Hard / adversarial scenario types ────────────────────────────────

HARD_SCENARIO_TYPES: List[str] = [
    "fault_during_event",
    "capacity_exceeded_load",
    "gradual_fault",
    "correlated_fault",
    "partial_load",
]

# Online-Boutique service dependency graph used for correlated_fault
# scenarios.  Each key maps to services that would be impacted by a
# fault propagating from that service.
_SERVICE_ADJACENCY: Dict[str, List[str]] = {
    "frontend": ["cart", "checkout", "productcatalog", "recommendation", "ad", "currency"],
    "cart": ["frontend", "redis-cart"],
    "checkout": ["frontend", "cart", "payment", "shipping", "email", "currency", "productcatalog"],
    "payment": ["checkout"],
    "shipping": ["checkout"],
    "email": ["checkout"],
    "currency": ["frontend", "checkout"],
    "productcatalog": ["frontend", "checkout", "recommendation"],
    "recommendation": ["frontend", "productcatalog"],
    "ad": ["frontend"],
    "redis-cart": ["cart"],
}


def generate_hard_dataset(
    n_fault_per_type: int = 5,
    n_load_per_type: int = 5,
    systems: Optional[List[str]] = None,
    seed: int = 42,
    *,
    n_per_type: Optional[int] = None,
) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
    """Generate hard/adversarial evaluation cases.

    Produces five scenario types where the distinction between FAULT and
    EXPECTED_LOAD is intentionally ambiguous:

    - **fault_during_event** (FAULT): A real fault occurs on top of a
      legitimate load spike. Metrics show both load and fault signatures.
    - **capacity_exceeded_load** (EXPECTED_LOAD): An extreme load spike
      (5-10x) causes error-rate increases and latency degradation purely
      from capacity exhaustion, not from a fault.
    - **gradual_fault** (FAULT): A fault that ramps up slowly (like a
      memory leak) across the full window, mimicking gradual load.
    - **correlated_fault** (FAULT): A fault in one service propagates to
      2-3 neighbors with attenuated severity, resembling a load spike
      that affects multiple services.
    - **partial_load** (EXPECTED_LOAD): A load spike that affects only
      3-4 services, overlapping with the localised-impact pattern of
      faults.

    Args:
        n_fault_per_type: Number of cases per fault scenario type.
        n_load_per_type: Number of cases per load scenario type.
        systems: System names to cycle through.
        seed: Random seed for reproducibility.
        n_per_type: Deprecated — if provided, used as both
            *n_fault_per_type* and *n_load_per_type* for backward
            compatibility.

    Returns:
        Tuple of (fault_cases, load_cases).
    """
    # Backward compatibility
    if n_per_type is not None:
        n_fault_per_type = n_per_type
        n_load_per_type = n_per_type

    if systems is None:
        systems = ["online-boutique"]

    rng = np.random.default_rng(seed)
    fault_gen = FaultGenerator(seed=seed)
    load_gen = SyntheticMetricsGenerator(seed=seed + 1)

    fault_cases: List[AnomalyCase] = []
    load_cases: List[AnomalyCase] = []

    # --- (a) FAULT_DURING_EVENT: fault injected on top of load spike ---
    for i in range(n_fault_per_type):
        system = systems[i % len(systems)]
        case_seed_load = seed * _CASE_SEED_MULT + i
        case_seed_fault = seed * _CASE_SEED_MULT + 1000 + i
        # Generate a load spike first
        services, context = load_gen.generate_load_spike_metrics(
            system=system, case_seed=case_seed_load,
        )
        # Pick a random service (not loadgenerator) to inject a fault
        eligible = [s for s in services if s.service_name != "loadgenerator"]
        fault_svc = eligible[rng.integers(len(eligible))]
        fault_type = str(rng.choice(fault_gen.FAULT_TYPES))
        # Inject fault at a random point during the spike
        n_ts = len(fault_svc.metrics)
        fault_start = int(rng.integers(n_ts // 4, 3 * n_ts // 4))
        fault_svc.metrics = fault_gen.inject_fault(
            fault_svc.metrics, fault_type, fault_start,
        )
        fault_cases.append(
            AnomalyCase(
                case_id=f"hard_fault_during_event_{i:04d}",
                system=system,
                label="FAULT",
                services=services,
                context=context,  # event IS happening
                fault_service=fault_svc.service_name,
                fault_type=fault_type,
            )
        )

    # --- (b) CAPACITY_EXCEEDED_LOAD: extreme load causes errors ---
    for i in range(n_load_per_type):
        system = systems[i % len(systems)]
        case_seed_b = seed * _CASE_SEED_MULT + 2000 + i
        high_mult = float(rng.uniform(5.0, 10.0))
        services, context = load_gen.generate_load_spike_metrics(
            system=system, load_multiplier=high_mult, case_seed=case_seed_b,
        )
        # Add load-correlated error increases (0.05-0.15) during the spike
        spike_start = context.get("spike_start", 0)
        spike_end = context.get("spike_end", len(services[0].metrics))
        for svc in services:
            df = svc.metrics
            n_ts = len(df)
            actual_end = min(spike_end, n_ts)
            spike_sl = slice(spike_start, actual_end)
            # Moderate error increase correlated with load envelope
            actual_len = len(df.loc[spike_sl, "error_rate"])
            if actual_len > 0:
                err_increase = rng.uniform(0.05, 0.15) * np.ones(actual_len)
                err_increase *= np.linspace(0.3, 1.0, actual_len)  # ramp with load
                df.loc[spike_sl, "error_rate"] = np.clip(
                    df.loc[spike_sl, "error_rate"].values + err_increase, 0, 1,
                )
        load_cases.append(
            AnomalyCase(
                case_id=f"hard_capacity_exceeded_{i:04d}",
                system=system,
                label="EXPECTED_LOAD",
                services=services,
                context=context,
            )
        )

    # --- (c) GRADUAL_FAULT: slow ramp-up over full window ---
    for i in range(n_fault_per_type):
        system = systems[i % len(systems)]
        case_seed_c = seed * _CASE_SEED_MULT + 3000 + i
        services, fault_service, fault_type = fault_gen.generate_fault_metrics(
            system=system, case_seed=case_seed_c,
        )
        # Replace the sudden fault with a gradual ramp over the full window
        for svc in services:
            if svc.service_name == fault_service:
                df = svc.metrics
                n_ts = len(df)
                ramp = np.linspace(0, 1, n_ts)
                # Gradually increase error_rate
                df["error_rate"] = np.clip(
                    df["error_rate"].values + ramp * rng.uniform(0.1, 0.4),
                    0, 1,
                )
                # Gradually increase latency
                df["latency"] = np.clip(
                    df["latency"].values + ramp * rng.uniform(50, 300),
                    0, None,
                )
                # Gradually increase CPU
                df["cpu_usage"] = np.clip(
                    df["cpu_usage"].values + ramp * rng.uniform(10, 40),
                    0, 100,
                )
        fault_cases.append(
            AnomalyCase(
                case_id=f"hard_gradual_fault_{i:04d}",
                system=system,
                label="FAULT",
                services=services,
                context={},
                fault_service=fault_service,
                fault_type=fault_type,
            )
        )

    # --- (e) CORRELATED_FAULT: fault propagates to neighbor services ---
    for i in range(n_fault_per_type):
        system = systems[i % len(systems)]
        case_seed_e = seed * _CASE_SEED_MULT + 4000 + i
        services, fault_service, fault_type = fault_gen.generate_fault_metrics(
            system=system, case_seed=case_seed_e,
        )
        # Pick 2-3 neighbor services and inject attenuated faults
        neighbors = _SERVICE_ADJACENCY.get(fault_service, [])
        svc_map = {s.service_name: s for s in services}
        n_cascade = min(rng.integers(2, 4), len(neighbors))
        if n_cascade > 0 and neighbors:
            cascade_targets = list(
                rng.choice(neighbors, size=n_cascade, replace=False)
            )
            for target_name in cascade_targets:
                if target_name not in svc_map:
                    continue
                target_svc = svc_map[target_name]
                df = target_svc.metrics
                n_ts = len(df)
                # Attenuated severity: 50-80% of what the primary service got
                attenuation = rng.uniform(0.5, 0.8)
                fault_start = n_ts // 3
                fault_sl = slice(fault_start, n_ts)
                fault_len = n_ts - fault_start
                # Inject attenuated error/latency increases
                df.loc[fault_sl, "error_rate"] = np.clip(
                    df.loc[fault_sl, "error_rate"].values
                    + rng.uniform(0.05, 0.25) * attenuation,
                    0, 1,
                )
                df.loc[fault_sl, "latency"] = np.clip(
                    df.loc[fault_sl, "latency"].values
                    + rng.uniform(50, 200, fault_len) * attenuation,
                    0, None,
                )
                df.loc[fault_sl, "cpu_usage"] = np.clip(
                    df.loc[fault_sl, "cpu_usage"].values
                    + rng.uniform(5, 20, fault_len) * attenuation,
                    0, 100,
                )
        fault_cases.append(
            AnomalyCase(
                case_id=f"hard_correlated_fault_{i:04d}",
                system=system,
                label="FAULT",
                services=services,
                context={},
                fault_service=fault_service,
                fault_type=fault_type,
            )
        )

    # --- (d) PARTIAL_LOAD: spike affects only 3-4 services ---
    for i in range(n_load_per_type):
        system = systems[i % len(systems)]
        case_seed_d = seed * _CASE_SEED_MULT + 5000 + i
        services, context = load_gen.generate_load_spike_metrics(
            system=system, case_seed=case_seed_d,
        )
        # Reset most services to baseline (keep only 3-4 affected)
        n_affected = rng.integers(
            min(3, len(services)), min(5, len(services)) + 1,
        )
        affected_indices = set(
            rng.choice(len(services), size=n_affected, replace=False)
        )
        for idx, svc in enumerate(services):
            if idx not in affected_indices:
                # Regenerate as normal baseline metrics
                svc.metrics = load_gen.generate_base_metrics(svc.service_name)
        load_cases.append(
            AnomalyCase(
                case_id=f"hard_partial_load_{i:04d}",
                system=system,
                label="EXPECTED_LOAD",
                services=services,
                context=context,
            )
        )

    logger.info(
        "Hard dataset: %d fault cases, %d load cases "
        "(n_fault_per_type=%d, n_load_per_type=%d)",
        len(fault_cases), len(load_cases), n_fault_per_type, n_load_per_type,
    )
    return fault_cases, load_cases
