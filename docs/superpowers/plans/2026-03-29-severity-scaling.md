# Full Severity Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make synthetic faults genuinely ambiguous by scaling ALL metric injections by severity, adding disguised faults that mimic load spike patterns, and fixing the ContextConsistencyLoss to learn from low-confidence cases.

**Architecture:** Three changes: (1) fault_generator.py gets full severity scaling + new `generate_disguised_fault()` method, (2) dataset.py routes low-severity faults to disguised generation, (3) losses.py gets confidence clamping + unknown context penalty.

**Tech Stack:** Python, NumPy, PyTorch, existing CAAA framework

---

### File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/data_loader/fault_generator.py` | Modify | Scale all metric injections by `sev`, add `_scale_mult()`, add `generate_disguised_fault()` |
| `src/data_loader/dataset.py` | Modify | Route severity="low" to disguised fault generation |
| `src/training/losses.py` | Modify | Clamp confidence weighting, add unknown context penalty |
| `src/training/trainer.py` | Modify | Pass `unknown_weight` through to ContextConsistencyLoss |
| `scripts/ablation.py` | Modify | Add `--unknown-weight` CLI arg |
| `scripts/train.py` | Modify | Add `--unknown-weight` CLI arg |

---

### Task 1: Scale ALL metric injections by severity in `_inject_fault()`

Currently only error_rate is scaled by `sev`. CPU, latency, memory, network, and request_rate injections are at full magnitude regardless of severity.

**Files:**
- Modify: `src/data_loader/fault_generator.py:89-308`

- [ ] **Step 1: Add `_scale_mult` helper method**

Add after the `_ar1_signal` method (after line 109):

```python
    def _scale_mult(self, factor: float, sev: float) -> float:
        """Scale a multiplicative degradation factor by severity.

        Interpolates between 1.0 (no change) and the original factor.
        At sev=1.0: returns factor. At sev=0.0: returns 1.0.
        """
        return 1.0 - (1.0 - factor) * sev
```

- [ ] **Step 2: Scale additive metric injections (AR1 signals and ramps)**

In `_inject_fault()`, scale every AR1 signal and ramp by `sev`. Replace each occurrence:

**cpu_hog** (line 150): `self._ar1_signal(fault_len, 30, 60)` → `self._ar1_signal(fault_len, 30 * sev, 60 * sev)`

**memory_leak** (line 163): `np.linspace(0, self.rng.uniform(30, 55), fault_len)` → `np.linspace(0, self.rng.uniform(30, 55) * sev, fault_len)`

**network_delay** (line 177): `self._ar1_signal(fault_len, 200, 800)` → `self._ar1_signal(fault_len, 200 * sev, 800 * sev)`

**disk_io** (lines 201, 207):
- `self._ar1_signal(fault_len, 100, 500)` → `self._ar1_signal(fault_len, 100 * sev, 500 * sev)`
- `self._ar1_signal(fault_len, 10, 30)` → `self._ar1_signal(fault_len, 10 * sev, 30 * sev)`

**dns_failure** (line 233): `self._ar1_signal(fault_len, 500, 2000)` → `self._ar1_signal(fault_len, 500 * sev, 2000 * sev)`

**connection_pool_exhaustion** (line 248): `self._ar1_signal(fault_len, 300, 1000)` → `self._ar1_signal(fault_len, 300 * sev, 1000 * sev)`

**thread_leak** (lines 263, 270):
- `np.linspace(0, self.rng.uniform(20, 50), fault_len)` → `np.linspace(0, self.rng.uniform(20, 50) * sev, fault_len)`
- `self._ar1_signal(fault_len, 50, 300)` → `self._ar1_signal(fault_len, 50 * sev, 300 * sev)`

**config_error** (line 288): `self._ar1_signal(fault_len, 50, 200)` → `self._ar1_signal(fault_len, 50 * sev, 200 * sev)`

**dependency_failure** (line 301): `self._ar1_signal(fault_len, 200, 600)` → `self._ar1_signal(fault_len, 200 * sev, 600 * sev)`

- [ ] **Step 3: Scale multiplicative metric injections**

Replace hardcoded multipliers with `_scale_mult`:

**packet_loss** (lines 188-192):
- `df.loc[fault_slice, "network_in"].values * 0.1` → `df.loc[fault_slice, "network_in"].values * self._scale_mult(0.1, sev)`
- `df.loc[fault_slice, "network_out"].values * 0.1` → `df.loc[fault_slice, "network_out"].values * self._scale_mult(0.1, sev)`

**pod_failure** (lines 219, 227):
- `df.loc[fault_slice, "request_rate"].values * 0.05` → `df.loc[fault_slice, "request_rate"].values * self._scale_mult(0.05, sev)`
- `df.loc[fault_slice, "cpu_usage"].values * 0.1` → `df.loc[fault_slice, "cpu_usage"].values * self._scale_mult(0.1, sev)`

**dns_failure** (line 242):
- `df.loc[fault_slice, "network_in"].values * 0.3` → `df.loc[fault_slice, "network_in"].values * self._scale_mult(0.3, sev)`

**connection_pool_exhaustion** (line 258):
- `df.loc[fault_slice, "request_rate"].values * 0.4` → `df.loc[fault_slice, "request_rate"].values * self._scale_mult(0.4, sev)`

**dependency_failure** (line 305):
- `df.loc[fault_slice, "request_rate"].values * 0.5` → `df.loc[fault_slice, "request_rate"].values * self._scale_mult(0.5, sev)`

- [ ] **Step 4: Run existing tests**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_data_loader.py -v --tb=short`
Expected: All data loader tests pass (they use default severity="high" which produces sev=1.0, unchanged behavior).

- [ ] **Step 5: Commit**

```bash
git add src/data_loader/fault_generator.py
git commit -m "feat: scale all metric injections by severity factor

Previously only error_rate was scaled. Now CPU, latency, memory,
network, and request_rate injections are all scaled by the severity
factor (low=0.05, medium=0.3, high=1.0). Multiplicative changes
use interpolation: scaled = 1.0 - (1.0 - factor) * sev."
```

---

### Task 2: Add `generate_disguised_fault()` method

New method on `FaultGenerator` that produces faults metrically indistinguishable from load spikes.

**Files:**
- Modify: `src/data_loader/fault_generator.py` (add method after `generate_fault_metrics`)

- [ ] **Step 1: Add the method**

Add after `generate_fault_metrics()` (after line 407):

```python
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

        results: List[ServiceMetrics] = []
        for name in self.SERVICE_NAMES[: self.n_services]:
            df = self._base_metrics(name)

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
```

- [ ] **Step 2: Run data loader tests**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_data_loader.py -v --tb=short`
Expected: All pass (new method is not called by existing tests).

- [ ] **Step 3: Commit**

```bash
git add src/data_loader/fault_generator.py
git commit -m "feat: add generate_disguised_fault() for load-mimicking faults

Disguised faults use the same envelope shape, spread, and magnitude
as load spikes, with a small fault perturbation on one service.
Without context, these are metrically indistinguishable from loads."
```

---

### Task 3: Route low-severity faults to disguised generation in dataset.py

**Files:**
- Modify: `src/data_loader/dataset.py:59-104`

- [ ] **Step 1: Update fault generation loop**

In `generate_combined_dataset()`, change the low-severity fault generation (lines ~76-78) to call `generate_disguised_fault()` instead of `generate_fault_metrics()`:

Replace:
```python
        services, fault_service, fault_type = fault_gen.generate_fault_metrics(
            system=system, case_seed=case_seed, severity=severity,
        )
        fault_context = {}
        if rng.random() < 0.3:
            fault_context["recent_deployment"] = True
        # 30% of fault cases get a fake context with event_type to prevent
        # event_active from being a perfect proxy for the label.
        if rng.random() < 0.30:
            fake_event = str(rng.choice([
                "flash_sale", "marketing_campaign", "scheduled_batch",
            ]))
            fault_context["event_type"] = fake_event
            fault_context["event_name"] = f"{fake_event}_event"
            fault_context["load_multiplier"] = float(
                rng.uniform(1.2, 2.5)
            )
```

With:
```python
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
            # 30% of fault cases get a fake context with event_type to prevent
            # event_active from being a perfect proxy for the label.
            if rng.random() < 0.30:
                fake_event = str(rng.choice([
                    "flash_sale", "marketing_campaign", "scheduled_batch",
                ]))
                fault_context["event_type"] = fake_event
                fault_context["event_name"] = f"{fake_event}_event"
                fault_context["load_multiplier"] = float(
                    rng.uniform(1.2, 2.5)
                )
```

- [ ] **Step 2: Run data loader tests**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_data_loader.py -v --tb=short`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add src/data_loader/dataset.py
git commit -m "feat: route low-severity faults to disguised generation

Low-severity faults (35% of cases) now use generate_disguised_fault()
which produces load-like metric patterns with empty context."
```

---

### Task 4: Fix ContextConsistencyLoss — clamp + unknown context penalty

**Files:**
- Modify: `src/training/losses.py:62-132`

- [ ] **Step 1: Add `unknown_weight` parameter to `__init__`**

Replace the `__init__` (lines 62-72):

```python
    def __init__(self, alpha: float = 0.3, beta: float = 0.1, unknown_weight: float = 0.5) -> None:
        """Initializes the ContextConsistencyLoss.

        Args:
            alpha: Weight for context consistency loss.
            beta: Weight for confidence calibration loss.
            unknown_weight: Weight for the unknown-context penalty that
                penalizes EXPECTED_LOAD predictions when no event is active
                and context confidence is low.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.unknown_weight = unknown_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
```

- [ ] **Step 2: Fix the confidence weighting and add unknown context penalty**

Replace the consistency loss computation (lines 109-114):

```python
        # Weight by clamped context_confidence so that low-confidence cases
        # (disguised faults, unscheduled loads) still produce learning signal.
        # Floor of 0.3 ensures disguised faults contribute 30% of the penalty
        # that high-confidence cases do.
        per_sample_penalty = penalty_when_event + penalty_when_no_event
        conf_weight = torch.clamp(context_confidence, min=0.3)
        consistency_loss = torch.mean(conf_weight * per_sample_penalty)

        # Unknown context penalty: explicitly penalize predicting EXPECTED_LOAD
        # when no event is active and confidence is low.  This gives the model
        # a direct gradient signal for disguised faults.
        unknown_context = (1.0 - event_active) * (1.0 - context_confidence)
        unknown_penalty = torch.mean(unknown_context * load_prob)
```

- [ ] **Step 3: Update the combined loss line**

Replace line 124:

```python
        total_loss = (
            cls_loss
            + self.alpha * consistency_loss
            + self.beta * calibration_loss
            + self.unknown_weight * unknown_penalty
        )
```

- [ ] **Step 4: Add unknown_penalty to components dict**

Replace lines 126-130:

```python
        components = {
            "cls_loss": cls_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "calibration_loss": calibration_loss.item(),
            "unknown_penalty": unknown_penalty.item(),
        }
```

- [ ] **Step 5: Run loss-related tests**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_integration.py -k "ContextConsistency" -v --tb=short`
Expected: All pass — the tests check loss computation and gradient flow, both of which are preserved.

- [ ] **Step 6: Commit**

```bash
git add src/training/losses.py
git commit -m "fix: clamp confidence weighting + add unknown context penalty

The confidence**2 weighting zeroed out gradients for low-confidence
cases (disguised faults). Now uses clamp(min=0.3) + an explicit
penalty for predicting EXPECTED_LOAD when no event is active."
```

---

### Task 5: Thread `unknown_weight` through trainer and CLI

**Files:**
- Modify: `src/training/trainer.py:37-84`
- Modify: `scripts/ablation.py` (argparse section)
- Modify: `scripts/train.py` (argparse section)

- [ ] **Step 1: Add `unknown_weight` to CAAATrainer.__init__**

In `trainer.py`, add `unknown_weight: float = 0.5` parameter to `__init__` (line ~39). Then pass it when creating the loss (line ~84):

Replace:
```python
            self.criterion = ContextConsistencyLoss()
```
With:
```python
            self.criterion = ContextConsistencyLoss(unknown_weight=unknown_weight)
```

Store it: add `self.unknown_weight = unknown_weight` in the constructor body.

- [ ] **Step 2: Add `--unknown-weight` CLI arg to ablation.py**

After the `--calibration` argument (line ~302), add:

```python
    parser.add_argument("--unknown-weight", type=float, default=0.5,
                        help="Weight for unknown-context penalty in ContextConsistencyLoss")
```

Then in every `CAAATrainer(...)` call in ablation.py, pass `unknown_weight=args.unknown_weight`. There are multiple trainer instantiations — search for `CAAATrainer(` and add the parameter to each.

- [ ] **Step 3: Add `--unknown-weight` CLI arg to train.py**

Same pattern as ablation.py: add the argparse line and pass to CAAATrainer.

- [ ] **Step 4: Run full test suite**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/ -v --tb=short`
Expected: All 137 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/trainer.py scripts/ablation.py scripts/train.py
git commit -m "feat: thread unknown_weight through trainer and CLI

Adds --unknown-weight CLI arg (default 0.5) to ablation.py and
train.py, passed through CAAATrainer to ContextConsistencyLoss."
```

---

### Task 6: Smoke test — verify differentiation

**Files:** None (verification only)

- [ ] **Step 1: Run quick ablation**

```bash
OMP_NUM_THREADS=1 python scripts/ablation.py \
    --n-fault 100 --n-load 100 --epochs 30 --n-runs 3
```

Expected:
- Output file: `outputs/results/ablation_results_synthetic.csv`
- Full CAAA accuracy: ~85-92% (NOT 99%)
- Baseline RF accuracy: ~78-85% (NOT 100%)
- Per-difficulty breakdown shows differentiation: hard tier < medium < easy
- Full CAAA outperforms No Context Features on hard tier by >10%

- [ ] **Step 2: If differentiation is insufficient, adjust unknown_weight**

Run sweep:
```bash
for w in 0.1 0.3 0.5 0.7 1.0; do
    echo "=== unknown_weight=$w ==="
    OMP_NUM_THREADS=1 python scripts/ablation.py \
        --n-fault 50 --n-load 50 --epochs 30 --n-runs 1 \
        --unknown-weight $w 2>&1 | grep -A 5 "PER-DIFFICULTY"
done
```

Pick the weight that maximizes hard-tier fault recall while keeping no-context load FP rate < 20%.

- [ ] **Step 3: Final commit with tuned default**

If the default (0.5) isn't optimal, update it in `losses.py` and commit.

---

### Task 7: Full test suite final verification

- [ ] **Step 1: Run complete test suite**

```bash
OMP_NUM_THREADS=1 python -m pytest tests/ -v --tb=short
```
Expected: 137 passed.

- [ ] **Step 2: Run demo as sanity check**

```bash
OMP_NUM_THREADS=1 python scripts/demo.py --n-fault 30 --n-load 30 --epochs 30
```
Expected: Accuracy ~85-95% (not 100%), runs without error.
