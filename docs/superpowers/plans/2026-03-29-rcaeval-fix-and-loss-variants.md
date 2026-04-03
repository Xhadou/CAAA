# RCAEval Pre-Injection Split + Loss Variant Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two research issues: (1) RCAEval evaluation showing trivial 100% accuracy by switching to pre-injection split, and (2) CAAA's 24% FP rate by adding loss variant ablation (clamp/gated/full).

**Architecture:** Two independent changes: rcaeval_loader.py gets a `load_dataset_split()` method that splits each case at inject_time into NORMAL + FAULT halves from the same real data; losses.py gets a `loss_variant` parameter controlling three unknown_penalty behaviors.

**Tech Stack:** Python, NumPy, Pandas, PyTorch, existing CAAA framework

---

### File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/data_loader/rcaeval_loader.py` | Modify | Add `load_dataset_split()` method |
| `src/data_loader/dataset.py` | Modify | Add `split_mode` parameter to `generate_rcaeval_dataset()` |
| `src/training/losses.py` | Modify | Add `loss_variant` parameter with clamp/gated/full branches |
| `src/training/trainer.py` | Modify | Thread `loss_variant` through to loss |
| `scripts/ablation.py` | Modify | Add `--loss-variant` CLI arg + new ablation variants |
| `scripts/train.py` | Modify | Add `--loss-variant` CLI arg |

---

### Task 1: Add `load_dataset_split()` to RCAEvalLoader

**Files:**
- Modify: `src/data_loader/rcaeval_loader.py:239-339`

- [ ] **Step 1: Add the `load_dataset_split` method**

Add after the `load_dataset` method (after line 339), before the `load_rcaeval` convenience function:

```python
    def load_dataset_split(
        self,
        dataset: str = "RE1",
        system: str = "online-boutique",
        fault_types: Optional[List[str]] = None,
    ) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
        """Load RCAEval cases split at injection time into FAULT + NORMAL.

        Each case is split at ``inject_time.txt``: metrics before injection
        become a NORMAL (EXPECTED_LOAD) case, metrics after become a FAULT
        case.  Both halves share identical distributions, eliminating the
        real-vs-synthetic mismatch.

        Cases without ``inject_time.txt`` are returned as FAULT-only.

        Args:
            dataset: ``"RE1"``, ``"RE2"``, or ``"RE3"``.
            system: Microservice system name.
            fault_types: Filter by fault types. ``None`` returns all.

        Returns:
            Tuple of (fault_cases, normal_cases).
        """
        dataset_path = self.data_dir / dataset / system
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        fault_cases: List[AnomalyCase] = []
        normal_cases: List[AnomalyCase] = []

        # Reuse the same directory discovery logic as load_dataset
        case_dirs: List[Path] = []
        for child in sorted(dataset_path.iterdir()):
            if not child.is_dir():
                continue
            sub_dirs = [d for d in sorted(child.iterdir()) if d.is_dir()]
            has_own_csv = any(child.glob("*.csv"))
            if sub_dirs and not has_own_csv:
                for svc_fault_dir in sub_dirs:
                    instance_dirs = [d for d in sorted(svc_fault_dir.iterdir()) if d.is_dir()]
                    if instance_dirs:
                        for inst_dir in instance_dirs:
                            case_dirs.append(inst_dir)
                    else:
                        case_dirs.append(svc_fault_dir)
            else:
                case_dirs.append(child)

        for case_dir in case_dirs:
            if case_dir.parent.parent != dataset_path:
                case_name = case_dir.parent.name
            else:
                case_name = case_dir.name

            info = self.parse_case_name(case_name)
            if info["system"] == "unknown":
                info["system"] = system
            if fault_types and info["fault_type"] not in fault_types:
                continue

            metrics = self.load_metrics(case_dir)
            if metrics.empty:
                logger.warning("Empty metrics for case %s — skipping", case_dir.name)
                continue

            # Read inject_time
            inject_time = None
            inject_file = case_dir / "inject_time.txt"
            if inject_file.exists():
                try:
                    inject_time = int(inject_file.read_text().strip())
                except (ValueError, OSError):
                    pass

            case_id_base = (
                f"{case_name}_{case_dir.name}"
                if case_dir.parent.parent != dataset_path
                else case_dir.name
            )

            if inject_time is None:
                # No inject_time — return as fault-only (same as load_dataset)
                services = self.parse_wide_format(metrics)
                fault_cases.append(AnomalyCase(
                    case_id=case_id_base,
                    system=info["system"],
                    label="FAULT",
                    services=services,
                    fault_service=info["service"],
                    fault_type=info["fault_type"],
                ))
                continue

            # Find the timestamp column
            ts_col = None
            for candidate in ("timestamp", "time", "ts"):
                if candidate in metrics.columns:
                    ts_col = candidate
                    break

            if ts_col is None:
                # No timestamp column — fall back to fault-only
                services = self.parse_wide_format(metrics)
                fault_cases.append(AnomalyCase(
                    case_id=case_id_base,
                    system=info["system"],
                    label="FAULT",
                    services=services,
                    fault_service=info["service"],
                    fault_type=info["fault_type"],
                ))
                continue

            # Split at inject_time
            pre_mask = metrics[ts_col] < inject_time
            post_mask = metrics[ts_col] >= inject_time

            pre_df = metrics[pre_mask].reset_index(drop=True)
            post_df = metrics[post_mask].reset_index(drop=True)

            if len(pre_df) < 10 or len(post_df) < 10:
                # Too few rows in one half — skip split, use as fault-only
                services = self.parse_wide_format(metrics)
                fault_cases.append(AnomalyCase(
                    case_id=case_id_base,
                    system=info["system"],
                    label="FAULT",
                    services=services,
                    fault_service=info["service"],
                    fault_type=info["fault_type"],
                ))
                continue

            # FAULT case: post-injection
            fault_services = self.parse_wide_format(post_df)
            fault_cases.append(AnomalyCase(
                case_id=f"{case_id_base}_fault",
                system=info["system"],
                label="FAULT",
                services=fault_services,
                context={},
                fault_service=info["service"],
                fault_type=info["fault_type"],
            ))

            # NORMAL case: pre-injection
            normal_services = self.parse_wide_format(pre_df)
            normal_cases.append(AnomalyCase(
                case_id=f"{case_id_base}_normal",
                system=info["system"],
                label="EXPECTED_LOAD",
                services=normal_services,
                context={
                    "event_type": "normal_operation",
                    "context_confidence": 0.8,
                },
            ))

        logger.info(
            "RCAEval split: %d fault + %d normal from %s/%s",
            len(fault_cases), len(normal_cases), dataset, system,
        )
        return fault_cases, normal_cases
```

Also add the `Tuple` import at the top of the file if not already present. Check line 5 — it should already have `from typing import ... List, Optional`. Add `Tuple` to that import.

- [ ] **Step 2: Run existing RCAEval tests**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_rcaeval_pipeline.py -v --tb=short`
Expected: All pass (new method not called by existing tests).

- [ ] **Step 3: Commit**

```bash
git add src/data_loader/rcaeval_loader.py
git commit -m "feat: add load_dataset_split() for pre-injection NORMAL/FAULT split

Splits each RCAEval case at inject_time into two halves from the
same real data, eliminating the distribution mismatch between real
fault traces and synthetic load data."
```

---

### Task 2: Add `split_mode` to `generate_rcaeval_dataset()`

**Files:**
- Modify: `src/data_loader/dataset.py:240-319`

- [ ] **Step 1: Update function signature and body**

Replace the `generate_rcaeval_dataset` function (lines 240-319) with:

```python
def generate_rcaeval_dataset(
    dataset: "str | List[str]" = "RE1",
    system: "str | List[str]" = "online-boutique",
    n_load_per_fault: int = 1,
    data_dir: str = "data/raw",
    seed: int = 42,
    split_mode: str = "pre_injection",
) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
    """Load FAULT cases from RCAEval and pair with EXPECTED_LOAD cases.

    Args:
        dataset: RCAEval dataset identifier(s).
        system: Microservice system(s).
        n_load_per_fault: Number of synthetic load cases per fault case
            (only used when ``split_mode="synthetic"``).
        data_dir: Path to downloaded RCAEval data.
        seed: Random seed.
        split_mode: How to generate EXPECTED_LOAD cases:
            ``"pre_injection"`` — split each case at inject_time (default).
            ``"synthetic"`` — generate synthetic load spikes (legacy).

    Returns:
        Tuple of (fault_cases, load_cases).

    Raises:
        FileNotFoundError: If no data found.
    """
    from src.data_loader.rcaeval_loader import RCAEvalLoader

    datasets = [dataset] if isinstance(dataset, str) else list(dataset)
    systems = [system] if isinstance(system, str) else list(system)

    loader = RCAEvalLoader(data_dir=data_dir)

    if split_mode == "pre_injection":
        fault_cases: List[AnomalyCase] = []
        load_cases: List[AnomalyCase] = []
        for ds in datasets:
            for sys_name in systems:
                try:
                    faults, normals = loader.load_dataset_split(
                        dataset=ds, system=sys_name,
                    )
                except FileNotFoundError:
                    logger.warning("No data at %s/%s/%s — skipping", data_dir, ds, sys_name)
                    continue
                fault_cases.extend(faults)
                load_cases.extend(normals)

        if not fault_cases:
            raise FileNotFoundError(
                f"No RCAEval data found for datasets={datasets}, systems={systems} "
                f"in {data_dir}. Download the data first."
            )

        logger.info(
            "RCAEval dataset (pre_injection split): %d faults + %d normals = %d total",
            len(fault_cases), len(load_cases), len(fault_cases) + len(load_cases),
        )
        return fault_cases, load_cases

    # Legacy synthetic mode
    fault_cases = []
    for ds in datasets:
        for sys_name in systems:
            try:
                cases = loader.load_dataset(dataset=ds, system=sys_name)
            except FileNotFoundError:
                logger.warning("No data at %s/%s/%s — skipping", data_dir, ds, sys_name)
                continue
            if cases:
                logger.info("Loaded %d fault cases from RCAEval %s/%s", len(cases), ds, sys_name)
                fault_cases.extend(cases)
            else:
                logger.warning("Empty dataset for %s/%s — skipping", ds, sys_name)

    if not fault_cases:
        raise FileNotFoundError(
            f"No RCAEval data found for datasets={datasets}, systems={systems} "
            f"in {data_dir}. Download the data first."
        )

    load_gen = SyntheticMetricsGenerator(seed=seed)
    load_cases = []
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
```

- [ ] **Step 2: Run tests**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_rcaeval_pipeline.py tests/test_data_loader.py -v --tb=short`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add src/data_loader/dataset.py
git commit -m "feat: add split_mode to generate_rcaeval_dataset

Default 'pre_injection' splits each case at inject_time for
genuine NORMAL/FAULT data. Legacy 'synthetic' mode preserved."
```

---

### Task 3: Add `loss_variant` to ContextConsistencyLoss

**Files:**
- Modify: `src/training/losses.py:62-146`

- [ ] **Step 1: Update `__init__` signature**

Replace the `__init__` (lines 62-76):

```python
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        unknown_weight: float = 0.2,
        loss_variant: str = "gated",
    ) -> None:
        """Initializes the ContextConsistencyLoss.

        Args:
            alpha: Weight for context consistency loss.
            beta: Weight for confidence calibration loss.
            unknown_weight: Weight for the unknown-context penalty.
            loss_variant: Penalty variant:
                ``"clamp"`` — no unknown penalty, clamp-only consistency.
                ``"gated"`` — penalty only when load_prob > 0.7 (default).
                ``"full"`` — penalty on all no-context cases (aggressive).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.unknown_weight = unknown_weight
        self.loss_variant = loss_variant
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
```

- [ ] **Step 2: Update the unknown_penalty computation in `forward`**

Replace lines 121-125 (the current unknown_penalty block):

```python
        # Unknown context penalty — variant-dependent
        unknown_context = (1.0 - event_active) * (1.0 - context_confidence)
        if self.loss_variant == "clamp":
            unknown_penalty = torch.tensor(0.0, device=logits.device)
        elif self.loss_variant == "gated":
            # Only penalize when model is very confident about LOAD without context
            gated_load_prob = torch.clamp(load_prob - 0.7, min=0.0)
            unknown_penalty = torch.mean(unknown_context * gated_load_prob)
        else:  # "full" — current aggressive behavior
            unknown_penalty = torch.mean(unknown_context * load_prob)
```

- [ ] **Step 3: Run loss tests**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_integration.py -k "ContextConsistency" -v --tb=short`
Expected: All pass. The default changed from `unknown_weight=0.5` to `0.2` and `loss_variant="gated"`, but the tests should still pass since they test loss computation and gradient flow, not exact values.

- [ ] **Step 4: Commit**

```bash
git add src/training/losses.py
git commit -m "feat: add loss_variant parameter (clamp/gated/full)

Gated variant (default) only penalizes LOAD predictions above 0.7
confidence when context is absent. Clamp variant removes the
unknown penalty entirely. Full preserves the original behavior."
```

---

### Task 4: Thread `loss_variant` through trainer and CLI

**Files:**
- Modify: `src/training/trainer.py:36-88`
- Modify: `scripts/ablation.py` (argparse + variant calls)
- Modify: `scripts/train.py` (argparse)

- [ ] **Step 1: Add `loss_variant` to CAAATrainer**

In `src/training/trainer.py`, update the `__init__` signature (line ~36) to add `loss_variant: str = "gated"`:

```python
    def __init__(
        self,
        model: CAAAModel,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "auto",
        use_context_loss: bool = True,
        loss_type: str = "context_consistency",
        max_grad_norm: float = 1.0,
        unknown_weight: float = 0.2,
        loss_variant: str = "gated",
    ) -> None:
```

Add `self.loss_variant = loss_variant` in the body. Update the ContextConsistencyLoss instantiation (line ~88):

```python
            self.criterion = ContextConsistencyLoss(
                unknown_weight=unknown_weight, loss_variant=loss_variant,
            )
```

- [ ] **Step 2: Add `--loss-variant` CLI arg to ablation.py**

After the `--unknown-weight` argument, add:

```python
    parser.add_argument("--loss-variant", type=str, default="gated",
                        choices=["clamp", "gated", "full"],
                        help="ContextConsistencyLoss penalty variant")
```

Update `run_caaa_variant` signature (line ~36) to accept `loss_variant="gated"`:

```python
def run_caaa_variant(
    X_train, y_train, X_test, y_test, naive_fp, epochs, batch_size, lr,
    use_context_loss=True, loss_type="context_consistency", seed=42,
    X_val=None, y_val=None, unknown_weight=0.2, loss_variant="gated",
):
```

Pass it to CAAATrainer:

```python
    trainer = CAAATrainer(
        model, learning_rate=lr, device=device,
        use_context_loss=use_context_loss,
        loss_type=loss_type,
        unknown_weight=unknown_weight,
        loss_variant=loss_variant,
    )
```

- [ ] **Step 3: Add loss variant ablation entries in ablation.py**

Add two new variants to the `variants` list (after "CAAA + Contrastive"):

```python
    variants = [
        "Full CAAA",
        "CAAA + Contrastive",
        "CAAA (clamp loss)",
        "CAAA (full penalty)",
        "No Context Features",
        ...
    ]
```

In the variant training loop, add the two new variants after the CAAA + Contrastive block:

```python
            # --- CAAA (clamp loss) ---
            print("  CAAA (clamp loss)...")
            m = run_caaa_variant(
                X_train, y_train, X_test, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr, seed=run_seed,
                X_val=X_val, y_val=y_val,
                unknown_weight=args.unknown_weight,
                loss_variant="clamp",
            )
            for k in metrics_to_track:
                all_results["CAAA (clamp loss)"][k].append(m.get(k, 0.0))

            # --- CAAA (full penalty) ---
            print("  CAAA (full penalty)...")
            m = run_caaa_variant(
                X_train, y_train, X_test, y_test, naive_fp,
                args.epochs, args.batch_size, args.lr, seed=run_seed,
                X_val=X_val, y_val=y_val,
                unknown_weight=0.5,
                loss_variant="full",
            )
            for k in metrics_to_track:
                all_results["CAAA (full penalty)"][k].append(m.get(k, 0.0))
```

Pass `loss_variant=args.loss_variant` to the existing "Full CAAA" variant call and all direct `CAAATrainer(` instantiations in the main() body (fault-type breakdown, SHAP, calibration sections).

- [ ] **Step 4: Add `--loss-variant` to train.py**

Same pattern: add the argparse arg, pass to CAAATrainer.

- [ ] **Step 5: Run full test suite**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/ -v --tb=short`
Expected: All 137 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/training/trainer.py scripts/ablation.py scripts/train.py
git commit -m "feat: thread loss_variant through trainer and CLI

Adds --loss-variant CLI arg (clamp/gated/full) and two new ablation
variants: CAAA (clamp loss) and CAAA (full penalty)."
```

---

### Task 5: Smoke test — synthetic ablation with loss variants

**Files:** None (verification only)

- [ ] **Step 1: Run quick synthetic ablation**

```bash
OMP_NUM_THREADS=1 python scripts/ablation.py \
    --n-fault 50 --n-load 50 --epochs 30 --n-runs 1
```

Expected:
- Output file: `outputs/results/ablation_results_synthetic.csv`
- "Full CAAA" (gated, default) should have FP rate < 15% (down from 24%)
- "CAAA (clamp loss)" should have similar or lower FP rate
- "CAAA (full penalty)" should have ~24% FP rate (original behavior)
- All three CAAA variants should have accuracy > 85%

- [ ] **Step 2: Verify the three loss variants differentiate**

Check the output table for the three CAAA loss rows. If gated and clamp are identical, the gating threshold (0.7) may need adjustment.

---

### Task 6: Smoke test — RCAEval with pre-injection split

**Files:** None (verification only)

- [ ] **Step 1: Run RCAEval ablation on one system**

```bash
OMP_NUM_THREADS=1 python scripts/ablation.py \
    --data rcaeval --dataset RE1 --system online-boutique \
    --epochs 30 --n-runs 1
```

Expected:
- Output file: `outputs/results/ablation_results_RE1_online-boutique.csv`
- NOT 100% accuracy for all variants
- Meaningful differentiation between variants
- No distribution mismatch (both classes from same real data)

If it still shows 100% accuracy, the pre-injection metrics may still be too easily separable from post-injection metrics (fault patterns are obvious in real data). This would be a valid result — it means real faults produce detectable metric changes — but at least it's not a data artifact.

- [ ] **Step 2: If 100% persists, verify it's genuine**

Check that both FAULT and NORMAL cases have the same:
- Number of services
- Metric scales (memory, latency, network)
- Timestamp formats
- Sequence lengths (approximately equal)

If they match, 100% accuracy on real data is a genuine result (the fault pattern is just that obvious in the metrics), not a distribution artifact.

---

### Task 7: Full test suite final verification

- [ ] **Step 1: Run complete test suite**

```bash
OMP_NUM_THREADS=1 python -m pytest tests/ -v --tb=short
```
Expected: All tests pass.

- [ ] **Step 2: Run demo as sanity check**

```bash
OMP_NUM_THREADS=1 python scripts/demo.py --n-fault 30 --n-load 30 --epochs 30
```
Expected: Runs without error.
