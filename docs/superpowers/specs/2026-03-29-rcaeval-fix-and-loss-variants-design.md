# Design: RCAEval Pre-Injection Split + Loss Variant Ablation

**Date:** 2026-03-29
**Status:** Draft
**Goal:** Fix two critical research issues: (1) CAAA's 24% FP rate from the unknown_penalty making it lose to tree baselines, and (2) RCAEval evaluation showing 100% accuracy across all variants due to real-vs-synthetic distribution mismatch.

---

## Problem 1: CAAA Loses to Tree Baselines on Synthetic Data

**Current results:** LightGBM 93.8% > Full CAAA 86.6%. CAAA has 24.3% FP rate vs trees' 5%.

**Root cause:** The `unknown_penalty` in `ContextConsistencyLoss` (losses.py:120-121) computes:
```python
unknown_context = (1.0 - event_active) * (1.0 - context_confidence)
unknown_penalty = torch.mean(unknown_context * load_prob)
```

For the 30% of legitimate load spikes with empty context: `unknown_context = 1.0`, so the penalty = `load_prob`. This directly pushes the model to predict FAULT for ALL no-context cases, including correct ones. At `unknown_weight=0.5`, this overwhelms the classification loss.

Trees bypass this entirely — they learn metric thresholds without any loss-function bias.

## Problem 2: RCAEval Evaluation is Invalid

**Current results:** Every variant (including Context Only, Rule-Based, Statistical Only) scores 100.0% accuracy on all 9 dataset/system combos.

**Root cause:** RCAEval provides only FAULT cases. EXPECTED_LOAD cases are synthetically generated with completely different distributions:

| Feature | Real (FAULT) | Synthetic (LOAD) |
|---------|-------------|-----------------|
| memory_usage | ~10^8 (bytes) | ~20 (percentage) |
| network_in/out | 0 (not collected) | 500-8000 |
| sequence length | 4201 rows | 60 rows |
| services | 14 (with "service" suffix) | 12 (short names) |
| timestamps | epoch seconds (~1.685B) | integers 0-59 |

A single threshold on `mean_memory_usage` achieves 100%. The model learns "real vs synthetic data source" not "fault vs load."

**Additional context:** RCAEval is a root-cause-localization benchmark (AC@k metrics), not a fault-detection benchmark. CAAA is using it for a different task than intended.

---

## Fix 1: RCAEval Pre-Injection Split

Each RCAEval case contains `inject_time.txt` with a Unix timestamp marking when fault injection started. The data before that timestamp is genuine normal operation from the same system.

**Approach:** Split each case at inject_time:
- **Pre-injection slice** (timestamps < inject_time) → labeled NORMAL
- **Post-injection slice** (timestamps >= inject_time) → labeled FAULT

Both halves share identical metric scales, service names, timestamp formats, and noise characteristics. No distribution mismatch.

### Implementation

**File: `src/data_loader/rcaeval_loader.py`**

Add `load_dataset_split()` method:
- For each case directory, read the metrics CSV and `inject_time.txt`
- Split the DataFrame at the inject_time timestamp
- Return two `AnomalyCase` objects:
  - FAULT case: post-injection metrics, empty context `{}`, label="FAULT"
  - NORMAL case: pre-injection metrics, context `{"event_type": "normal_operation", "context_confidence": 0.8}`, label="EXPECTED_LOAD"
- Both cases get the same `system`, `fault_service`, `fault_type` metadata
- The NORMAL case's `fault_service` and `fault_type` are set to `None`

**File: `src/data_loader/dataset.py`**

Modify `generate_rcaeval_dataset()`:
- Add `split_mode: str = "pre_injection"` parameter (default: new behavior)
- When `split_mode="pre_injection"`: use `load_dataset_split()` instead of generating synthetic loads
- When `split_mode="synthetic"`: keep current behavior (backward compatible)
- Remove `n_load_per_fault` parameter when using split mode (always 1:1)

**Sequence length handling:** Pre-injection is ~2100 rows, post-injection ~2101 rows. The feature extractor (`extractors.py`) computes statistics (means, stds, correlations) that work on any length. PELT change-point detection and spectral features benefit from longer sequences. No extractor changes needed.

### Expected Outcomes

- Accuracy will NOT be 100% — both classes come from the same data source
- Context features (event_active, context_confidence) will differentiate: NORMAL cases have context, FAULT cases don't
- Different systems will show different difficulty (train-ticket with 64 services harder than online-boutique with 12)
- Tree baselines will still perform well (they can detect fault-induced metric changes) but won't be trivially perfect

---

## Fix 2: Loss Variant Ablation

Three loss configurations, selectable via `--loss-variant` CLI arg:

### Variant A: `clamp` (clamp-only)

Remove the `unknown_penalty` entirely. Keep only the `clamp(min=0.3)` fix on the consistency loss weighting:

```python
conf_weight = torch.clamp(context_confidence, min=0.3)
consistency_loss = torch.mean(conf_weight * per_sample_penalty)
# No unknown_penalty term
total_loss = cls_loss + alpha * consistency_loss + beta * calibration_loss
```

**Rationale:** The clamp alone ensures gradients flow for low-confidence cases. No FAULT-bias on no-context cases.

### Variant B: `gated` (default, recommended)

Apply the unknown_penalty only when the model is highly confident about EXPECTED_LOAD despite no context:

```python
# Only penalize when model is very confident about LOAD without context
gated_load_prob = torch.clamp(load_prob - 0.7, min=0.0)
unknown_penalty = torch.mean(unknown_context * gated_load_prob)
total_loss = cls_loss + alpha * consistency_loss + beta * calibration_loss + 0.2 * unknown_penalty
```

**Rationale:** Below 0.7 load_prob, no penalty fires — legitimate no-context loads aren't pushed toward FAULT. Above 0.7, the model is confidently predicting LOAD without context support, which is suspicious enough to penalize.

### Variant C: `full` (current behavior)

Current implementation: `unknown_weight=0.5`, no gate. Kept for comparison.

### Implementation

**File: `src/training/losses.py`**

Add `loss_variant: str = "gated"` parameter to `ContextConsistencyLoss.__init__()`. In `forward()`, branch on the variant:

```python
if self.loss_variant == "clamp":
    unknown_penalty_val = torch.tensor(0.0, device=logits.device)
elif self.loss_variant == "gated":
    gated_load_prob = torch.clamp(load_prob - 0.7, min=0.0)
    unknown_penalty_val = torch.mean(unknown_context * gated_load_prob)
else:  # "full" — current behavior
    unknown_penalty_val = torch.mean(unknown_context * load_prob)
```

**File: `src/training/trainer.py`**

Add `loss_variant: str = "gated"` parameter, pass to `ContextConsistencyLoss`.

**File: `scripts/ablation.py`**

Add `--loss-variant` CLI arg. Add two new ablation variants:
- "CAAA (clamp loss)" — uses loss_variant="clamp"
- "CAAA (gated loss)" — uses loss_variant="gated"

The existing "Full CAAA" uses "gated" (new default). Add "CAAA (full penalty)" with loss_variant="full" for comparison.

**File: `scripts/train.py`**

Add `--loss-variant` CLI arg, pass through.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/data_loader/rcaeval_loader.py` | Add `load_dataset_split()` method |
| `src/data_loader/dataset.py` | Modify `generate_rcaeval_dataset()` to use split mode |
| `src/training/losses.py` | Add `loss_variant` parameter with clamp/gated/full branches |
| `src/training/trainer.py` | Thread `loss_variant` through |
| `scripts/ablation.py` | Add `--loss-variant` CLI arg + new ablation variants |
| `scripts/train.py` | Add `--loss-variant` CLI arg |

## Backward Compatibility

- `generate_rcaeval_dataset(split_mode="synthetic")` preserves current behavior
- `loss_variant="gated"` becomes the new default — existing tests that don't pass loss_variant get the improved behavior
- Existing 137 tests should pass (they use default parameters)

## Verification

1. `python -m pytest tests/ -v --tb=short` — all tests pass
2. Quick synthetic ablation: `python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 1`
   - Full CAAA (gated) accuracy ~90%+, FP rate < 15%
   - Full CAAA (clamp) similar or slightly lower
   - Full CAAA (full penalty) ~86%, FP rate ~24% (current behavior, for comparison)
3. RCAEval ablation: `python scripts/ablation.py --data rcaeval --dataset RE1 --system online-boutique --epochs 30 --n-runs 3`
   - NOT 100% for all variants
   - Meaningful differentiation between variants
   - Full CAAA outperforms No Context Features
