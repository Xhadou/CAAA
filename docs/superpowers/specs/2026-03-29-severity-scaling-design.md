# Design: Full Severity Scaling for Synthetic Fault Generation

**Date:** 2026-03-29
**Status:** Draft
**Goal:** Make tree baselines unable to achieve 100% accuracy on synthetic data by making low-severity faults genuinely ambiguous without context.

---

## Problem

Synthetic faults are trivially separable from load spikes via three structural differences:

1. **Magnitude**: Faults inject massive metric changes (CPU +30-60%, error +0.1-0.5) while loads produce proportional but different-shaped changes
2. **Locality**: Faults hit one service; loads affect all services uniformly
3. **Onset shape**: Faults are step changes at a random point; loads are gradual ramp-up/plateau/ramp-down

Current severity scaling (from the previous implementation) only reduces error_rate magnitude. CPU spikes, latency jumps, network drops, and the locality/onset patterns remain dead giveaways. Random Forest achieves 100% by learning "if only one service's CPU spiked with a step change, it's a fault."

## Design

### Three Severity Tiers

| Tier | % of faults | Difficulty | Approach | Expected tree accuracy |
|------|------------|------------|----------|----------------------|
| **High** | 30% | easy | Current behavior unchanged | ~99% |
| **Medium** | 35% | medium | All metrics scaled by sev=0.3 | ~80-85% |
| **Low** | 35% | hard | Disguised — mimics load spike pattern | ~60-70% |

### High Severity (unchanged)

Existing `_inject_fault()` with `sev=1.0`. Step-change onset, single-service, full magnitude. This is the "obvious fault" that validates the pipeline works.

### Medium Severity (scaled everything)

Existing `_inject_fault()` with `sev=0.3`. ALL metric injections are scaled:

**Additive changes** (CPU, latency, memory, error_rate):
- AR1 signals: `_ar1_signal(fault_len, lo * sev, hi * sev)`
- Uniform error: `uniform(lo, hi) * sev`
- Ramps: `linspace(0, uniform(lo, hi) * sev, fault_len)`

**Multiplicative changes** (network, request_rate):
- Formula: `scaled_factor = 1.0 - (1.0 - original_factor) * sev`
- Helper: `_scale_mult(factor, sev)` on FaultGenerator
- Example: packet_loss network `*= 0.1` becomes `*= 0.73` at sev=0.3

| Fault type | Metric | High (sev=1.0) | Medium (sev=0.3) |
|-----------|--------|-----------------|-------------------|
| cpu_hog | CPU | +30-60 | +9-18 |
| cpu_hog | error | +0.1-0.5 | +0.03-0.15 |
| pod_failure | request_rate | *0.05 | *0.715 |
| pod_failure | error | +0.4-0.8 | +0.12-0.24 |
| packet_loss | network | *0.1 | *0.73 |
| dns_failure | latency | +500-2000 | +150-600 |
| memory_leak | memory | +0-55 ramp | +0-16.5 ramp |

Still single-service, still step-change onset. Trees can likely distinguish these from loads because the pattern shape differs, but the signal is weaker.

### Low Severity: Disguised Faults

New method `generate_disguised_fault()` on `FaultGenerator`. These faults are **metrically indistinguishable from load spikes**.

**How it works:**

1. **Build a load-like envelope** identical to `SyntheticMetricsGenerator.generate_load_spike_metrics()`:
   - Random spike_start in [15%, 35%] of sequence
   - Ramp-up (10-20% of sequence), plateau, ramp-down
   - Same `np.zeros(n)` → linspace → plateau → linspace shape

2. **Pick a "fake" load multiplier** from 2.0-5.0 (same range as real load events)

3. **Apply to ALL services** (not just one):
   - CPU, request_rate, latency, network: `baseline * (1 + (mult-1) * envelope * uniform(0.7, 1.3))`
   - Error: `baseline + (mult-1) * uniform(0.002, 0.01) * envelope`
   - This is identical to load spike generation

4. **Add fault perturbation on the faulty service only:**
   - Extra CPU: +3-8% above the load envelope
   - Extra error_rate: +0.01-0.03 above the load-induced error
   - Extra latency: +20-80ms above the load-induced latency
   - Applied with the same envelope shape (gradual, not step)

5. **Return with empty context** `{}`:
   - No `event_type`, no `load_multiplier`, no `event_name`
   - This is the critical differentiator: a real load spike has context explaining it; a disguised fault does not

**Why trees fail on these cases:**

A tree model seeing a disguised fault sees: all services ramped up uniformly, small error increase proportional to load, gradual onset. This is statistically identical to an EXPECTED_LOAD case. The only difference is in the context features (event_active=0, context_confidence=low). Trees trained on all features including context might learn this rule, but context-free variants (Statistical Only, No Context Features) are stuck at ~50% on these cases.

**Why CAAA succeeds:**

The context integration module sees: metric pattern looks like load, BUT event_active=0 and context_confidence is low. The Context Consistency Loss penalizes predicting EXPECTED_LOAD when no event is active. CAAA learns: "load-like metrics + no context = probably a fault."

### Load Spike Changes (for symmetry)

Load spikes already produce proportional error increases (from A2). No further changes needed. But to maintain the signal for CAAA:

- 70% of load cases retain their event context (event_type, load_multiplier) — providing the positive signal
- 30% of load cases already have empty context (existing behavior, line ~98 of dataset.py) — these are the "unscheduled load spikes" that CAAA must also learn to handle

## Expected Outcomes

### Ablation table (synthetic, n=200, 10 runs)

| Variant | Easy (~30%) | Medium (~35%) | Hard (~35%) | Overall |
|---------|------------|---------------|-------------|---------|
| Full CAAA | ~99% | ~88% | ~80% | ~88% |
| No Context Features | ~99% | ~85% | ~55% | ~78% |
| No Context Loss | ~99% | ~87% | ~70% | ~84% |
| Baseline RF | ~99% | ~85% | ~60% | ~80% |
| CatBoost | ~99% | ~87% | ~62% | ~82% |
| Context Only | ~70% | ~72% | ~73% | ~72% |
| Naive | ~50% | ~50% | ~50% | ~50% |

Key differentiator: Full CAAA outperforms No Context and tree baselines **specifically on the hard tier** where context is the only discriminator.

### Research questions validated

- **RQ1**: Context features provide >15% accuracy improvement on hard cases
- **RQ2**: Context Consistency Loss outperforms standard cross-entropy on disguised faults (penalizes predicting LOAD when no event is active)
- **RQ3**: SHAP plots will show context features ranked high for hard cases but low for easy cases

## Training Loss Fix: ContextConsistencyLoss

**Problem discovered:** In `src/training/losses.py` line ~114, the consistency penalty is weighted by `context_confidence ** 2`:

```python
consistency_loss = torch.mean(context_confidence ** 2 * per_sample_penalty)
```

For disguised faults: `context_confidence ~ 0.0`, so `0.0**2 * penalty = 0.0`. The loss function produces **zero gradient** for disguised faults — the model gets no signal to learn that "load-like metrics + no context = fault."

**Fix:** Replace the squared weighting with a minimum floor so the consistency penalty is never fully zeroed:

```python
# Ensure consistency signal even when confidence is low
conf_weight = torch.clamp(context_confidence, min=0.3)
consistency_loss = torch.mean(conf_weight * per_sample_penalty)
```

Using `clamp(min=0.3)` instead of removing weighting entirely preserves the original intent (high-confidence context gets more weight) while ensuring disguised faults still produce meaningful gradients. The `0.3` floor means disguised faults contribute 30% of the penalty that high-confidence cases do — enough for learning, not so much that it dominates.

**Additional term: Unknown context penalty.** When event_active=0 AND context_confidence is low, explicitly penalize predicting EXPECTED_LOAD:

```python
# Penalize LOAD predictions when no event explains the anomaly
unknown_context = (1.0 - event_active) * (1.0 - context_confidence)
unknown_penalty = unknown_context * load_prob
total_consistency += self.unknown_weight * unknown_penalty.mean()
```

`unknown_weight` is a new hyperparameter (default 0.5, configurable). It's lower than `alpha` (the main consistency weight) to avoid over-penalizing the 30% of real load spikes with empty context — those cases have load-like metrics + no context, same as disguised faults, but the cross-entropy loss on the correct label should dominate.

**Tuning strategy for `unknown_weight`:**

The penalty affects both disguised faults (should predict FAULT) and no-context load spikes (should predict EXPECTED_LOAD). Too high → model always predicts FAULT when context is absent, hurting load accuracy. Too low → no learning signal for disguised faults.

Approach: after implementation, run a quick sweep with `--n-fault 50 --n-load 50 --epochs 30 --n-runs 3`:
- `unknown_weight` in {0.1, 0.3, 0.5, 0.7, 1.0}
- Measure: (a) hard-tier fault recall, (b) no-context load FP rate
- Pick the weight that maximizes hard-tier fault recall while keeping no-context load FP rate < 20%
- Expected sweet spot: 0.3-0.5

Add `--unknown-weight` CLI arg to ablation.py and train.py for easy tuning. Default to 0.5, adjustable.

**File:** `src/training/losses.py` — clamp change + new penalty term (~10 lines).

## Context Feature Analysis

When disguised faults have `context={}`, the 5 context features become:

| Feature | Value | Why |
|---------|-------|-----|
| event_active | 0.0 | No event_type in context |
| event_expected_impact | ~0.0 + noise | No load_multiplier |
| time_seasonality | 0.5 | Synthetic data default |
| recent_deployment | ~0.045 | 15% base rate |
| context_confidence | ~0.0 + noise | No context keys present |

This is correct behavior. CAAA's Context Integration Module uses confidence gating (`sigmoid([confidence, event_active, deployment] + bias=1.0)`) which naturally reduces context influence when confidence is low. No model architecture changes needed.

The key metric features that disguised faults are designed to fool:
- `cross_service_sync`: HIGH (all services ramp → looks like load)
- `affected_service_ratio`: ~1.0 (all services affected → looks like load)
- `global_load_ratio`: ~1.0 (all services increase → looks like load)
- `error_rate_delta`: ~0.0 (no error spike → looks like load)

The small fault perturbation on one service (+3-8% CPU, +0.01-0.03 error, +20-80ms latency) produces a subtle signal in `max_error_service_ratio` and `cpu_spread` that CAAA can learn to detect — but only when combined with the context absence signal.

## Files to Modify (Updated)

| File | Change |
|------|--------|
| `src/data_loader/fault_generator.py` | (1) Scale ALL metric injections by `sev`. (2) Add `_scale_mult()` helper. (3) New `generate_disguised_fault()` method. |
| `src/data_loader/dataset.py` | Route low-severity faults to `generate_disguised_fault()`. |
| `src/training/losses.py` | Clamp context_confidence to min=0.3 in consistency loss weighting. |

Files NOT changed: `synthetic_generator.py`, `extractors.py`, model architecture, tests.

## Backward Compatibility

- Default severity is "high" → existing tests and API calls produce unchanged behavior
- `generate_disguised_fault()` is only called from `generate_combined_dataset()` when severity="low"
- Loss change: `clamp(min=0.3)` means high-confidence cases are unaffected (they already have confidence > 0.3). Only low-confidence cases get the floor.
- All 137 existing tests should pass without modification
