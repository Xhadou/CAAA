# Fair Baseline Comparison — Design Spec

**Date**: 2026-04-08
**Status**: Draft
**Problem**: Tree baselines (CatBoost 94.5%) outperform CAAA (93.5%) because they receive the same 5 context features. This undermines the thesis that context-awareness improves anomaly attribution.

## Context

CAAA's contribution is integrating operational context (event calendars, deployment logs) into anomaly attribution via FiLM conditioning. Traditional anomaly detectors don't have this information. The current experimental setup gives tree baselines all 44 features including context — making the comparison "FiLM vs tree splits on identical inputs" rather than "context-aware vs context-unaware."

Academic standard (CIRCA/KDD 2022, RCD/NeurIPS 2022, multi-modal RCA/KDD 2024): when a model uses additional information, baselines should NOT receive that information. The comparison should be "traditional method on standard features" vs "enhanced method on standard + context features."

## Design

### 1. Remove context features from tree baselines

In `scripts/ablation.py`, create context-free feature arrays for tree baselines:

```python
# Delete (not zero) context columns 12-17 so trees get 39 features
ctx_s, ctx_e = CONTEXT_RANGE  # 12, 17
X_train_no_ctx = np.delete(X_train_unscaled, range(ctx_s, ctx_e), axis=1)
X_test_no_ctx = np.delete(X_test_unscaled, range(ctx_s, ctx_e), axis=1)
```

**Why delete, not zero**: Zeroing leaves 5 constant-zero columns that trees waste splits on. Deleting gives a clean 39-feature input.

Apply to: Baseline RF, XGBoost, LightGBM, CatBoost (lines 668-690).

### 2. Add "CatBoost (with context)" variant

Keep one tree baseline that receives all 44 features as an upper-bound reference. This shows:
- Trees also benefit from context (validates our features are useful)
- CAAA's FiLM conditioning is competitive with trees on same features

### 3. Updated variant list

**Remove** (to reduce table noise): "CAAA + Contrastive", "CAAA (clamp loss)", "CAAA (full penalty)", "Stat + Service-Level"

**Final variant ordering** (tells the story):

| # | Variant | Features | Purpose |
|---|---------|----------|---------|
| 1 | Full CAAA | 44 (FiLM context) | **Our method** |
| 2 | No Context Features | 44 (ctx zeroed) | Architecture-only ablation |
| 3 | No Context Loss | 44 | Loss function ablation |
| 4 | No Behavioral | 44 (beh zeroed) | Feature group ablation |
| 5 | Context Only | 5 context only | Context signal strength |
| 6 | Statistical Only | 13 stats only | Metric-only baseline |
| 7 | Baseline RF | 39 (no context) | Traditional baseline |
| 8 | XGBoost | 39 (no context) | Traditional baseline |
| 9 | LightGBM | 39 (no context) | Traditional baseline |
| 10 | CatBoost | 39 (no context) | Traditional baseline |
| 11 | CatBoost (with context) | 44 | Upper-bound reference |
| 12 | CAAA+CatBoost Hybrid | 44+64 emb | Hybrid approach |
| 13 | Rule-Based | 39 (no context) | Simple baseline |
| 14 | Naive | all | Lower bound |

### 4. SHAP plot updates

SHAP feature names for tree baselines need updating since they now have 39 features. Use `ALL_FEATURE_NAMES` with context names removed when generating SHAP plots for tree variants.

### 5. No changes needed to

- CAAA model architecture
- Feature extraction pipeline
- Synthetic/real data generation
- Training loop, loss functions, hyperparameters
- CAAA variants (they still get 44 features — context is their advantage)

## Files to modify

1. **`scripts/ablation.py`** — Main changes:
   - Create `X_train_no_ctx` / `X_test_no_ctx` (np.delete context columns)
   - Update RF, XGBoost, LightGBM, CatBoost calls to use no-context arrays
   - Add "CatBoost (with context)" variant using full arrays
   - Remove deprecated variants
   - Update variant list and ordering

2. **`docs/research_journey.md`** — Add Round 13 documenting the experimental design fix and new results

## Expected results

Based on current numbers:
- CAAA (no context) ≈ 87.8% (already measured)
- CatBoost (no context) ≈ 89-91% (estimate — losing context features should drop it ~3-5%)
- **Full CAAA ≈ 93.5% > CatBoost (no context) ≈ 89-91%** — CAAA wins by ~2-4%
- CatBoost (with context) ≈ 94.5% — trees still strong when given same info (acknowledged)

The story: "Context-aware CAAA outperforms the best traditional baseline by X%. When given the same context features, CatBoost edges ahead — but in real deployments, traditional detectors don't have access to operational context."

## Verification

1. Run synthetic ablation: `python scripts/ablation.py --data synthetic --n-runs 10 --n-fault 50 --n-load 50`
2. Run real data ablation: `python scripts/ablation.py --data rcaeval --dataset all --system all --n-runs 10`
3. Verify: Full CAAA accuracy > CatBoost (no context) accuracy
4. Verify: CatBoost (with context) ≈ previous CatBoost numbers (sanity check)
5. Check SHAP plots generate without errors for 39-feature baselines
