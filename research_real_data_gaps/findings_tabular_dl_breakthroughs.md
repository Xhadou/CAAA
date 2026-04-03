# Tabular Deep Learning Breakthroughs (2024-2025): Closing the Gap with Trees

**Research date:** 2026-03-30
**Context:** CAAA model — 44 features, ~400 training samples, 2 classes, MLP gets 85%, CatBoost gets 91%.
Current architecture: 2-layer MLP (FeatureEncoder) + FiLM context conditioning (ContextIntegrationModule).

---

## Executive Summary

The 2024-2025 period saw a decisive shift: deep learning models now **routinely match or exceed** gradient-boosted trees on tabular data. The key enablers are (1) parameter-efficient ensembling (TabM), (2) foundation models via in-context learning (TabPFN v2, TabICL v2), (3) better default configurations (RealMLP), and (4) numerical feature embeddings (piecewise linear encodings). For CAAA's specific regime (400 samples, 44 features, binary classification), the most promising directions are **TabPFN v2 as a strong baseline/ceiling**, **TabM + PLE as the trainable architecture replacement**, and **RealMLP tricks applied to the existing FeatureEncoder**.

---

## 1. TabM: Parameter-Efficient Ensembling (ICLR 2025)

**Paper:** "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"
**Authors:** Yandex Research (Gorishniy et al.)
**Venue:** ICLR 2025
**Code:** https://github.com/yandex-research/tabm

### Key Architectural Innovation

TabM replaces a single MLP with k "implicit" MLPs that share most parameters via BatchEnsemble mechanics. For each linear layer, instead of k separate weight matrices, TabM uses:

```
l_i(x_i) = s_i * (W @ (r_i * x_i)) + b_i
```

Where W is shared across all k members, and r_i, s_i, b_i are small per-member adapter vectors. Only the first adapter R is randomly initialized with +/-1; remaining adapters initialize to 1 (identity). This produces k diverse predictions that are individually weak but collectively powerful.

- **k=32** is the recommended default (performance plateaus at k=16-32)
- The only new hyperparameter vs a standard MLP is k
- Optional: piecewise linear embeddings (PLE) for numerical features (TabM-dagger variant)

### Performance

- **Best-performing tabular DL model** across academic benchmarks
- Competitive with CatBoost: median rank 5.0 vs CatBoost's 5.5 on TabArena
- On TabArena (post-hoc ensembling): TabM, LightGBM, and RealMLP are the top 3 models
- Mixed results on domain-aware splits with distribution shift (where GBDTs remain strong)

### Small Dataset Suitability (400 samples)

- Paper's smallest benchmark datasets start at ~1.8K samples; no dedicated sub-1K analysis
- However, the paper notes "on small datasets, all methods become almost equally affordable"
- The ensembling mechanism (k=32 implicit models) should help with variance reduction on small data
- Risk: 32 implicit models on 400 samples may overfit unless properly regularized

### FiLM Conditioning Compatibility

**Moderate difficulty.** TabM's backbone is still an MLP — FiLM gamma/beta projections can be inserted between MLP blocks. The challenge is that the BatchEnsemble adapter vectors operate at the linear layer level, so FiLM modulation would need to either:
- (a) Apply uniformly across all k members (simpler, likely sufficient for CAAA)
- (b) Apply per-member (more complex, unclear benefit)

The authors note "more advanced MLP-like backbones can be used" but found "no benefits" in preliminary experiments. This may differ for CAAA where FiLM conditioning is a domain-specific requirement, not a general architectural choice.

### Implementation Complexity

**Low.** Core implementation is a single file (tabm.py). The only additions to a standard MLP are the Clone module (creating k copies of input) and ElementwiseAffine layers (per-member scaling). Estimated ~100-150 lines of core code on top of a standard MLP.

### Recommendation for CAAA

**HIGH PRIORITY.** TabM is the most natural upgrade path for CAAA's FeatureEncoder:
1. Replace the 2-layer MLP with a TabM backbone (k=32, same hidden_dim=64)
2. Keep the FiLM ContextIntegrationModule unchanged
3. Average the k predictions at the output head
4. Add PLE embeddings for the 44 numerical features (TabM-dagger)

Expected gain: 2-4% accuracy improvement from ensembling + PLE alone.

---

## 2. RealMLP: Better Defaults for MLPs (NeurIPS 2024)

**Paper:** "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data"
**Authors:** Holzmuller et al.
**Venue:** NeurIPS 2024
**Code:** https://github.com/dholzmueller/pytabkit (full) / https://github.com/dholzmueller/realmlp-td-s_standalone (standalone)

### Key Architectural Innovation

RealMLP is not a new architecture but a "bag of tricks" applied to standard MLPs that collectively close the gap with GBDTs:

1. **Robust scaling + smooth clipping**: Replaces standard normalization. Handles outliers without destroying signal. Smooth clipping is a differentiable version of winsorization.
2. **Piecewise Binary Linear Decomposition (PBLD) embeddings**: A numerical embedding variant that bins each scalar feature into segments, creating richer initial representations. Replaces raw numerical inputs.
3. **Diagonal weight layer**: An element-wise scaling layer (similar to a 1x1 convolution) that allows per-feature importance weighting before full linear mixing.
4. **Improved training schedule**: Custom learning rate schedules and initialization methods tuned via meta-learning on 118 datasets.
5. **Default hyperparameters**: 3 hidden layers of 256, lr=0.04, batch_size=256, n_epochs=256 — pre-tuned to work well across datasets without HPO.

### Performance

- Competitive with GBDTs (CatBoost, LightGBM) on medium-to-large datasets (1K-500K samples)
- Favorable time-accuracy tradeoff compared to other neural baselines
- On TabArena: RealMLP is among the top 3 models (alongside TabM and LightGBM)
- The "extreme" preset for datasets <= 30K samples suggests specific small-data tuning exists

### Small Dataset Suitability (400 samples)

- Meta-train benchmark used 118 datasets, meta-test used 90 datasets — covers a range of sizes
- The standalone implementation (211 lines for MLP, 91 for preprocessing) is lightweight enough for small data
- PBLD embeddings may help on small data by creating richer feature representations that capture nonlinearities the MLP would otherwise need more data to learn

### FiLM Conditioning Compatibility

**Easy.** RealMLP is still fundamentally an MLP. All the "tricks" are either preprocessing (robust scaling, smooth clipping), input transformations (PBLD embeddings, diagonal weight layer), or training schedule changes. FiLM conditioning can be inserted between hidden layers exactly as in CAAA's current architecture. No architectural conflicts.

### Implementation Complexity

**Very low.** Standalone implementation is 302 lines total (211 MLP + 91 preprocessing). The tricks are modular and can be applied incrementally to CAAA's existing FeatureEncoder.

### Recommendation for CAAA

**HIGH PRIORITY — lowest-effort improvement.** Apply RealMLP tricks to the existing FeatureEncoder without changing the overall architecture:
1. Replace raw feature input with robust scaling + smooth clipping (preprocessing)
2. Add PBLD numerical embeddings before the first linear layer
3. Insert a diagonal weight layer after embeddings
4. Adjust training schedule (lr=0.04, cosine schedule)
5. Consider 3 layers of 256 instead of 2 layers of 64

Expected gain: 2-3% from better preprocessing and embeddings alone. These improvements compose with TabM ensembling.

---

## 3. TabPFN v2 / v2.5: Foundation Model for Small Data (Nature 2024/2025)

**Paper:** "Accurate predictions on small data with a tabular foundation model"
**Authors:** Prior Labs (Hollmann et al.)
**Venue:** Nature, volume 637, pages 319-326 (2025)
**Code:** https://github.com/PriorLabs/TabPFN

### Key Architectural Innovation

TabPFN v2 is a **tabular foundation model** — a transformer pre-trained on ~130 million synthetic datasets that performs in-context learning. Instead of training a model per dataset, you pass the entire training set + test query as a single forward pass through the transformer, and it returns predictions.

- Input: 3D tensor of shape (N+1) x (d+1) x k (training samples + test, features + target, embedding dim)
- Architecture: Alternating self-attention across samples and features
- No hyperparameter tuning needed — the model has already learned "how tabular data works"

### Performance

- **Best on small datasets**: Outperforms all methods on datasets up to 10K samples
- On 273 small-to-medium datasets: best on 26% of datasets (vs ModernNCA 12%, TabM 10%)
- Outperforms CatBoost and XGBoost without any tuning
- TabPFN v2.5 is the top tabular foundation model on TabArena

### Small Dataset Suitability (400 samples)

**This is TabPFN's sweet spot.** The model was specifically designed for small data:
- Handles up to ~10K samples, ~500 features, ~10 classes
- 44 features and 400 samples is well within operating range
- No training required — just fit() and predict()
- Scikit-learn compatible API

### FiLM Conditioning Compatibility

**Not applicable / incompatible.** TabPFN is a frozen foundation model. You cannot add FiLM layers or modify its architecture. The context features would need to be included as regular input features alongside the 44 existing features (making it 49 features total). This loses the explicit context-conditioning mechanism that is central to CAAA's design.

### Limitations

- **License:** v2.5 and v2.6 are non-commercial license (requires HuggingFace login); v2.0 is Apache 2.0
- **No customization:** Cannot add FiLM, custom losses, or domain-specific inductive biases
- **Black box:** No interpretability of the conditioning mechanism
- **GPU recommended:** CPU feasible only for ~1000 samples (fine for CAAA)
- Fine-tuning is technically possible (finetune_classifier.py exists) but this is unexplored territory for adding structural modifications like FiLM

### Recommendation for CAAA

**Use as a CEILING BENCHMARK, not as a replacement.** TabPFN v2 should be evaluated on CAAA's data to establish an upper bound on what's achievable:
1. Run TabPFN v2 with all 49 features (44 temporal + 5 context) as a flat feature vector
2. Compare its accuracy to both the current MLP (85%) and CatBoost (91%)
3. If TabPFN matches or exceeds CatBoost, it validates the data quality and feature set
4. Do NOT replace CAAA's architecture with TabPFN — the FiLM conditioning is a core research contribution

```python
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier()
clf.fit(X_train_all_49_features, y_train)
preds = clf.predict(X_test_all_49_features)
```

---

## 4. ModernNCA: Deep Nearest-Neighbor Analysis (ICLR 2025)

**Paper:** "Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later"
**Authors:** Ye et al.
**Venue:** ICLR 2025
**Code:** Referenced in paper

### Key Architectural Innovation

ModernNCA revives Neighbourhood Components Analysis (NCA) — a differentiable K-nearest-neighbors — by adding modern deep learning:

1. **Deep feature projection**: MLP backbone with batch normalization, ReLU, dropout projects features into an embedding space
2. **PLR (lite) embeddings**: Periodic + Linear + ReLU encoding for numerical features
3. **Soft nearest-neighbor prediction**: Euclidean distance in learned space, weighted average over all training instances
4. **Stochastic neighbor sampling**: During training, sample a subset of neighbors for efficiency; use all neighbors at inference
5. **Custom NCA loss**: Tailored for both classification and regression

### Performance

- On par with CatBoost across 300 datasets: wins on 114, ties on 81, loses on 105
- Outperforms all existing deep tabular models
- High training speed and low model size compared to other deep tabular methods

### Small Dataset Suitability (400 samples)

- Evaluated on a "tiny benchmark" of 45 datasets, suggesting small-data awareness
- The nearest-neighbor nature is inherently suited to small data (every training sample contributes directly to predictions)
- With 400 training samples, inference computes distances to all 400 — computationally trivial
- Risk: the deep projection network may overfit with only 400 samples to learn the embedding space

### FiLM Conditioning Compatibility

**Moderate difficulty.** The MLP backbone can accommodate FiLM layers, but the NCA prediction mechanism (distance-based) is fundamentally different from CAAA's linear classifier head. FiLM would modulate the feature embeddings before distance computation. This is architecturally possible but changes the semantics — context would influence which features are important for similarity, rather than directly modulating the prediction.

### Implementation Complexity

**Moderate.** Requires implementing: the deep projection network (standard MLP), PLR embeddings, the NCA loss with stochastic sampling, and the weighted nearest-neighbor inference. The training loop differs from standard classification (each sample's loss depends on its relationship to all other training samples).

### Recommendation for CAAA

**EXPERIMENTAL — worth trying as a secondary experiment.** ModernNCA's nearest-neighbor approach could be powerful for CAAA's small dataset, but integrating FiLM conditioning changes the model's semantics significantly. Consider as a comparison baseline rather than a replacement.

---

## 5. TabICL v2: Open-Source Foundation Model (2025-2026)

**Paper:** "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data"
**Authors:** INRIA (Giovanelli et al.)
**Venue:** Preprint / Under review
**Code:** https://github.com/soda-inria/tabicl

### Key Architectural Innovation

TabICL v2 uses a two-stage attention architecture for in-context learning:

1. **Column-then-row attention**: First, a transformer processes each column (feature) to build cell embeddings; then a row-wise transformer processes entire rows with rotary positional encoding
2. **[CLS] tokens**: 4 trainable class tokens prepended to each row for aggregation
3. **Pre-trained on synthetic data**: Datasets with 300-60K training samples
4. **No hyperparameter tuning needed**

### Performance

- **State-of-the-art** on TabArena and TALENT benchmarks
- Outperforms tuned CatBoost, XGBoost, LightGBM on ~80% of TabArena datasets
- 10x faster than TabPFN v2.5 on comparable tasks
- Fully open-source (unlike TabPFN v2.5's non-commercial license)

### Small Dataset Suitability (400 samples)

- Pre-trained on datasets starting from 300 samples — CAAA's 400 is within range
- On an H100 GPU: fit + predict on 50K samples in <10 seconds (400 samples: near-instant)
- Generalizes to dataset sizes beyond pre-training range

### FiLM Conditioning Compatibility

**Not applicable.** Like TabPFN, TabICL v2 is a frozen foundation model. Context features must be included as regular input features.

### Recommendation for CAAA

**Use alongside TabPFN v2 as a benchmark ceiling.** TabICL v2 is open-source and may be more practical for academic work. Same caveats as TabPFN regarding inability to integrate FiLM conditioning.

---

## 6. Piecewise Linear Embeddings (PLE) — Cross-cutting Innovation

**Paper:** "On Embeddings for Numerical Features in Tabular Deep Learning" (NeurIPS 2022, but widely adopted 2024-2025)
**Authors:** Gorishniy et al. (Yandex Research)

### What It Does

PLE transforms each scalar numerical feature into a multi-dimensional embedding by:
1. Computing bin boundaries from the training data (quantile-based)
2. For each feature value, computing a piecewise linear interpolation across bins
3. Output: a d_e-dimensional vector per feature (instead of a single scalar)

This is the tabular equivalent of positional encoding — it gives the network a richer, more expressive input representation that mimics the decision boundaries of tree-based models.

### Why It Matters for CAAA

CAAA's FeatureEncoder currently takes raw 44-dimensional input. With PLE:
- Input becomes 44 x d_e dimensional (e.g., 44 x 8 = 352 dimensions)
- Each feature gets its own learned representation
- The first linear layer sees richer patterns without needing to learn binning from scratch
- This is the single most impactful change identified by multiple papers

### Adoption

- Used in TabM-dagger (significant boost over base TabM)
- Used in RealMLP (PBLD variant)
- Used in ModernNCA (PLR variant)
- Universally recommended by the Yandex Research tabular DL group

---

## Synthesis: Recommended Upgrade Path for CAAA

### Priority 1: Low-effort, high-impact (1-2 days)

**Apply RealMLP tricks to existing FeatureEncoder:**
- Add piecewise linear embeddings for all 44 numerical features
- Add robust scaling + smooth clipping in preprocessing
- Add diagonal weight layer after embeddings
- Increase hidden_dim from 64 to 128-256
- Add a third hidden layer
- Adjust learning rate schedule

**Expected gain:** 3-5% accuracy improvement. No architectural changes to FiLM conditioning.

### Priority 2: Moderate effort, high impact (3-5 days)

**Replace FeatureEncoder MLP with TabM backbone:**
- Use k=32 implicit models via BatchEnsemble
- Combine with PLE from Priority 1
- Keep FiLM ContextIntegrationModule unchanged (apply FiLM after TabM encoder)
- Average k predictions at the classifier head

**Expected gain:** Additional 2-3% on top of Priority 1. Total: 5-8% improvement, potentially reaching 90-93%.

### Priority 3: Establish ceiling benchmarks (0.5 days)

**Run TabPFN v2 and TabICL v2 on CAAA data:**
- Flatten all 49 features (44 + 5 context) into a single feature vector
- Run both foundation models with zero tuning
- This establishes the maximum achievable accuracy on this data/feature set
- If foundation models reach 93%+, the feature set is sufficient and architecture improvements can close the gap
- If foundation models also plateau at ~91%, the bottleneck is data/features, not architecture

### Priority 4: Experimental (1 week)

**ModernNCA with FiLM-augmented embeddings:**
- Use the deep NCA approach with context-modulated distance computation
- Novel contribution: FiLM conditioning influences which features matter for similarity
- Higher risk, higher novelty

---

## Quick Reference: Architecture Comparison

| Model | Type | Params | Small Data? | FiLM Compatible? | Impl. Lines | Expected Gain |
|-------|------|--------|-------------|-------------------|-------------|---------------|
| Current MLP | 2-layer MLP | ~5K | Adequate | Yes (current) | Baseline | Baseline (85%) |
| RealMLP tricks | Enhanced MLP | ~50K | Yes | Yes (easy) | +100 lines | +3-5% |
| TabM + PLE | Ensemble MLP | ~10K | Likely | Yes (moderate) | +200 lines | +5-8% |
| TabPFN v2 | Foundation | 100M+ | Excellent | No | 5 lines | Ceiling test |
| TabICL v2 | Foundation | Large | Excellent | No | 5 lines | Ceiling test |
| ModernNCA | Deep KNN | ~10K | Good | Partial | +500 lines | Unknown |

---

## Key Takeaway

The 6% gap between CAAA's MLP (85%) and CatBoost (91%) is **exactly the kind of gap these 2024-2025 innovations were designed to close**. The combination of piecewise linear embeddings + parameter-efficient ensembling (TabM) + better training defaults (RealMLP) has been shown across hundreds of benchmarks to bring MLPs to parity with GBDTs. For CAAA, these improvements can be applied incrementally while preserving the FiLM context conditioning that is central to the research contribution.

---

## Sources

### Papers
- [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (ICLR 2025)](https://arxiv.org/abs/2410.24210)
- [Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data (NeurIPS 2024)](https://arxiv.org/abs/2407.04491)
- [Accurate predictions on small data with a tabular foundation model (Nature 2025)](https://www.nature.com/articles/s41586-024-08328-6)
- [Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later (ICLR 2025)](https://arxiv.org/abs/2407.03257)
- [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564)
- [On Embeddings for Numerical Features in Tabular Deep Learning (NeurIPS 2022)](https://arxiv.org/abs/2203.05556)
- [A Closer Look at TabPFN v2 (2025)](https://arxiv.org/abs/2502.17361)
- [TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2506.16791)

### Code Repositories
- [TabM (Yandex Research)](https://github.com/yandex-research/tabm)
- [TabPFN (Prior Labs)](https://github.com/PriorLabs/TabPFN)
- [TabICL v2 (INRIA)](https://github.com/soda-inria/tabicl)
- [PyTabKit / RealMLP](https://github.com/dholzmueller/pytabkit)
- [RealMLP-TD-S Standalone](https://github.com/dholzmueller/realmlp-td-s_standalone)

### Benchmarks and Analysis
- [TabArena Living Benchmark](https://tabarena.ai)
- [The State of Tabular Foundation Models (2026)](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models)
- [Is Boosting Still All You Need for Tabular Data? (2026)](https://m-clark.github.io/posts/2026-03-01-dl-for-tabular-foundational/)
