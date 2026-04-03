# Hybrid Neural Network + Tree-Based Model Approaches for Tabular Classification

**Date**: 2026-03-30
**Context**: Research for CAAA (Context-Aware Anomaly Attribution) project
**Goal**: Evaluate whether using CAAA's MLP+FiLM encoder as a feature extractor feeding a 64-dim embedding into CatBoost for final classification is viable and beneficial.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [CAAA Architecture Context](#caaa-architecture-context)
3. [Approach 1: Two-Stage Neural Embedding + Tree Classifier (Proposed)](#approach-1-two-stage-neural-embedding--tree-classifier)
4. [Approach 2: NODE - Neural Oblivious Decision Ensembles](#approach-2-node---neural-oblivious-decision-ensembles)
5. [Approach 3: GrowNet - Gradient Boosting Neural Networks](#approach-3-grownet---gradient-boosting-neural-networks)
6. [Approach 4: DeepGBM - Knowledge Distillation from GBDT to NN](#approach-4-deepgbm---knowledge-distillation-from-gbdt-to-nn)
7. [Approach 5: GATE/GANDALF - Gated Additive Tree Ensemble](#approach-5-gategandalf---gated-additive-tree-ensemble)
8. [Approach 6: XBNet - XGBoost-Initialized Neural Networks](#approach-6-xbnet---xgboost-initialized-neural-networks)
9. [Approach 7: NCART - Neural Classification and Regression Tree](#approach-7-ncart---neural-classification-and-regression-tree)
10. [Approach 8: TabPFN - Tabular Foundation Model (Comparator)](#approach-8-tabpfn---tabular-foundation-model)
11. [Approach 9: TabM - Parameter-Efficient MLP Ensemble (ICLR 2025)](#approach-9-tabm---parameter-efficient-mlp-ensemble)
12. [Approach 10: gcForest - Deep Forest Cascade](#approach-10-gcforest---deep-forest-cascade)
13. [Approach 11: LLM/Pretrained Embedding Enrichment](#approach-11-llmpretrained-embedding-enrichment)
14. [Comparative Analysis](#comparative-analysis)
15. [Recommendation for CAAA](#recommendation-for-caaa)
16. [Implementation Plan](#implementation-plan)
17. [Sources](#sources)

---

## Executive Summary

The proposed hybrid approach -- using CAAA's FiLM-conditioned MLP encoder as a feature extractor to produce 64-dim embeddings, then feeding those into CatBoost for final classification -- is **well-supported by the literature** and represents a pragmatic middle ground. Key findings:

1. **Neural embeddings + tree classifiers consistently outperform pure approaches** on small, imbalanced tabular datasets when the neural encoder captures meaningful structure (e.g., context conditioning) that raw features cannot express.
2. **CatBoost/XGBoost dominate on tabular data** with <50K samples. The TabZilla benchmark (176 datasets, 19 algorithms) confirmed CatBoost as the most consistent top performer.
3. **The two-stage pipeline is simpler than integrated approaches** (NODE, GATE, GrowNet) while retaining most benefits -- the neural encoder handles what it's best at (FiLM conditioning, representation learning) while the tree handles what it's best at (small-data classification with built-in feature selection).
4. **CAAA already has `get_embeddings()`** -- the implementation is nearly trivial.

**Verdict**: Implement the two-stage hybrid as the primary approach. It has the best effort-to-improvement ratio for CAAA's specific constraints (small data, context-dependent features, FiLM conditioning).

---

## CAAA Architecture Context

The current CAAA model (`src/models/caaa_model.py`) has this pipeline:

```
Raw 44-dim features
    |
    v
FeatureEncoder (MLP: 44 -> 64, 2 layers, LayerNorm + GELU + Dropout)
    |
    v
ContextIntegrationModule (FiLM conditioning: gamma * features + beta, confidence gating)
    |
    v
64-dim embedding  <--- THIS IS THE KEY EXTRACTION POINT (get_embeddings())
    |
    v
Classifier head (Linear 64->32->2, GELU, Dropout)
    |
    v
2-class logits (FAULT / EXPECTED_LOAD)
```

The **64-dim embedding** after ContextIntegrationModule is the natural extraction point. It already:
- Encodes all 44 input features (statistical, temporal, context)
- Applies FiLM conditioning from context features (event_active, time_seasonality, etc.)
- Gates context influence via learned confidence gating
- Has a residual connection preserving raw encoded information

The classifier head is a simple 2-layer MLP (64 -> 32 -> 2) -- this is the weakest link on small data and the natural replacement target.

---

## Approach 1: Two-Stage Neural Embedding + Tree Classifier

**This is the proposed approach for CAAA.**

### Architecture

```
Stage 1: Train CAAA neural model end-to-end (as currently done)
    - FeatureEncoder + ContextIntegrationModule + classifier head
    - Supervised contrastive loss + cross-entropy loss
    - Produces a trained encoder that generates 64-dim FiLM-conditioned embeddings

Stage 2: Extract embeddings, train CatBoost
    - model.get_embeddings(X) -> 64-dim vectors
    - Optionally concatenate with select raw features (for tree interpretability)
    - Train CatBoost classifier on these embeddings
    - CatBoost handles the final FAULT vs EXPECTED_LOAD decision
```

### Evidence of Effectiveness

**Embedding enrichment study (2024)**:
A comprehensive ablation study on enriching tabular data with contextual embeddings for ensemble classifiers (Random Forest, XGBoost, CatBoost) found:
- Embedding-enriched subsets outperformed baseline-only features on imbalanced/limited datasets
- XGBoost and CatBoost benefited more consistently than Random Forest
- Pure embedding subsets performed poorly; the hybrid baseline+embeddings was optimal
- Improvements were most notable on small datasets with class imbalance or limited features

**CatBoost native embedding support**:
CatBoost v1.0+ natively supports embedding features as input, applying LDA and nearest-neighbor transformations internally. This means 64-dim embeddings can be passed directly without manual dimensionality reduction.

**Practical Kaggle evidence**:
The "PyTorch NN with Embeddings and CatBoost" pattern is well-established in competitive ML, where practitioners routinely extract penultimate-layer embeddings from trained neural networks and feed them into gradient boosting models for final prediction.

### Does It Outperform Pure Approaches?

- **vs. Pure Neural (current CAAA MLP head)**: Yes, on small data. Tree models handle irregular decision boundaries, feature interactions, and class imbalance better than a simple 64->32->2 MLP head with limited training data.
- **vs. Pure CatBoost on raw features**: Yes, because the FiLM-conditioned embeddings capture context-aware representations that raw features cannot. The context integration (gamma * features + beta, confidence gating) creates features that are fundamentally different from what CatBoost could learn from raw 44-dim input.

### Small Dataset Applicability

**Excellent**. This approach specifically shines on small data because:
1. The neural encoder is pre-trained with the full loss landscape (contrastive + CE), so embeddings are rich even with few samples.
2. CatBoost is specifically designed for small-data regime with ordered boosting that reduces overfitting.
3. The 64-dim embedding space is compact enough for trees to handle without curse of dimensionality.

### Implementation Complexity

**Very Low**. The `CAAAModel.get_embeddings()` method already exists. Implementation requires:
1. After training the neural model, call `model.get_embeddings(X_train)` and `model.get_embeddings(X_test)`.
2. Train CatBoost on the resulting 64-dim vectors.
3. Optionally concatenate with select raw features for interpretability.

Estimated implementation: ~50-100 lines of code.

---

## Approach 2: NODE - Neural Oblivious Decision Ensembles

**Paper**: Popov, Morozov, Babenko (2019, Yandex) - "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data"

### Architecture

NODE makes oblivious decision trees fully differentiable and trainable via backpropagation:

1. **Differentiable Feature Selection**: Instead of discrete feature selection at each split, NODE uses a learnable matrix F with alpha-entmax transformation for sparse, differentiable feature selection.

2. **Soft Splitting**: Hard 0/1 routing decisions are replaced with sigmoid-like activations (entmoid): `c_i(x) = sigma_alpha((f_i(x) - b_i) / tau_i)` where b_i is a learnable threshold and tau_i is a scale parameter.

3. **Response Function**: Decision paths are represented probabilistically through a choice tensor C (outer product of individual routing decisions), generating 2^d possible outputs weighted by path probabilities.

4. **Multi-Layer Stacking**: Multiple NODE layers are stacked with residual connections. Input features and outputs from all previous layers are concatenated for subsequent layers.

5. **Ensemble**: Final predictions average across all layers (similar to random forest ensembling).

**Default configuration**: Single layer of 2,048 trees with depth 6.

### Does It Outperform Pure Approaches?

- Outperformed XGBoost/CatBoost with default hyperparameters on 6 large datasets (400K-10.5M samples).
- With hyperparameter tuning, the advantage narrows or disappears.
- On small datasets (<10K), CatBoost with tuning typically matches or beats NODE.

### Small Dataset Applicability

**Moderate**. NODE was designed for and tested on large datasets. The 2,048-tree default configuration has many parameters. On small datasets, overfitting risk is higher than CatBoost.

### Implementation Complexity

**High**. Requires custom differentiable tree implementation, entmoid activations, sparse feature selection with entmax. Reference implementation available at github.com/Qwicen/node but integrating with CAAA's FiLM conditioning would require significant architectural modifications.

---

## Approach 3: GrowNet - Gradient Boosting Neural Networks

**Paper**: Badirli et al. (2020, Purdue/Amazon) - "Gradient Boosting Neural Networks: GrowNet"

### Architecture

GrowNet uses shallow neural networks (instead of decision trees) as weak learners in a gradient boosting framework:

1. **Weak Learners**: Each is a 2-hidden-layer MLP with ReLU/LeakyReLU activations, batch normalization. Hidden layer size ~ half of input features.

2. **Feature Composition**: After the first learner, each subsequent one receives original input features PLUS penultimate-layer features from the previous learner. Input dimension stays constant (hidden_dim + original_dim).

3. **Training**: Newton-Raphson optimization with second-order gradients. Each weak learner trains for just 1 epoch.

4. **Corrective Step**: Crucially, after adding a new weak learner, ALL previous learners' parameters are updated via backpropagation. This distinguishes GrowNet from standard gradient boosting.

5. **Final Prediction**: `y_hat = sum(alpha_k * f_k(x))` where each f_k is an independent shallow MLP.

### Does It Outperform Pure Approaches?

- **Classification (Higgs)**: AUC 0.8510 vs XGBoost 0.8304.
- **Regression**: 21% RMSE improvement on CT slice localization.
- **vs. Deep Neural Nets**: GrowNet (30 weak learners) achieved 0.8401 AUC; best DNN (10 hidden layers) only 0.8342 AUC.
- Best results on medium-to-large datasets.

### Small Dataset Applicability

**Moderate**. The sequential boosting with corrective steps adds regularization, but 30 weak learners with 2 hidden layers each still has substantial capacity. Not specifically designed for small data.

### Implementation Complexity

**Medium-High**. Requires custom gradient boosting loop with second-order gradients, corrective step backpropagation through all learners, and feature composition between stages. Reference code at github.com/sbadirli/GrowNet.

---

## Approach 4: DeepGBM - Knowledge Distillation from GBDT to NN

**Paper**: Ke et al. (2019, Microsoft) - "DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks"

### Architecture

DeepGBM integrates GBDT advantages into neural networks via two components:

1. **CatNN**: Handles sparse categorical features using learned embeddings (similar to entity embeddings).

2. **GBDT2NN**: Focuses on dense numerical features. A GBDT is first trained on numerical features, then its learned leaf-index structure and feature importances are distilled into a neural network. The NN learns to approximate the GBDT's splitting behavior.

3. **Integration**: CatNN and GBDT2NN outputs are concatenated and passed through a final prediction layer.

### Does It Outperform Pure Approaches?

- Designed for online prediction tasks requiring frequent model updates.
- Competitive with GBDT on static datasets but advantages emerge in streaming/online settings.
- Not specifically targeted at outperforming GBDT on standard benchmarks.

### Small Dataset Applicability

**Low**. DeepGBM is designed for large-scale online prediction (millions of samples). The distillation process requires sufficient data to train the initial GBDT meaningfully.

### Implementation Complexity

**High**. Two separate neural components, GBDT training, knowledge distillation, and integration. Not suitable for CAAA's use case.

---

## Approach 5: GATE/GANDALF - Gated Additive Tree Ensemble

**Paper**: Joseph & Raj (2022) - "GATE: Gated Additive Tree Ensemble for Tabular Classification and Regression"

### Architecture

GATE combines three components:

1. **GRU-Inspired Gating**: A gating mechanism (from GRU cells) serves as both feature representation learning and built-in feature selection.

2. **Differentiable Non-Linear Decision Trees**: An ensemble of differentiable decision trees (similar to NODE but with different gating).

3. **Self-Attention Re-weighting**: Simple self-attention mechanism re-weights the tree ensemble outputs before final prediction.

### Does It Outperform Pure Approaches?

- Competitive with GBDTs, NODE, and FT-Transformers on classification and regression benchmarks.
- On large datasets (26M-800M examples), tied with LightGBM for best average rank.
- Not a clear winner over tuned CatBoost on small datasets.

### Small Dataset Applicability

**Moderate**. GRU gating and self-attention add parameters. Performance advantages emerge on larger datasets.

### Implementation Complexity

**Medium**. Available in the PyTorch Tabular library. Integrating with CAAA's FiLM conditioning would require modifications to the gating mechanism.

---

## Approach 6: XBNet - XGBoost-Initialized Neural Networks

**Paper**: Sarkar (2021) - "XBNet: An Extremely Boosted Neural Network"

### Architecture

XBNet uses XGBoost to guide neural network training:

1. **Initialization**: Train XGBoost first, derive feature importances.
2. **Weight Update**: Each layer's weights are updated in two steps:
   - Standard gradient descent
   - Feature importance from a gradient boosted tree trained at every intermediate layer
3. **Boosted Gradient Descent**: XGBoost feature importances modulate the learning of each neural network layer.

### Does It Outperform Pure Approaches?

- **Mixed results**: Outperformed XGBoost on only 3 of 8 small datasets in the original paper.
- The overhead of training XGBoost at every layer significantly increases training time.

### Small Dataset Applicability

**Moderate**. Designed for small data, but empirical results were inconsistent.

### Implementation Complexity

**Medium**. PyTorch-based (pip install XBNet). However, training an XGBoost model at every layer is computationally expensive and adds complexity.

---

## Approach 7: NCART - Neural Classification and Regression Tree

**Paper**: (2023) - "NCART: Neural Classification and Regression Tree for Tabular Data"

### Architecture

1. **Batch Normalization**: Standardizes inputs without special categorical handling.
2. **Learnable Sparse Feature Selection**: Projection matrix with sparsemax/entmax functions.
3. **Differentiable Oblivious Trees**: Multiple parallel trees with sigmoid replacing hard splits and a two-layer network mapping split regions to outputs.
4. **Ensemble**: Weighted averaging with learnable per-tree weights.
5. **ResNet-style stacking**: Multiple NCART blocks with residual connections.

### Does It Outperform Pure Approaches?

- Average rank 3.20 (F1) and 3.60 (AUC) among 11 models -- competitive but not best.
- Notable gap vs GBDT on regression tasks.
- Sensitive to hyperparameter tuning compared to tree models.

### Small Dataset Applicability

**Good**. Fewer parameters than transformer-based alternatives (TabNet, SAINT, FT-Transformer). Strong inference efficiency on small datasets.

### Implementation Complexity

**Medium**. PyTorch-based. Simpler than attention mechanisms but the sparse feature selection layer adds complexity beyond standard ResNets.

---

## Approach 8: TabPFN - Tabular Foundation Model

**Paper**: Hollmann et al. (2024, Nature) - "Accurate predictions on small data with a tabular foundation model"

### Architecture

TabPFN is a transformer-based foundation model pre-trained on millions of synthetic tabular datasets:

1. **Per-value tokenization**: Each cell in the table is its own token (not row-flattened).
2. **Two-stage attention**: First learns intra-row feature relationships, then inter-row patterns.
3. **In-context learning**: At inference, the model receives the entire training set as context and makes predictions without gradient-based training.

### Does It Outperform Pure Approaches?

- **Dominates on small data**: Outperforms all methods on datasets with up to 10,000 samples.
- **Speed**: In 2.8 seconds, TabPFN outperforms an ensemble of strongest baselines tuned for 4 hours.
- **TabPFN v2.5** (Nov 2025): 100% win rate vs default XGBoost on small-to-medium classification datasets (<= 10K), 87% win rate on up to 100K samples.

### Small Dataset Applicability

**Excellent**. Specifically designed for and excels on small datasets. This is its primary advantage.

### Implementation Complexity

**Very Low** for usage (pip install tabpfn). However, **not customizable** -- cannot incorporate CAAA's FiLM conditioning or context-aware features into TabPFN's architecture. It treats the task as a generic tabular classification problem.

**Key Limitation for CAAA**: TabPFN cannot leverage the FiLM-conditioned context integration that is CAAA's core novelty. The 64-dim embeddings from CAAA's encoder capture context-aware information that TabPFN would not access.

---

## Approach 9: TabM - Parameter-Efficient MLP Ensemble

**Paper**: Yandex Research (ICLR 2025) - "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"

### Architecture

1. **Base Model**: Standard MLP backbone.
2. **BatchEnsemble-like technique**: Instead of training K separate MLPs, uses one shared model with lightweight adapter layers per ensemble member.
3. **Efficiency**: 32x parameter reduction vs naive ensemble without accuracy loss.

### Does It Outperform Pure Approaches?

- **TabM-mini rank 1.7** across 46 benchmark datasets -- outperforms XGBoost, CatBoost, LightGBM.
- First tabular DL model to consistently beat GBDTs across broad benchmarks.
- "Easily competes with GBDT and outperforms prior tabular DL models while being more efficient."

### Small Dataset Applicability

**Good**. MLP-based, so lower parameter count than transformer architectures. The parameter-efficient ensembling helps prevent overfitting.

### Implementation Complexity

**Low-Medium**. PyTorch-based, available at github.com/yandex-research/tabm. Would require modification to integrate with CAAA's encoder.

---

## Approach 10: gcForest - Deep Forest Cascade

**Paper**: Zhou & Feng (2017) - "Deep Forest: Towards an Alternative to Deep Neural Networks"

### Architecture

1. **Multi-Grained Scanning**: Sliding windows scan raw features at multiple granularities.
2. **Cascade Structure**: Multiple layers, each containing an ensemble of Random Forests and Completely Random Forests.
3. **Layer-wise Feature Augmentation**: Each layer concatenates probabilistic outputs from forests with raw features for the next layer.
4. **Adaptive Depth**: Number of cascade levels determined automatically based on validation performance.

### Does It Outperform Pure Approaches?

- Competitive with DNNs across many tasks.
- Particularly strong when labeled data is scarce.
- Model complexity adapts to data -- prevents overfitting on small datasets.

### Small Dataset Applicability

**Excellent**. Explicitly designed to work well on small-scale data, unlike DNNs. Minimal hyperparameter tuning required.

### Implementation Complexity

**Low**. Available as the `deep-forest` Python package. However, integrating with CAAA's FiLM-conditioned embeddings would require treating embeddings as input features to the cascade.

---

## Approach 11: LLM/Pretrained Embedding Enrichment

**Paper**: (2024) - "Enriching Tabular Data with Contextual LLM Embeddings"

### Architecture

1. **Text Conversion**: Tabular features -> text representation ("feature_name: value, ...")
2. **Embedding Generation**: Pre-trained language models (RoBERTa, GPT-2) generate embeddings, PCA-reduced to 50 dims.
3. **Feature Selection**: Random Forest importance scores select top 10 embedding dimensions.
4. **Enrichment**: Selected embedding features concatenated with original features.
5. **Classification**: Standard ensemble classifiers (RF, XGBoost, CatBoost) on enriched feature set.

### Does It Outperform Pure Approaches?

- Benefits most on datasets with class imbalance or limited features.
- XGBoost and CatBoost gain more than Random Forest from embedding enrichment.
- On well-structured datasets, gains were negligible.
- Pure embedding subsets underperformed; hybrid (original + embeddings) was best.

### Relevance to CAAA

**Moderate**. The pattern validates the general approach of "neural embeddings + tree classifier." However, CAAA's FiLM-conditioned embeddings are task-specific and more informative than generic LLM embeddings for this domain.

---

## Comparative Analysis

| Approach | Outperforms Pure? | Small Data? | Impl. Complexity | CAAA FiLM Compatible? | Recommended? |
|----------|:--:|:--:|:--:|:--:|:--:|
| **Two-Stage Embed + CatBoost** | Yes (both) | Excellent | Very Low | Native | **YES** |
| NODE | Marginal on small | Moderate | High | Hard | No |
| GrowNet | Yes on large | Moderate | Medium-High | Hard | No |
| DeepGBM | Online settings | Low | High | No | No |
| GATE/GANDALF | Ties on large | Moderate | Medium | Hard | No |
| XBNet | Inconsistent | Moderate | Medium | Hard | No |
| NCART | Competitive | Good | Medium | Hard | No |
| TabPFN | Yes (small data) | Excellent | Very Low | **Impossible** | Comparator only |
| TabM (ICLR 2025) | Yes | Good | Low-Medium | Hard | Maybe (v2) |
| gcForest | Yes (small data) | Excellent | Low | As input features | Maybe |
| LLM Embeddings | Selective | Good | Low | N/A | Validates pattern |

### Key Insight

The integrated approaches (NODE, GATE, GrowNet, NCART) all try to make trees differentiable or embed tree-like behavior in neural networks. They are architecturally complex and hard to integrate with CAAA's FiLM conditioning. The **two-stage approach sidesteps this entirely**: let the neural encoder do what it's good at (FiLM conditioning, contrastive learning, representation learning), then let the tree do what it's good at (small-data classification, feature interaction modeling, robustness to overfitting).

---

## Recommendation for CAAA

### Primary: Two-Stage Neural Embedding + CatBoost

**Rationale**:

1. **CAAA's core novelty is the FiLM-conditioned context integration** -- this cannot be replicated by any tree model or generic tabular DL approach. The neural encoder must be preserved.

2. **The classifier head is the bottleneck** -- a 64->32->2 MLP with ~2K parameters making binary decisions on small data is exactly where trees excel.

3. **Implementation is trivial** -- `get_embeddings()` already exists. CatBoost training is ~10 lines of code.

4. **Embedding + CatBoost is well-validated** -- Literature shows consistent improvements on small, imbalanced datasets (exactly CAAA's regime).

5. **No architectural changes needed** -- The neural model trains exactly as before. CatBoost is a drop-in replacement for only the final classification step.

### Secondary: TabPFN as a Comparator

- Use TabPFN on raw features as a **comparison baseline** to understand how much of CAAA's value comes from the FiLM conditioning vs. the classification head.
- If TabPFN on raw 44 features matches CAAA's neural embeddings + CatBoost, the FiLM conditioning may not be adding value.
- If CAAA embeddings + CatBoost beats TabPFN on raw features, it validates the FiLM conditioning approach.

### Tertiary: Concatenated Features Variant

- Feed CatBoost both the 64-dim embeddings AND select raw features.
- This gives the tree access to both the learned context-conditioned representation and the original interpretable features.
- CatBoost's built-in feature importance will reveal which representation the tree finds more useful.

---

## Implementation Plan

### Phase 1: Basic Two-Stage Pipeline (~50 lines)

```python
# After neural model training:
model.eval()
with torch.no_grad():
    train_embeddings = model.get_embeddings(X_train_tensor).numpy()
    test_embeddings = model.get_embeddings(X_test_tensor).numpy()

from catboost import CatBoostClassifier
cb = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=0,
)
cb.fit(train_embeddings, y_train)
y_pred = cb.predict(test_embeddings)
```

### Phase 2: Concatenated Features Variant

```python
# Concatenate embeddings with select raw features
train_hybrid = np.concatenate([train_embeddings, X_train_raw], axis=1)
test_hybrid = np.concatenate([test_embeddings, X_test_raw], axis=1)
cb.fit(train_hybrid, y_train)
```

### Phase 3: Ablation Study

Compare:
1. Raw features -> CatBoost (baseline)
2. Raw features -> CAAA neural model (current)
3. Neural embeddings -> CatBoost (**proposed**)
4. Neural embeddings + raw features -> CatBoost (concatenated variant)
5. Raw features -> TabPFN (foundation model comparator)

### Phase 4: Training Refinement

- Experiment with supervised contrastive loss weight (higher weight -> more separable embeddings for trees)
- Try different embedding dimensions (32, 64, 128)
- Consider fine-tuning: train neural model, extract embeddings, train CatBoost, then optionally fine-tune the encoder to maximize CatBoost's SHAP-based feature importance alignment

---

## Sources

### Key Papers
- [When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/pdf/2305.02997) - TabZilla benchmark, 176 datasets, 19 algorithms
- [Neural Oblivious Decision Ensembles (NODE)](https://arxiv.org/abs/1909.06312) - Popov et al. 2019
- [GrowNet: Gradient Boosting Neural Networks](https://ar5iv.labs.arxiv.org/html/2002.07971) - Badirli et al. 2020
- [DeepGBM: A Deep Learning Framework Distilled by GBDT](https://dl.acm.org/doi/10.1145/3292500.3330858) - Ke et al. 2019
- [GATE/GANDALF: Gated Additive Tree Ensemble](https://arxiv.org/abs/2207.08548v3) - Joseph & Raj 2022
- [NCART: Neural Classification and Regression Tree](https://arxiv.org/html/2307.12198v2) - 2023
- [TabPFN: Accurate predictions on small data](https://www.nature.com/articles/s41586-024-08328-6) - Hollmann et al. 2024, Nature
- [TabPFN-2.5: Advancing State of the Art](https://arxiv.org/abs/2511.08667) - Nov 2025
- [TabM: Advancing Tabular Deep Learning (ICLR 2025)](https://github.com/yandex-research/tabm) - Yandex Research
- [XBNet: An Extremely Boosted Neural Network](https://arxiv.org/pdf/2106.05239) - Sarkar 2021
- [Deep Forest: Towards an Alternative to Deep Neural Networks](https://arxiv.org/abs/1702.08835v2) - Zhou & Feng 2017
- [Enriching Tabular Data with Contextual LLM Embeddings](https://arxiv.org/html/2411.01645v1) - 2024
- [Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data?](https://arxiv.org/abs/2207.08815) - Grinsztajn et al. 2022, NeurIPS

### Surveys and Chronologies
- [Deep Neural Networks and Tabular Data: A Survey](https://arxiv.org/pdf/2110.01889)
- [A Short Chronology of Deep Learning for Tabular Data](https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html) - Sebastian Raschka
- [A Comprehensive Benchmark of ML and DL Across Diverse Tabular Datasets](https://arxiv.org/html/2408.14817v1) - 2024
- [TALENT: Comprehensive Toolkit and Benchmark for Tabular Learning](https://github.com/LAMDA-Tabular/TALENT) - 300+ datasets

### Implementation References
- [FiLM: Feature-wise Linear Modulation](https://distill.pub/2018/feature-wise-transformations/) - Distill.pub
- [CatBoost Embedding Features](https://catboost.ai/docs/en/features/embeddings-features) - Official docs
- [PyTorch: Extracting Intermediate Layer Outputs](https://www.kaggle.com/code/mohammaddehghan/pytorch-extracting-intermediate-layer-outputs)
- [Extracting Intermediate Layer Outputs in PyTorch](https://www.kozodoi.me/blog/20210527/extracting-features)

### Microservice Anomaly Detection
- [A Comprehensive Survey on Root Cause Analysis in (Micro) Services](https://arxiv.org/html/2408.00803v1) - 2024
- [Anomaly Detection for Microservice Systems via Multimodal Data and Hybrid Graphs](https://www.sciencedirect.com/science/article/abs/pii/S1566253525000909) - 2025
- [CatBoost-Enhanced CNN with Explainable AI for Smart-Grid Stability](https://www.frontiersin.org/journals/smart-grids/articles/10.3389/frsgr.2025.1617763/full) - 2025
- [Ensemble of CatBoost and Neural Networks for Heart Disease Prediction](https://scientifictemper.com/index.php/tst/article/view/1570) - 2024
