# Findings: Causal & Counterfactual Evaluation for Context-Dependent Classification

## Research Question

When one class (e.g., "fault") always has associated context metadata and another class (e.g., "normal") never does, how do researchers evaluate whether a model has learned genuine patterns versus simply exploiting the context as a shortcut? This document synthesizes findings from the shortcut learning, spurious correlation, causal inference, and counterfactual evaluation literatures.

---

## 1. Shortcut Learning: Foundational Framework

### Definition (Geirhos et al., 2020)

Shortcut learning is the central framing for this problem. Geirhos et al. define shortcuts as **"decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios."** The paper, published in Nature Machine Intelligence, argues that many deep learning failures are symptoms of this single underlying problem.

**Key insight**: The model finds a statistically valid but semantically meaningless rule. In our anomaly detection context, if faults always carry context metadata (e.g., "deployment event at 14:00") and normal windows never do, the model can achieve perfect training accuracy by simply checking whether context is present -- without learning any metric-level anomaly pattern.

**Classic examples from the literature**:
- Models classifying cows vs. camels learned to use background (green pasture vs. desert) rather than the animal itself.
- In a toy task with stars and moons, the network learned object *location* instead of *shape* because stars were always shown in top-right/bottom-left and moons in top-left/bottom-right.
- Medical imaging models learned to use hospital-specific annotations and imaging artifacts rather than pathological features.

**Recommendations from Geirhos et al.**:
1. **Out-of-distribution (OOD) testing**: Deploy test sets where the shortcut correlation is broken (e.g., randomized context assignment).
2. **Ablation studies**: Remove candidate shortcut features and measure performance degradation.
3. **Context manipulation**: Systematically vary context features to expose shortcut dependencies.

> Source: Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R.S., Brendel, W., Bethge, M., & Wichmann, F.A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2, 665-673. https://arxiv.org/abs/2004.07780

---

## 2. Detecting Shortcut Reliance: A Taxonomy of Methods

The comprehensive survey by Marcinowski et al. (2024), "Navigating Shortcuts, Spurious Correlations, and Confounders," provides a structured taxonomy of detection and mitigation methods.

### 2.1 Model Utility-Based Detection

These methods identify shortcuts by analyzing model behavior during training:

- **Generalized Cross Entropy (GCE)**: Trains an auxiliary detector to distinguish easy-to-learn from hard-to-learn samples. Easy samples often correspond to shortcut-reliant predictions (Luo et al.; Nam et al.).
- **SPARE**: Clusters network outputs early in training to separate majority/minority groups -- since shortcuts are learned first, early-training clusters reveal shortcut reliance (Yang et al.).
- **Low-capacity networks**: Train a deliberately limited model; features it learns are likely shortcuts because they are the "easiest" signals (Dagaev et al.).
- **Mutual information analysis**: Measures MI between input features and learned representations to identify which features the model disproportionately relies on (Adnan et al.).

**Relevance to our problem**: Train a minimal model (e.g., logistic regression) on just the context features. If it achieves high accuracy, context is acting as a shortcut.

### 2.2 Perturbation-Based Detection

- **Manual perturbations**: Domain experts craft modifications to reveal shortcut dependencies (e.g., removing context metadata from fault samples).
- **Semi-automated**: GAN-guided semantic manipulations to test robustness.
- **Fully automated**: Frequency-based perturbations that sequentially remove information bands.

**Relevance**: Systematically perturb context features (null them, randomize them, swap them between classes) and measure accuracy changes.

### 2.3 Explainability-Based Detection (XAI)

- **Heatmap clustering**: Use Grad-CAM or LRP to generate explanation heatmaps, then cluster them. If explanations focus on context features rather than metric patterns, shortcuts are present (Lapuschkin; Schramowski).
- **Sufficient Input Subsets**: Find the minimal set of features needed for prediction. If context alone is sufficient, the model has learned a shortcut (Carter et al.).
- **Counterfactual generation**: Generate samples where only the shortcut feature differs, then check if predictions change (DeGrave et al.; Sikka et al.).

**Relevance**: Use SHAP/feature importance on the trained model. If context-related features dominate, the model is shortcutting.

### 2.4 Causality-Based Detection

- **Interventional data**: Use interventional distributions (breaking natural correlations) to estimate causal effects of features (Kumar et al.).
- **Causal DAGs**: Model the data-generating process as a DAG; shortcut features can be formally identified as those connected to the label through non-causal (backdoor) paths (Zheng and Makar).
- **Hidden confounders across environments**: Identify confounders that create spurious associations in some environments but not others (Karlsson and Krijthe).

> Source: Marcinowski et al. (2024). Navigating Shortcuts, Spurious Correlations, and Confounders: From Origins via Detection to Mitigation. https://arxiv.org/html/2412.05152v1

---

## 3. Structural Characterization of Shortcut Features via Causal DAGs

Defined formally in Roelofs et al. (2022), **shortcut features are characterized using causal directed acyclic graphs (DAGs)**. This represents "the first attempt at defining shortcuts in terms of their causal relationship with prediction targets."

### The Causal Graph Perspective

In the context of anomaly detection with context metadata:

```
Context Metadata (C) <-- Root Cause Event (R) --> Fault Label (Y)
                                                      |
                                                      v
                                                Metric Pattern (X)
```

- **R** (root cause event) causes both **C** (context metadata like "deployment happened") and **Y** (the fault).
- **R** also causes **X** (the metric anomaly pattern).
- The model should learn X -> Y, but C is a collider/confounder that provides a shortcut path.

**Formal criterion**: A feature is a shortcut if:
1. It is statistically associated with the label in the training distribution.
2. This association is mediated through a non-causal (backdoor) path.
3. The association does not hold under interventional/counterfactual distributions.

**Key insight**: Context is a *collider* or *mediator* in the causal graph. The fact that normal data never has context metadata means the mere presence/absence of context is a perfect proxy for the label -- a textbook spurious correlation through a confounded data collection process.

> Source: Roelofs et al. (2022). A structural characterization of shortcut features for prediction. *European Journal of Epidemiology*. https://link.springer.com/article/10.1007/s10654-022-00892-3

---

## 4. Evaluation Metrics for Spurious Correlation Robustness

From the comprehensive survey by Ye et al. (2024), "Spurious Correlations in Machine Learning":

### 4.1 Worst-Group Accuracy (WGA)

**The primary metric in the literature**. Defined as the minimum accuracy across all groups formed by combinations of label and spurious attribute values.

For our problem, the groups would be:
| Group | Label | Context Present? | Expected in Real World? |
|-------|-------|-------------------|------------------------|
| G1    | Fault | Yes               | Common (training)      |
| G2    | Fault | No                | Rare but possible      |
| G3    | Normal | No               | Common (training)      |
| G4    | Normal | Yes              | Rare but possible      |

**WGA = min(Acc_G1, Acc_G2, Acc_G3, Acc_G4)**

A model that shortcuts on context will have near-zero accuracy on G2 (faults without context) and G4 (normals with context), yielding low WGA despite high average accuracy.

### 4.2 Bias-Conflicting Accuracy (Acc_bc)

Evaluates performance specifically on minority groups that *conflict* with the spurious correlation. In our case: faults without context and normals with context.

### 4.3 Deep Feature Reweighting (DFR)

From Kirichenko et al.: **Re-train only the last layer of the model on a held-out set where the spurious correlation is broken.** If performance improves dramatically, the model's features contain genuine signal but the classifier head learned the shortcut.

**Practical protocol**: Train the full model normally, then create a balanced evaluation set (faults with and without context, normals with and without context), freeze the feature extractor, and re-train just the classification head on this balanced set.

> Source: Ye et al. (2024). Spurious Correlations in Machine Learning: A Survey. https://arxiv.org/html/2402.12715v2

---

## 5. Counterfactual Evaluation: Practical Techniques

### 5.1 Counterfactual Data Generation

The core idea: **generate samples where only the suspected shortcut feature is changed, then test whether the model's prediction flips.**

From Chang et al. (CVPR 2021), "Towards Robust Classification Model by Counterfactual and Invariant Data Generation":
- **Counterfactual samples**: Change the spurious attribute while keeping the causal features fixed. If the model's prediction changes, it relied on the spurious attribute.
- **Invariant samples**: Change the causal features while keeping spurious attributes fixed. If the model's prediction does not change, it is not using causal features.
- **Adversarial game framework**: Alternate between (a) a generator that finds the model's "weakness" by generating counterfactuals, and (b) the classifier that must overcome the weakness.

**Application to our problem**:
1. Take a fault sample WITH context. Remove the context (set to null/zero). Does the model still classify it as a fault? If not, it learned the context shortcut.
2. Take a normal sample WITHOUT context. Inject synthetic context metadata. Does the model now classify it as a fault? If so, it learned the context shortcut.
3. Generate "counterfactual faults": real fault metric patterns but with context removed.
4. Generate "counterfactual normals": real normal metric patterns but with context injected.

> Source: Chang et al. (2021). Towards Robust Classification Model by Counterfactual and Invariant Data Generation. CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/papers/Chang_Towards_Robust_Classification_Model_by_Counterfactual_and_Invariant_Data_Generation_CVPR_2021_paper.pdf

### 5.2 Counterfactual Augmentation for Training

From Kaushik et al. (2020) and related work:
- **Automatically generated counterfactuals**: Identify likely causal features using statistical matching, then generate counterfactual samples by substituting causal features with their antonyms and assigning opposite labels.
- **Data rebalancing with counterfactuals**: Incorporate counterfactual data to balance label distribution and mitigate spurious correlations.

> Source: Wu et al. (2021). Robustness to Spurious Correlations in Text Classification via Automatically Generated Counterfactuals. https://arxiv.org/abs/2012.10040

### 5.3 Conceptual Counterfactual Explanations

From Abid et al. (2021): Use conceptual counterfactual explanations to identify spurious correlations. The method **"identifies spurious correlations in more than 90% of misclassified test samples"** across models trained on skewed datasets.

> Source: Abid et al. (2021). Meaningfully Debugging Model Mistakes using Conceptual Counterfactual Explanations. https://ar5iv.labs.arxiv.org/html/2106.12723

---

## 6. Invariant Risk Minimization (IRM) and Domain-Invariant Evaluation

### 6.1 IRM Framework

Arjovsky et al. (2019) introduce IRM as a learning paradigm to **"estimate invariant correlations across multiple training distributions."** The key idea: learn a data representation such that the optimal classifier is the same for all training environments.

**Core principle**: Causal relationships are invariant by definition -- they hold across different circumstances and environments. Spurious correlations vary across environments.

**IRM Objective** (simplified):
```
min_phi sum_e R_e(phi) + lambda * ||grad_w R_e(w . phi)||^2
```
where phi is the feature representation, w is the classifier, R_e is the risk in environment e, and the penalty term enforces that the optimal w is the same across all environments.

**Application to our problem**: Create "environments" with different context-label correlations:
- Environment 1: Training data as-is (faults always have context).
- Environment 2: Subset where some faults have context removed.
- Environment 3: Subset where some normals have context added.
- IRM will learn features that are predictive across ALL environments, forcing the model to ignore context (which is not invariant).

**Key quote**: "IRM detects that colors have spurious correlations with labels and uses only relevant features like digits to predict, obtaining better generalization to new test environments."

> Source: Arjovsky, M., Bottou, L., Gulczynski, D., & Lopez-Paz, D. (2019). Invariant Risk Minimization. https://arxiv.org/abs/1907.02893

### 6.2 Environment Inference for Invariant Learning (EIIL)

When explicit environment labels are unavailable, EIIL **"automates the inference of different environments within datasets."** This is relevant when you cannot manually label which samples have spurious context associations.

### 6.3 Related Invariant Learning Methods

- **StableNet**: Removes "both linear and non-linear dependencies between features" to learn stable, invariant representations.
- **Chroma-VAE**: Uses a "dual-pronged Variational Auto-Encoder" to separate shortcut features from general representations.
- **Correct-n-Contrast (CnC)**: Trains an ERM model first to "identify samples within the same class but with dissimilar spurious features," then applies contrastive learning to learn robust representations.

---

## 7. Mitigation Strategies: Complete Taxonomy

### 7.1 Dataset-Level Interventions

| Strategy | Description | Applicability |
|----------|-------------|---------------|
| **Data curation** | Remove spurious correlations during preprocessing | Remove context from dataset entirely |
| **Masking shortcut features** | Directly mask suspected shortcut features | Set context features to null/zero |
| **Foreground augmentation** | Combine foreground with random backgrounds | Combine fault metrics with random context |
| **Style transfer augmentation** | Transfer spurious attributes across classes | Transfer context metadata between fault/normal |
| **Random shortcut swapping** | Stochastically swap suspected shortcuts between classes (Lee et al.) | Randomly assign context to normals and remove from faults |
| **Latent-space augmentation** | Augment in latent space to break correlations | Generate latent representations with varied context |
| **Environment splitting** | Partition data by shortcut presence for Group DRO | Split into context-present and context-absent subsets |

### 7.2 Model-Level Approaches

| Strategy | Description | Applicability |
|----------|-------------|---------------|
| **Adversarial training** | Train model to be invariant to shortcut features | Adversarial loss penalizing context-based predictions |
| **Explanation-based regularization** | Penalize predictions explained by shortcut features | SHAP-based penalty when context features are important |
| **Group DRO** | Minimize worst-group loss across context groups | Optimize for worst accuracy across G1-G4 |
| **Sample reweighting** | Upweight minority (bias-conflicting) samples | Upweight faults-without-context and normals-with-context |
| **Feature reweighting** | Downweight shortcut-correlated features | Reduce weight of context features in representation |
| **Contrastive learning** | Learn representations invariant to spurious features | Contrast faults with/without context |
| **Causal loss regularization** | Regularize causal effects between features | Penalize causal path from context to prediction |

### 7.3 Inference-Time Mitigation

- **Test-time interventions**: Correct shortcut concepts via interventions at inference (Steinmann et al.).
- **Majority voting**: Aggregate predictions over noisy/perturbed samples (Sarkar et al.).

---

## 8. Practical Evaluation Protocol for Context-Dependent Anomaly Detection

Based on the literature synthesis, here is a recommended evaluation protocol for our specific problem (one class always has context, the other never does):

### Step 1: Diagnose the Shortcut

1. **Minimal model test**: Train a logistic regression using ONLY context features (presence/absence, metadata values). If accuracy is high, context is a shortcut.
2. **Feature importance analysis**: Run SHAP on the full model. If context features rank highest, the model is shortcutting.
3. **Early-training clustering (SPARE)**: Check if the model's early-epoch representations cluster by context rather than by genuine metric patterns.

### Step 2: Counterfactual Evaluation

4. **Context ablation test**: Remove all context features from test data. Measure accuracy drop. Large drop = shortcut reliance.
5. **Context swap test**:
   - Take fault samples, remove their context --> measure fault recall.
   - Take normal samples, inject synthetic context --> measure false positive rate.
   - A robust model should maintain performance; a shortcutting model will collapse.
6. **Context randomization test**: Randomly assign context (present/absent) to all test samples regardless of label. Measure accuracy. Should be similar to full-context accuracy for a robust model.

### Step 3: Invariant Evaluation

7. **Worst-group accuracy**: Compute accuracy across all four groups (fault+context, fault-no-context, normal+context, normal-no-context). Report WGA.
8. **Cross-environment consistency**: Create environments with different context-label correlation strengths (100%, 75%, 50%, 25%, 0%). Plot accuracy vs. correlation strength. A robust model shows flat accuracy; a shortcutting model degrades as correlation weakens.

### Step 4: Robustness Training (if shortcut confirmed)

9. **IRM training**: Create synthetic environments with varying context-label correlations and train with IRM objective.
10. **DFR evaluation**: Freeze feature extractor, re-train last layer on balanced (context-decorrelated) data. If accuracy recovers, features are good but the classifier head learned the shortcut.
11. **Adversarial context training**: Add an adversarial loss that prevents the model from predicting context presence from its learned features (domain-adversarial approach).
12. **Group DRO**: Train with worst-group loss across the four context-label groups.

---

## 9. Key Limitations and Open Problems

1. **Group label requirement**: Most methods (Group DRO, WGA) require knowing which samples have the spurious attribute. In our case, we know which samples have context, so this is less of an issue.

2. **Trade-off with average performance**: Methods that improve worst-group accuracy often reduce average accuracy. This is the "robustness-accuracy trade-off."

3. **Multiple shortcuts**: Context metadata may be one of several shortcuts. Methods should be applied iteratively for each candidate shortcut feature.

4. **Determining relevance vs. spuriousness**: As noted in the survey, "determining whether a feature is relevant or spurious is inherently complex and particularly challenging without access to commonsense knowledge." In our case, domain knowledge tells us that context *can* be informative but should not be the *sole* predictor.

5. **Partial context informativeness**: Context may carry genuine causal information (a deployment event genuinely causes faults). The goal is not to ignore context entirely but to ensure the model also learns metric-level patterns and does not rely exclusively on context.

---

## 10. Summary of Key References

| Paper | Year | Key Contribution | URL |
|-------|------|-----------------|-----|
| Geirhos et al. | 2020 | Foundational framework for shortcut learning | https://arxiv.org/abs/2004.07780 |
| Arjovsky et al. | 2019 | Invariant Risk Minimization (IRM) | https://arxiv.org/abs/1907.02893 |
| Marcinowski et al. | 2024 | Comprehensive taxonomy of shortcut detection & mitigation | https://arxiv.org/html/2412.05152v1 |
| Ye et al. | 2024 | Survey of spurious correlations in ML | https://arxiv.org/html/2402.12715v2 |
| Roelofs et al. | 2022 | Structural characterization of shortcuts via causal DAGs | https://link.springer.com/article/10.1007/s10654-022-00892-3 |
| Chang et al. | 2021 | Counterfactual & invariant data generation (CVPR) | https://openaccess.thecvf.com/content/CVPR2021/papers/Chang_Towards_Robust_Classification_Model_by_Counterfactual_and_Invariant_Data_Generation_CVPR_2021_paper.pdf |
| Kirichenko et al. | 2022 | Deep Feature Reweighting (DFR) | Referenced in Ye et al. survey |
| Sagawa et al. | 2019 | Group Distributionally Robust Optimization | Referenced in both surveys |
| Kaushik et al. / Wu et al. | 2020/2021 | Counterfactual augmentation for robustness | https://arxiv.org/abs/2012.10040 |
| Abid et al. | 2021 | Conceptual counterfactual explanations for debugging | https://ar5iv.labs.arxiv.org/html/2106.12723 |
| Springer KAIS | 2024 | Adversarial counterfactual generation for anomaly detection | https://link.springer.com/article/10.1007/s10115-024-02172-w |

---

*Research compiled: March 2026. Searches conducted across arXiv, Springer, CVPR proceedings, ACL Anthology, and general academic sources.*
