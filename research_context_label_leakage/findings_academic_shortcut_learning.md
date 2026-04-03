# Academic Literature Review: Shortcut Learning and Context-Label Leakage

## Problem Statement

In our CAAA fault-vs-normal classifier, NORMAL cases always receive context metadata
(`event_type="normal_operation"`) while FAULT cases receive empty context. A trivial
"Context Only" baseline achieves 100% accuracy by simply checking `event_active`. This
is a textbook instance of **shortcut learning**: the model exploits a spurious correlation
between metadata presence and the target label instead of learning the intended causal
features (metric patterns). This document surveys the academic literature on shortcut
learning, spurious correlations, and evaluation protocols that prevent or detect such
shortcuts.

---

## 1. Geirhos et al. (2020) -- Shortcut Learning in Deep Neural Networks

### Citation (APA)
Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., &
Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine
Intelligence*, *2*(11), 665--673. https://doi.org/10.1038/s42256-020-00257-z

### Key Insight
Shortcuts are **decision rules that perform well on standard (i.i.d.) benchmarks but fail
to transfer to more challenging testing conditions**. The paper unifies many seemingly
disparate deep learning failures (texture bias in CNNs, Clever Hans effects, adversarial
vulnerability) under a single explanatory framework: the model found a simpler statistical
regularity that correlates with the label in the training distribution but does not reflect
the intended decision rule.

**Direct relevance**: In CAAA, `event_active` is the shortcut. It correlates perfectly
with the FAULT/NORMAL label in the training distribution. A model that checks only this
feature will achieve 100% i.i.d. accuracy but has learned nothing about fault patterns.

### Concrete Evaluation Protocols Proposed
1. **Out-of-distribution (OOD) test sets**: Construct test data where the shortcut
   correlation is deliberately broken. The paper calls these "good OOD tests" that have:
   - A clear distribution shift (the spurious correlation is removed or reversed)
   - A well-defined intended solution (what the model *should* rely on)
   - The ability to expose where models learn a shortcut vs. the intended rule

2. **Controlled test conditions**: Move beyond single-metric i.i.d. evaluation. Report
   performance under multiple conditions that vary the shortcut feature independently of
   the true signal.

3. **Three categories of mitigation**:
   - *Data-level*: Augment/modify training data so the shortcut is no longer predictive
   - *Evaluation-level*: Use OOD test sets and stress tests beyond i.i.d. accuracy
   - *Model-level*: Inductive biases, architectural constraints, or regularization to
     discourage shortcut reliance

### Application to CAAA
- **Protocol**: Create a "context-ablated" test split where NORMAL samples have empty
  context (same as FAULT) and an "inverted-context" split where FAULT samples get
  `event_type="normal_operation"`. A model relying on shortcuts will collapse to random
  or worse accuracy; one relying on true features will be unaffected.

---

## 2. Sagawa et al. (2020) -- Distributionally Robust Neural Networks (Group DRO)

### Citation (APA)
Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust
neural networks for group shifts: On the importance of regularization for worst-case
generalization. In *Proceedings of the International Conference on Learning
Representations (ICLR 2020)*.

### Key Insight
When a spurious attribute correlates with labels, the dataset naturally decomposes into
**groups** defined by the cross-product of (label x spurious attribute). Models trained
with standard ERM achieve high average accuracy but **catastrophically fail on minority
groups** where the spurious correlation does not hold. The critical metric is not average
accuracy but **worst-group accuracy**.

**Canonical example (Waterbirds)**: Birds are classified as waterbird/landbird. Background
(water/land) is spuriously correlated with the label. Four groups emerge:
(waterbird, water), (waterbird, land), (landbird, land), (landbird, water). A model that
uses background as a shortcut achieves ~95% average accuracy but ~0% on "waterbird on
land" -- the minority group.

### Concrete Evaluation Protocol
1. **Define groups**: Cross the target label with the spurious attribute:
   - G1: FAULT x has_context=True (should not exist in current data)
   - G2: FAULT x has_context=False (all current FAULT cases)
   - G3: NORMAL x has_context=True (all current NORMAL cases)
   - G4: NORMAL x has_context=False (should not exist in current data)

2. **Construct balanced test set**: Ensure the test set contains examples from all four
   groups, with sufficient samples in each to accurately estimate group-specific accuracy.

3. **Report worst-group accuracy**: Instead of overall accuracy, report
   `min(accuracy_G1, accuracy_G2, accuracy_G3, accuracy_G4)`. A model relying on the
   shortcut will achieve ~0% on groups G1 and G4.

4. **Group DRO training**: Optimize for the worst-case group loss rather than average
   loss, combined with strong regularization.

### Application to CAAA
- **Protocol**: Synthetically create G1 (FAULT + normal context) and G4 (NORMAL + empty
  context) samples. Report four-group accuracy matrix. The worst-group accuracy becomes
  the primary evaluation metric.

---

## 3. Arjovsky et al. (2019) -- Invariant Risk Minimization (IRM)

### Citation (APA)
Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant risk
minimization. *arXiv preprint arXiv:1907.02893*.

### Key Insight
IRM formalizes the idea that a good representation should support a classifier that is
**simultaneously optimal across all training environments**. If a feature (like
`event_active`) is only predictive in some environments but not others, IRM will
de-emphasize it in favor of features that are **invariantly** predictive across
environments.

The core principle: learn a feature representation Phi(x) such that the optimal classifier
w on top of Phi(x) is the same for every environment e. Spurious correlations vary across
environments; causal features do not.

### Concrete Evaluation Protocol (IRMv1)
1. **Define training environments** where the spurious correlation varies:
   - Environment 1 (status quo): NORMAL has context, FAULT has no context (correlation=1.0)
   - Environment 2 (context-shuffled): Context is randomly assigned to both classes (correlation=0.0)
   - Environment 3 (inverted): FAULT has context, NORMAL has no context (correlation=-1.0)

2. **IRMv1 penalty**: Add a gradient-norm penalty to the loss that measures whether the
   classifier is simultaneously optimal across all environments:
   ```
   L_IRM = sum_e [R_e(Phi, w)] + lambda * sum_e [||grad_w R_e(w * Phi(x))|w=1.0||^2]
   ```
   This penalizes features whose optimal classifier varies across environments.

3. **Evaluation**: Test on held-out data from all environments. A model that has learned
   invariant features will perform consistently across all environments; one relying on
   shortcuts will collapse in environments where the shortcut is broken.

### Application to CAAA
- **Protocol**: Construct at least two training environments: one where context correlates
  with label (original data) and one where it does not (context randomized). Train with
  IRM penalty. Evaluate on all environments. If the model's accuracy drops significantly
  in the decorrelated environment, it was relying on the shortcut.

---

## 4. Veitch et al. (2021) -- Counterfactual Invariance to Spurious Correlations

### Citation (APA)
Veitch, V., D'Amour, A., Yadlowsky, S., & Eisenstein, J. (2021). Counterfactual
invariance to spurious correlations: Why and how to pass stress tests. In *Advances in
Neural Information Processing Systems (NeurIPS 2021)*.

### Key Insight
A model achieves **counterfactual invariance** with respect to a spurious feature if
changing that feature (while holding everything else fixed) does not change the
prediction. The paper formalizes the "stress test" idea: perturb the irrelevant part of
the input and check if the model's prediction changes. The paper also shows that the
correct regularization strategy depends on the **causal direction** between features and
labels.

### Concrete Evaluation Protocol
1. **Stress test construction**: For each test sample, create a counterfactual by
   modifying only the spurious feature:
   - Original NORMAL sample: has context -> remove context
   - Original FAULT sample: no context -> add context
   Check if predictions change. If they do, the model is not counterfactually invariant.

2. **Counterfactual invariance metric**: Measure the fraction of test samples where
   the model's prediction changes when the spurious feature is toggled:
   ```
   CI_score = 1 - (1/N) * sum_i [I(f(x_i) != f(x_i_counterfactual))]
   ```
   A CI_score of 1.0 means the model is perfectly invariant to the spurious feature.

3. **Causal-structure-aware regularization**: The paper shows that when labels cause
   features (Y -> X), invariance requires different techniques than when features cause
   labels (X -> Y).

### Application to CAAA
- **Protocol**: For every test sample, create a paired counterfactual by toggling
  `event_active` / `event_type`. Measure the prediction flip rate. A robust model should
  have a flip rate of ~0%; a shortcut-dependent model will have a flip rate near 100%.

---

## 5. Kaufman et al. (2012) -- Leakage in Data Mining

### Citation (APA)
Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining:
Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from
Data*, *6*(4), Article 15, 1--21. https://doi.org/10.1145/2382577.2382579

### Key Insight
Leakage is "the introduction of information about the data mining target that should not
be legitimately available to mine from." The paper provides a **formal taxonomy of leakage
types** and the **learn-predict separation** principle: any feature used at training time
must be obtainable at prediction time under the same conditions.

The paper identifies leakage as one of "the top ten data mining mistakes" and shows it
can arise from: (a) features derived from the target, (b) temporal contamination, (c)
aggregation across train/test, or (d) metadata that acts as a proxy for the label.

**Direct relevance**: In CAAA, `event_type` is metadata that is a perfect proxy for the
label. It is available at training time but in a degenerate way -- its value is
deterministically set by the label. This is Type (d) leakage: a metadata-as-label-proxy.

### Concrete Evaluation Protocol
1. **Learn-Predict Separation Audit**: For every feature, ask: "Would this feature have
   the same value if the label were different?" If `event_type` is always
   `"normal_operation"` for NORMAL and always empty for FAULT, it is definitionally
   leaked -- its value is a function of the label.

2. **Detection via EDA (Exploratory Data Analysis)**:
   - Compute mutual information between each feature and the target label
   - Flag features with suspiciously high MI (MI approaching 1.0)
   - Check if removing the flagged feature substantially degrades accuracy

3. **Detection via "too-good-to-be-true" baselines**:
   - Train a simple model (e.g., logistic regression, decision stump) on each feature
     individually
   - If any single feature achieves near-perfect accuracy, it is likely leaked
   - This is exactly what the "Context Only" baseline reveals in CAAA

4. **Prevention**: Either remove the leaked feature entirely, or ensure it is available
   with the same distribution for both classes.

### Application to CAAA
- **Protocol**: Run the single-feature accuracy audit. `event_active` yields 100% -> flag
  it. Either (a) remove context features entirely during evaluation, or (b) ensure both
  FAULT and NORMAL samples can have context or lack context.

---

## 6. Ribeiro et al. (2020) -- Beyond Accuracy: Behavioral Testing with CheckList

### Citation (APA)
Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond accuracy: Behavioral
testing of NLP models with CheckList. In *Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics (ACL 2020)*, 4902--4912.

### Key Insight
Standard held-out accuracy overestimates model competence because it does not distinguish
between models that have learned the right features and models that exploit spurious
patterns. CheckList proposes **behavioral testing** inspired by software engineering:
test specific capabilities of the model in isolation, not just aggregate performance.

### Concrete Evaluation Protocol (CheckList Matrix)
1. **Minimum Functionality Test (MFT)**: Test a specific capability with simple,
   unambiguous examples. For CAAA: "Can the model detect a fault from metric patterns
   alone, with no context?"

2. **Invariance Test (INV)**: Apply label-preserving perturbations and check if the
   prediction is stable. For CAAA: "If I add/remove context from a sample, does the
   prediction change? It should not."

3. **Directional Expectation Test (DIR)**: Apply perturbations that should change the
   prediction in a known direction. For CAAA: "If I inject a known fault pattern into
   a NORMAL sample (keeping its context), does the model predict FAULT?"

### Test Matrix for CAAA

| Capability              | MFT                              | INV                                | DIR                                 |
|-------------------------|----------------------------------|------------------------------------|-------------------------------------|
| Metric-based detection  | FAULT samples w/ no context      | Toggle context, prediction stable  | Add fault pattern to NORMAL -> FAULT|
| Context independence    | NORMAL samples w/ no context     | Toggle context, prediction stable  | Remove metric anomaly -> NORMAL     |
| Robustness              | FAULT w/ misleading context      | Random context, prediction stable  | Increase fault severity -> higher P |

### Application to CAAA
- **Protocol**: Implement all three test types. A model that passes MFTs (detects faults
  without context) and INVs (is invariant to context toggling) demonstrably does not rely
  on the context shortcut.

---

## 7. Wiles et al. (2022) -- A Fine-Grained Analysis on Distribution Shift

### Citation (APA)
Wiles, O., Gowal, S., Stimberg, F., Alvise-Rebuffi, S., Ktena, I., Dvijotham, K., &
Cemgil, T. (2022). A fine-grained analysis on distribution shift. In *Proceedings of the
International Conference on Learning Representations (ICLR 2022)*.

### Key Insight
The paper introduces a framework for evaluating models under **fine-grained distribution
shifts**, including spurious correlations as a specific shift type. Rather than testing
on a single held-out set, they systematically vary the degree and type of distribution
shift and measure robustness across the spectrum.

### Concrete Evaluation Protocol
1. **Decompose distribution shift into types**: spurious correlation, low-data drift,
   and unseen data shift. Evaluate models on each type independently.

2. **Vary shift magnitude**: Create test sets with different degrees of spurious
   correlation (from 100% correlated to 0% to -100% anti-correlated).

3. **Compare 19 method categories**: The paper benchmarks pretraining, augmentation,
   distributionally robust optimization, and other strategies across 85,000+ model
   configurations.

### Application to CAAA
- **Protocol**: Create a series of test sets where the context-label correlation varies:
  `{1.0, 0.8, 0.5, 0.2, 0.0, -0.5, -1.0}`. Plot model accuracy as a function of
  correlation strength. A robust model shows a flat curve; a shortcut-dependent model
  shows a steep decline as correlation weakens.

---

## 8. Jethani et al. (2023) -- Label Leakage in Explanation Methods

### Citation (APA)
Jethani, N., Saporta, A., & Ranganath, R. (2023). Don't be fooled: Label leakage in
explanation methods and the importance of their quantitative evaluation. In *Proceedings
of the International Conference on Artificial Intelligence and Statistics (AISTATS 2023)*.

### Key Insight
Class-dependent explanation methods (SHAP, LIME, Grad-CAM) can "leak" information about
the selected class, making that class appear more likely than it is. The paper advocates
for **distribution-aware** explanation methods and rigorous quantitative evaluation of
whether explanations actually reflect the model's true reasoning.

### Concrete Evaluation Protocol
1. **Distribution-aware attribution**: Use methods like SHAP-KL that favor explanations
   keeping the label's distribution close to its distribution given all features.

2. **Quantitative evaluation across multiple datasets**: Test explanation methods on
   datasets of different modalities (images, biosignals, text) to ensure they generalize.

### Application to CAAA
- **Protocol**: When using SHAP or similar methods to explain CAAA predictions, verify
  that high attribution to `event_active` is flagged as a leakage concern, not
  celebrated as a "useful feature." Compare class-dependent vs. distribution-aware
  attributions.

---

## Synthesis: Recommended Evaluation Protocol for CAAA

Drawing from all surveyed papers, here is a concrete multi-level evaluation protocol that
would prevent the context-label shortcut from inflating performance:

### Level 1: Leakage Detection (Kaufman et al., Jethani et al.)
- [ ] Compute per-feature mutual information with the label
- [ ] Train single-feature baselines for every input feature
- [ ] Flag any feature achieving >90% single-feature accuracy as a leakage suspect
- [ ] Run SHAP analysis and check if `event_active` dominates feature importance

### Level 2: Behavioral Testing (Ribeiro et al.)
- [ ] **MFT**: Evaluate on context-ablated test set (all context features zeroed)
- [ ] **INV**: For each sample, toggle context and measure prediction stability
- [ ] **DIR**: Inject known fault patterns into NORMAL samples and verify detection

### Level 3: Group Robustness (Sagawa et al.)
- [ ] Construct the 4-group test set: (FAULT/NORMAL) x (has_context/no_context)
- [ ] Report accuracy for each of the 4 groups
- [ ] Use **worst-group accuracy** as the primary metric
- [ ] If worst-group accuracy << average accuracy, the model uses the shortcut

### Level 4: Counterfactual Invariance (Veitch et al.)
- [ ] For every test sample, create a counterfactual by toggling context features
- [ ] Compute prediction flip rate
- [ ] Target: flip rate < 5% (model is robust to context changes)

### Level 5: Multi-Environment Evaluation (Arjovsky et al., Wiles et al.)
- [ ] Create environments with varying context-label correlation: {1.0, 0.5, 0.0, -0.5}
- [ ] Evaluate on each environment
- [ ] Plot accuracy vs. correlation strength
- [ ] A robust model maintains accuracy across all environments

### Level 6: OOD Generalization (Geirhos et al.)
- [ ] Design an OOD test set where context correlation is reversed
- [ ] Report both i.i.d. and OOD accuracy
- [ ] The gap between them quantifies shortcut reliance

---

## Summary Table

| Paper | Key Concept | Primary Metric | Protocol Type |
|-------|-------------|----------------|---------------|
| Geirhos et al. (2020) | Shortcut learning definition | OOD accuracy | OOD test sets |
| Sagawa et al. (2020) | Group DRO | Worst-group accuracy | Group-stratified evaluation |
| Arjovsky et al. (2019) | Invariant Risk Minimization | Cross-environment accuracy | Multi-environment training + eval |
| Veitch et al. (2021) | Counterfactual invariance | Prediction flip rate | Counterfactual stress tests |
| Kaufman et al. (2012) | Data leakage taxonomy | Single-feature accuracy | Learn-predict separation audit |
| Ribeiro et al. (2020) | Behavioral testing (CheckList) | MFT/INV/DIR pass rates | Capability-based test matrix |
| Wiles et al. (2022) | Fine-grained distribution shift | Accuracy vs. correlation curve | Parametric shift evaluation |
| Jethani et al. (2023) | Label leakage in explanations | Distribution-aware SHAP | Attribution audit |

---

## Sources

- [Geirhos et al. (2020) - Nature Machine Intelligence](https://www.nature.com/articles/s42256-020-00257-z)
- [Geirhos et al. (2020) - arXiv](https://arxiv.org/abs/2004.07780)
- [Geirhos et al. - GitHub](https://github.com/rgeirhos/shortcut-perspective)
- [Sagawa et al. (2020) - arXiv](https://arxiv.org/abs/1911.08731)
- [Sagawa et al. - GitHub (Group DRO)](https://github.com/kohpangwei/group_DRO)
- [Arjovsky et al. (2019) - arXiv](https://arxiv.org/abs/1907.02893)
- [Veitch et al. (2021) - arXiv](https://arxiv.org/abs/2106.00545)
- [Kaufman et al. (2012) - ACM TKDD](https://dl.acm.org/doi/10.1145/2382577.2382579)
- [Ribeiro et al. (2020) - ACL Anthology](https://aclanthology.org/2020.acl-main.442/)
- [Ribeiro et al. - GitHub (CheckList)](https://github.com/marcotcr/checklist)
- [Wiles et al. (2022) - OpenReview / ICLR](https://openreview.net/forum?id=Dl4LetuLdyK)
- [Wiles et al. - GitHub (Distribution Shift Framework)](https://github.com/google-deepmind/distribution_shift_framework)
- [Jethani et al. (2023) - arXiv](https://arxiv.org/abs/2302.12893)
- [Spurious Correlations Survey (2024) - arXiv](https://arxiv.org/html/2402.12715v2)
