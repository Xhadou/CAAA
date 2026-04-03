# Label Leakage and Feature Leakage Prevention in Machine Learning

## Research Summary

Date compiled: 2026-03-30
Context: Time-series classification, anomaly detection, and general ML evaluation methodology.

---

## 1. Standard Definitions

### 1.1 Data Leakage (General)

Data leakage (also called target leakage) occurs when information that would not be available at prediction time is used during model training. It causes a model to appear accurate during development but yield inaccurate results in deployment.

Two primary categories:
- **Target leakage (column-wise)**: Features that are proxies for, deterministically related to, or directly encode the label.
- **Train-test contamination (row-wise)**: Information from validation/test examples bleeding into training through shared preprocessing, duplicates, or temporal violations.

Source: [Leakage (machine learning) - Wikipedia](https://en.wikipedia.org/wiki/Leakage_(machine_learning))

### 1.2 Label Leakage (Formal)

Label leakage occurs when a feature selection or model input mechanism encodes information about the prediction target that should not be accessible. Formally (from Hedstrom et al. / Jethani et al.):

> A selector has no label leakage when "conditioning the label distribution on the selection of features does not change the label distribution."

Mathematically: `p(y | x[s_in]) = p(y | x[s_in], h[s_in]=1, h[s_ex]=0, selector)`

This means the act of selecting which features to use should not itself carry information about the label.

Source: [Local Feature Selection without Label or Feature Leakage](https://arxiv.org/html/2407.11778v1)

### 1.3 Feature Leakage (Formal)

Feature leakage occurs when a selection mask encodes information about the values of non-selected features. Formally:

> "Conditioning the feature distribution on the selection of features does not change the feature distribution" for non-selected features.

Mathematically: `p(x[s_ex] | x[s_in]) = p(x[s_ex] | x[s_in], h[s_in]=1, h[s_ex]=0, selector)`

**Critical relationship**: Feature leakage implies label leakage when features are correlated with labels. If the selection mask reveals information about excluded features, and those features correlate with the label, then the mask indirectly reveals label information.

Source: [Local Feature Selection without Label or Feature Leakage](https://arxiv.org/html/2407.11778v1)

### 1.4 Label Leakage in Explanation Methods

Class-dependent feature attribution methods (SHAP, LIME, Grad-CAM) can "leak" information about the selected class, making that class appear more likely than it actually is. Distribution-aware methods are proposed as alternatives that keep the label distribution close to its unconditional distribution.

Source: [Don't be fooled: label leakage in explanation methods](https://arxiv.org/abs/2302.12893)

---

## 2. Features Correlated with Labels by Construction

### 2.1 The Core Problem

In many ML pipelines -- particularly in anomaly detection and root cause analysis -- auxiliary features may be correlated with labels by the nature of how the data is generated. Examples include:

- **Fault type indicators** embedded in feature names or metadata columns
- **Temporal proximity** to injected anomalies in synthetic benchmarks
- **Process-generated features** in healthcare where diagnostic workup features are collected because of suspected outcomes (Guo et al., 2023)

> "Data generated during iterative diagnostic processes complicates the evaluation of label leakage... informative predictive model inputs may be collected to improve model performance and surrogate outcomes may be captured causing label leakage."

Source: [A framework for understanding label leakage in ML for health care (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/)

### 2.2 Framework for Evaluating Correlated Features

The Guo et al. (2023) healthcare framework proposes evaluating whether features constitute leakage by considering three dimensions:

1. **Cadence**: When is the feature available relative to the prediction time? Features collected after the outcome event constitute temporal leakage.
2. **Perspective**: From whose viewpoint is the feature legitimate? A feature may be valid for one use case but leaky for another.
3. **Applicability**: Would the feature be available in the intended deployment setting?

Source: [A framework for understanding label leakage in ML for health care (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/)

### 2.3 Connectome-Based ML Example

Data leakage inflates prediction performance in connectome-based machine learning models. When features are selected or preprocessed using information from the full dataset (including test samples), performance estimates are systematically inflated.

Source: [Data leakage inflates prediction performance in connectome-based ML models (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901797/)

---

## 3. Prevention Techniques

### 3.1 Strict Data Splitting

- **Split first, preprocess later**: Create train/validation/test sets before any transformations.
- **Fit on train only**: All preprocessing (scaling, imputation, encoding, feature selection) must be fit exclusively on training data.
- **Three-way splits**: Training, validation (for hyperparameter tuning), and test (for final evaluation).

Source: [Data Leakage in Machine Learning - MachineLearningMastery.com](https://machinelearningmastery.com/data-leakage-machine-learning/)

### 3.2 Temporal Validation for Time Series

For time-series data, random splitting (e.g., standard K-Fold) is a critical error because it allows the model to train on future data to predict the past:

- **Chronological splits**: Always split by time, using past data to predict future outcomes.
- **Time-based cutoffs**: Establish a cutoff time; no information after the cutoff is used for prediction.
- **Walk-forward validation**: Expanding or sliding window cross-validation that respects temporal ordering.
- **Embargo periods**: Buffer zones between training and test windows to prevent subtle temporal leakage through lagged features.

Source: [Avoiding Data Leakage in Timeseries 101 (Towards Data Science)](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/)

### 3.3 Pipeline-Based Prevention

Scikit-learn-style pipelines ensure that preprocessing steps are encapsulated:

- Transformations (normalization, imputation, encoding) are fitted only within each training fold.
- The pipeline prevents information from the test fold from leaking into the training process.
- This is especially critical for feature selection, where selecting features based on the full dataset is a common leakage source.

Source: [Preventing Data Leakage in Feature Engineering (dotData)](https://dotdata.com/blog/preventing-data-leakage-in-feature-engineering-strategies-and-solutions/)

### 3.4 SUWR (Sequential Unmasking Without Reversion)

A method for leakage-free local feature selection:

- Features are revealed one at a time in a sequential decision process.
- Each selection step conditions only on previously-selected features.
- The selector never has access to unselected feature values or the label.
- Prevents both feature leakage and label leakage by construction.

Alternative approach: Linear programming solutions can guarantee no leakage but require complete distributional knowledge and finite feature spaces (impractical for real data).

Source: [Local Feature Selection without Label or Feature Leakage](https://arxiv.org/html/2407.11778v1)

### 3.5 Partial Label Corruption and Randomization Tests

While not prominently named in literature as a single technique, the underlying principles appear in several methodologies:

- **Label permutation tests (label shuffling)**: Train a model on randomly shuffled labels to establish a baseline. If the model with real labels does not significantly outperform the shuffled-label model, the features may not carry genuine predictive signal (or leakage may be inflating results).
- **Feature ablation / knockout tests**: Systematically remove features suspected of leakage and measure performance degradation. If removing a single feature causes dramatic performance drops, it may be a proxy for the label.
- **Noise injection / corruption**: Adding noise to labels or features to test robustness. If performance is unaffected by label noise, the model may be relying on leaked metadata rather than genuine patterns.
- **Context randomization**: In settings where context (metadata, auxiliary signals) may carry label information, randomizing context features while preserving primary features can test whether the model relies on legitimate signal vs. contextual leakage.

### 3.6 Stratified Evaluation

- **Stratified cross-validation**: Ensures each fold has proportional representation of each class, preventing evaluation bias.
- **Group-aware splits**: When samples belong to groups (e.g., same patient, same system instance), all samples from a group must be in the same fold to prevent information leakage across related observations.
- **Per-class and per-group performance reporting**: Aggregated metrics can mask leakage that affects only certain subgroups.

---

## 4. Kaggle Competition Best Practices

### 4.1 Common Leakage Sources in Competitions

- Row IDs or timestamps that encode ordering information correlated with the target.
- Image EXIF metadata containing labels or class information.
- File naming conventions that reveal class membership.
- Duplicated or near-duplicate samples across train and test sets.
- External data that overlaps with the private test set.

### 4.2 Prevention Guidelines

1. **Drop duplicates** before any cross-validation.
2. **Never include external data** in validation sets.
3. **Three-way splits**: Train / validation (hyperparameter tuning) / test (final evaluation).
4. **Time-based splits** for temporal data -- never random splits.
5. **Feature importance auditing**: Inspect top features for suspiciously strong predictors that would not be available at prediction time.
6. **Shake-up awareness**: Large discrepancies between public and private leaderboard scores often indicate leakage exploitation.

Source: [How to Prevent Data Leakage (Kaggle)](https://www.kaggle.com/code/kaanboke/how-to-prevent-the-data-leakage)
Source: [Data Leakage (Kaggle)](https://www.kaggle.com/code/dansbecker/data-leakage)

---

## 5. Anomaly Detection Specific Considerations

### 5.1 Unique Challenges

Anomaly detection faces distinct leakage risks:

- **Threshold selection on test data**: Choosing anomaly thresholds using the test set leaks information about the optimal operating point.
- **Normalization across anomalous and normal data**: If the scaler is fit on data that includes anomalies, the normalization encodes information about anomaly distribution.
- **Synthetic anomaly benchmarks**: When anomalies are injected synthetically, features derived from the injection mechanism (e.g., fault type, injection parameters) can leak into the model.
- **Label-dependent preprocessing**: Any preprocessing step that differs for anomalous vs. normal samples constitutes leakage.

### 5.2 Best Practices for Anomaly Detection Evaluation

1. **Train on normal data only** (for semi-supervised approaches), ensuring no anomalous samples contaminate training.
2. **Fix thresholds before test evaluation**: Use validation set or domain knowledge to set thresholds.
3. **Report multiple metrics**: Precision, recall, F1, AUC-ROC, and AUC-PR to show robustness rather than a single metric that may be optimized through leakage.
4. **Use temporal holdout** for time-series anomaly detection: training data from early periods, test data from later periods.
5. **Ablate auxiliary features**: Remove metadata or context features to verify the model uses genuine signal.

---

## 6. Fairness and Metadata Leakage

### 6.1 Protected Attribute Leakage

Features can act as proxies for protected attributes (race, gender, age), encoding bias:

- Even if protected attributes are removed, correlated features (zip code, name patterns) may leak the same information.
- This is structurally identical to label leakage: a feature that should not influence the prediction carries information about a sensitive variable.

### 6.2 Mitigation Strategies

- **Fairness auditing**: Evaluate model performance across protected subgroups.
- **Disparate impact analysis**: Measure whether outcomes differ systematically across groups.
- **Feature debiasing**: Remove or decorrelate features that serve as proxies for protected attributes.

Source: [Trustworthy AI: Explainability, Bias, Fairness (Carpentries)](https://carpentries-incubator.github.io/fair-explainable-ml/instructor/3-model-eval-and-fairness.html)
Source: [Survey on Machine Learning Biases and Mitigation Techniques (MDPI)](https://www.mdpi.com/2673-6470/4/1/1)

---

## 7. Vertical Federated Learning Context (2024 Survey)

Label leakage is also studied in vertical federated learning (VFL), where passive parties attempt to infer labels held by the active party. The IJCAI 2024 survey proposes taxonomies for both label inference attacks and defenses, demonstrating that leakage is a concern at the system architecture level, not just the data preprocessing level.

Source: [Label Leakage in Vertical Federated Learning: A Survey (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/902)

---

## 8. Summary of Key Takeaways

| Leakage Type | Definition | Primary Prevention |
|---|---|---|
| Target / Label Leakage | Features that encode or proxy for the label | Feature auditing, ablation tests, causal analysis |
| Train-Test Contamination | Test information in training pipeline | Strict splitting, pipeline encapsulation |
| Temporal Leakage | Future information in historical predictions | Chronological splits, embargo periods |
| Feature Leakage | Selection mask reveals excluded feature values | SUWR, distribution-aware selection |
| Metadata Leakage | Auxiliary data (filenames, IDs, context) encoding labels | Metadata stripping, context randomization |
| Fairness Leakage | Features proxying protected attributes | Fairness auditing, feature debiasing |

### Core Principles

1. **Split before you preprocess** -- temporal or stratified, never random for time series.
2. **Fit on train only** -- all transformations, selections, and thresholds.
3. **Audit feature importance** -- suspiciously strong features may be proxies.
4. **Ablate and randomize** -- remove suspected leaky features and verify performance holds.
5. **Report comprehensively** -- multiple metrics, per-group, with confidence intervals.
6. **Use label permutation tests** -- establish that signal exceeds what randomized labels produce.

---

## Sources

- [Leakage (machine learning) - Wikipedia](https://en.wikipedia.org/wiki/Leakage_(machine_learning))
- [What is Data Leakage in Machine Learning? - IBM](https://www.ibm.com/think/topics/data-leakage-machine-learning)
- [A framework for understanding label leakage in ML for health care (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/)
- [Local Feature Selection without Label or Feature Leakage (arXiv)](https://arxiv.org/html/2407.11778v1)
- [Don't be fooled: label leakage in explanation methods (arXiv)](https://arxiv.org/abs/2302.12893)
- [Data leakage inflates prediction performance in connectome-based ML models (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901797/)
- [Data Leakage in Machine Learning - MachineLearningMastery.com](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Avoiding Data Leakage in Timeseries 101 (Towards Data Science)](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/)
- [Preventing Data Leakage in Feature Engineering (dotData)](https://dotdata.com/blog/preventing-data-leakage-in-feature-engineering-strategies-and-solutions/)
- [How to Prevent Data Leakage (Kaggle)](https://www.kaggle.com/code/kaanboke/how-to-prevent-the-data-leakage)
- [Data Leakage (Kaggle)](https://www.kaggle.com/code/dansbecker/data-leakage)
- [Label Leakage in Vertical Federated Learning: A Survey (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/902)
- [Data Leakage in Machine Learning (Built In)](https://builtin.com/machine-learning/data-leakage)
- [Trustworthy AI: Explainability, Bias, Fairness (Carpentries)](https://carpentries-incubator.github.io/fair-explainable-ml/instructor/3-model-eval-and-fairness.html)
- [Survey on Machine Learning Biases and Mitigation Techniques (MDPI)](https://www.mdpi.com/2673-6470/4/1/1)
