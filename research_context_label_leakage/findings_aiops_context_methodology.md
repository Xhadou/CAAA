# Research Findings: How AIOps Papers Handle Operational Context Features in Evaluation

## Research Question

When operational context metadata (scheduled events, deployments, time-of-day patterns) is available for one class but not another in anomaly/fault detection, how do researchers prevent it from becoming a trivial label proxy?

---

## 1. Key Finding: Most AIOps Papers Do NOT Use Operational Context Features

The most striking finding from this research is that the major microservice fault detection and root cause analysis papers **deliberately avoid operational context features entirely**. They operate exclusively on raw telemetry (metrics, logs, traces) without incorporating deployment metadata, scheduled event flags, time-of-day encodings, or maintenance window indicators.

This is itself a methodological choice that sidesteps the label leakage problem -- by never introducing context metadata, they never face the proxy problem.

### Papers Surveyed

| Paper | Venue | Features Used | Operational Context? |
|-------|-------|---------------|---------------------|
| DejaVu | FSE 2022 | Time-series metrics + failure dependency graph | **No** |
| BARO | FSE 2024 | Multivariate time-series metrics only | **No** |
| RCAEval benchmark | WWW 2025 / ASE 2024 | Metrics, logs, traces | **No** |
| Chain-of-Event | FSE 2024 | Logs + causal event chains | **No** |
| Night's Watch | Applied Sciences 2024 | Time-series metrics + LSTM | **No** |
| Lumos (Microsoft) | KDD 2020 | A/B testing metric regressions | Implicit (A/B framework handles it) |

---

## 2. DejaVu (FSE 2022) -- Feature and Evaluation Methodology

**Paper**: "Actionable and Interpretable Fault Localization for Recurring Failures in Online Service Systems"
**Source**: [GitHub](https://github.com/NetManAIOps/DejaVu) | [PDF](https://netman.aiops.org/wp-content/uploads/2022/11/DejaVu-paper.pdf)

### Features
- Extracts **time-series features** from a Failure Dependency Graph (FDG) using GRU-based feature extractors.
- Features are aggregated from raw metric data (CPU, memory, latency, etc.) per service node.
- **No deployment events, scheduled events, or time-of-day features** are included.

### Evaluation Protocol
- Supervised learning: trained on labeled historical failures.
- Uses `faults.csv` with ground-truth labels and `metrics.csv` with time-series data.
- Performs **node classification** on the FDG: each node is classified as faulty or normal.
- Class balancing is implemented (`bal=True` parameter) to address imbalance between faulty and non-faulty nodes.

### Context Assignment
- The paper frames the problem as **recurring failure localization** -- it only activates when a failure is already detected.
- There is no "normal vs anomalous" binary classification in the traditional sense; the model localizes which component is the root cause during an already-identified failure.
- This design inherently avoids the label proxy problem because context (the fact that a failure is occurring) is shared across all candidate nodes.

### Key Insight for CAAA
DejaVu avoids the context-as-label-proxy problem by **conditioning on failure already existing** -- it does not try to distinguish "normal operation" from "failure" using context. Instead, all nodes are evaluated during a known failure event, and the model determines *which* node is responsible.

---

## 3. BARO (FSE 2024) -- Unsupervised Approach

**Paper**: "Robust Root Cause Analysis for Microservices via Multivariate Bayesian Online Change Point Detection"
**Source**: [arXiv](https://arxiv.org/abs/2405.09330) | [GitHub](https://github.com/phamquiluan/baro)

### Features
- Operates on **multivariate time-series metrics only** (`<service>_<metric>` format, e.g., `cart_cpu`, `checkoutservice_latency`).
- **No operational context features** of any kind: no deployment flags, no scheduled event markers, no time-of-day encoding.
- Does not require service call graphs or causal graphs.

### Evaluation Protocol
- **Fully unsupervised**: eliminates the need for labeled data entirely.
- Uses Bayesian Online Change Point Detection to detect anomalies by identifying distributional shifts in metric time series.
- After detecting an anomaly window, applies nonparametric statistical hypothesis testing to rank root cause candidates.
- Sensitivity analysis varies `t_bias` parameter from -40 to +40 to test robustness.

### Context Assignment
- Because BARO is unsupervised, there is no class labeling at training time.
- The change point detection implicitly defines "before" (normal) and "after" (anomalous) periods within a single continuous time series.
- **Both periods come from the same operational context** -- the only difference is the injected fault.
- This eliminates the label proxy problem because context metadata is never a feature.

### Key Insight for CAAA
BARO's unsupervised change-point approach is a clean way to avoid the proxy problem: the "normal" and "anomalous" classes are derived from the same continuous time series, so any contextual metadata (time-of-day, deployment status) would be identical for both the pre-fault and post-fault windows.

---

## 4. RCAEval Benchmark (WWW 2025 / ASE 2024)

**Paper**: "RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data"
**Source**: [arXiv](https://arxiv.org/abs/2412.17015) | [GitHub](https://github.com/phamquiluan/RCAEval)

### Dataset Structure
- 735 failure cases across three microservice systems (Online Boutique, Sock Shop, Train Ticket).
- Three dataset tiers:
  - **RE1** (375 cases): Metrics only, 49-212 metrics per system, 5 fault types, 5 repetitions per fault-service pair.
  - **RE2** (270 cases): Multi-source (metrics + logs + traces), 77-376 metrics, 6 fault types, 3 repetitions.
  - **RE3** (90 cases): Multi-source, code-level faults.
- Fault types: CPU hog, memory leak, disk stress, network delay, packet loss, socket fault.

### Evaluation Protocol
- **Normal period**: 10 minutes of normal operation to collect baseline telemetry.
- **Fault period**: Fault injected into a randomly selected service; abnormal telemetry collected.
- Faults injected using standard tools: `stress-ng` (resource), `tc` (network), source code modifications (code-level).
- Evaluation metrics: AC@k (accuracy at top-k) and Avg@k (average rank in top-k).

### Context Assignment: Critical Observation
- **No operational context features** are collected or evaluated.
- The benchmark provides raw telemetry only -- timestamps, metric values, log lines, trace spans.
- **No discussion of label leakage, metadata proxies, or evaluation bias** appears in the paper.
- The normal-to-fault transition occurs within a single continuous experiment, so time-of-day and deployment status are held constant within each case.

### Key Insight for CAAA
The RCAEval benchmark implicitly avoids context leakage by running each fault injection as a short, self-contained experiment where environmental conditions are constant. However, this also means the benchmark does not test the real-world scenario where context features ARE available and must be properly handled. **This is a gap in the literature.**

---

## 5. Lumos (Microsoft, KDD 2020)

**Paper**: "Lumos: A Library for Diagnosing Metric Regressions in Web-Scale Applications"
**Source**: [arXiv](https://arxiv.org/abs/2006.12793) | [ACM DL](https://dl.acm.org/doi/10.1145/3394486.3403306)

### Unique Approach to Context
- Lumos is fundamentally different from other surveyed tools: it uses **A/B testing principles** to diagnose metric regressions.
- By comparing treatment and control groups, Lumos naturally controls for confounding context variables.
- Deployed across Microsoft's Real-Time Communication applications (Skype, Teams).
- Enabled teams to detect hundreds of real changes and reject thousands of false alarms.

### Context Assignment
- The A/B testing framework inherently provides the **counterfactual**: "what would this metric look like without the change?"
- Context features (time-of-day, user population shifts, telemetry processing bias) are controlled for by the experimental design itself.
- This is the **only surveyed system that explicitly handles operational context**, but it does so through experimental design rather than feature engineering.

### Key Insight for CAAA
Lumos demonstrates that the **gold standard** for handling operational context is counterfactual reasoning through controlled experiments. When you cannot run A/B tests, you need synthetic or statistical methods to approximate this counterfactual.

---

## 6. The Gap: No Paper Directly Addresses the Label Proxy Problem

### What the literature does NOT cover

After surveying the major AIOps/microservice fault detection papers from 2020-2024, a significant gap emerges:

1. **No paper explicitly discusses the risk of context metadata becoming a label proxy.** This topic is absent from DejaVu, BARO, RCAEval, Chain-of-Event, and other surveyed works.

2. **No paper assigns operational context to both normal and anomalous classes.** Context features are simply excluded from the feature set entirely.

3. **No paper provides guidance on what percentage of each class should receive context vs. no context.** This experimental design question is unaddressed.

4. **No paper discusses calibration of context as a useful-but-not-perfect signal.** The literature treats context as either perfectly controlled (Lumos/A/B testing) or completely ignored (all others).

### Why this gap exists

The surveyed papers avoid the problem through three mechanisms:

| Strategy | Papers | How It Works |
|----------|--------|-------------|
| **Context exclusion** | BARO, DejaVu, RCAEval, Chain-of-Event | Simply do not use context features; operate on raw telemetry only |
| **Same-context comparison** | BARO, RCAEval | Normal and anomalous data come from the same short experiment, so context is held constant |
| **Experimental control** | Lumos | Use A/B testing to explicitly control for context |

---

## 7. Implications for CAAA: How to Handle Context Without Leakage

Based on the literature survey, here are principled approaches to prevent context metadata from becoming a trivial label proxy:

### Approach A: Context Exclusion (Literature Standard)
- Simply do not use operational context as model features.
- **Pro**: Eliminates the problem entirely.
- **Con**: Loses potentially valuable signal.

### Approach B: Context Held Constant (RCAEval/BARO Style)
- Structure evaluation so that normal and anomalous examples share the same context.
- Each experiment runs continuously: normal phase then fault phase, within the same time window.
- **Pro**: Context cannot distinguish classes because it is identical for both.
- **Con**: Does not reflect real-world deployment where different faults occur at different times.

### Approach C: Balanced Context Assignment (Novel -- Not Found in Literature)
- Assign operational context (e.g., "scheduled deployment," "peak hours") to **both** normal and anomalous examples.
- Ensure each context type appears in both classes with similar frequency.
- For example: 30% of normal examples and 30% of anomalous examples are marked "during deployment."
- **Pro**: Context is a useful but non-deterministic signal.
- **Con**: Requires careful calibration; no existing literature provides guidance on the right proportions.

### Approach D: Counterfactual Evaluation (Lumos-Inspired)
- For each anomalous case with context, generate a counterfactual: "what would the normal case look like with the same context?"
- For each normal case, generate: "what would this look like if an anomaly occurred during the same context?"
- Evaluate model performance on matched pairs.
- **Pro**: Cleanest causal approach.
- **Con**: Requires generative modeling or careful data collection.

### Approach E: Ablation Over Context Features
- Train models with and without context features.
- If adding context dramatically increases apparent performance, that is a red flag for label leakage.
- Report both results to quantify the contribution of context vs. the contribution of actual anomaly patterns.
- **Pro**: Transparent; identifies the problem.
- **Con**: Does not solve it, only measures it.

---

## 8. Recommended Evaluation Protocol for CAAA

Based on the gap analysis, the following protocol would be novel and methodologically sound:

1. **Baseline without context**: Train and evaluate using only telemetry features (metrics, logs, traces). This matches the literature standard.

2. **Context-balanced evaluation**: When adding context features:
   - Ensure context metadata appears in **both classes** (normal and anomalous).
   - Use stratified splits: within each context stratum (e.g., "during deployment"), maintain the same anomaly base rate.
   - Report per-context-stratum performance.

3. **Mutual information check**: Measure mutual information between each context feature and the label. If MI is very high (close to the label entropy), the feature is a near-perfect proxy and should be treated with caution.

4. **Ablation requirement**: Report performance with and without context features. The delta quantifies how much the model relies on context vs. anomaly patterns.

5. **Adversarial evaluation**: Test whether a model trained ONLY on context features (no telemetry) can achieve above-chance performance. If so, the evaluation has a leakage problem.

---

## Sources

- [DejaVu GitHub Repository](https://github.com/NetManAIOps/DejaVu)
- [DejaVu Paper PDF](https://netman.aiops.org/wp-content/uploads/2022/11/DejaVu-paper.pdf)
- [BARO arXiv](https://arxiv.org/abs/2405.09330)
- [BARO GitHub Repository](https://github.com/phamquiluan/baro)
- [BARO ACM DL](https://dl.acm.org/doi/10.1145/3660805)
- [RCAEval arXiv](https://arxiv.org/abs/2412.17015)
- [RCAEval GitHub Repository](https://github.com/phamquiluan/RCAEval)
- [RCAEval Full HTML](https://arxiv.org/html/2412.17015v1)
- [Chain-of-Event FSE 2024 PDF](https://netman.aiops.org/wp-content/uploads/2024/07/Chain-of-Event_Interpretable-Root-Cause-Analysis-for-MicroservicesFSE24-Camera-Ready.pdf)
- [Lumos arXiv](https://arxiv.org/abs/2006.12793)
- [Lumos ACM DL](https://dl.acm.org/doi/10.1145/3394486.3403306)
- [Night's Watch Algorithm (Context-Aware Anomaly Detection)](https://www.mdpi.com/2076-3417/15/23/12762)
- [Context-Aware Anomaly Detection in Microservices Using GCN](https://www.etasr.com/index.php/ETASR/article/view/13590)
- [Anomaly Detection in Large-Scale Cloud Systems: Industry Case](https://arxiv.org/pdf/2411.09047)

---

*Research conducted: 2026-03-30*
*For project: CAAA (Context-Aware Anomaly Attribution)*
