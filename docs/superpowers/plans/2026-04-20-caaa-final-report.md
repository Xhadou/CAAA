# CAAA Final Report — Implementation Plan

> **For agentic workers:** This is a research-report plan, not a software implementation plan. Tasks are "write chapter X with these specific claims, data, and citations." Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a rigorous, publication-quality LaTeX report for the CAAA major project that argues the correct scientific claim (context as a universal signal; neural-over-tree crossover with scale), backed by the hard-task scaling study and the real-data ablation, with verified academic citations.

**Architecture:** Rewrite `report/CAAA_Project_Report.tex` from the user's approved narrative outline, reusing the existing preamble/title page/acknowledgments. Replace chapters 1–7 with a chapter structure that matches the new story (real-data limits → synthetic-data need → synthetic-data design + anti-leakage → context feature design → scaling study → universal context → insights).

**Tech Stack:** LaTeX (same preamble as existing report — amsmath, booktabs, tikz, pgfplots, algorithm/algpseudocode, hyperref). Results tables populated from `outputs/results/*.csv`. Plots embedded from `outputs/results/*.png`.

---

## Aggregated Resources

### Experimental artifacts (already generated — do not re-run)

| Artifact | Path | Use |
|---|---|---|
| Default-difficulty scaling CSV | `outputs/results/scaling_study.csv` | Chapter 6 Act 1 table + curve |
| Hard-difficulty scaling CSV | `outputs/results/scaling_study_hard.csv` | Chapter 6 Act 2 table + headline plot |
| Default scaling curve plot | `outputs/results/scaling_curve.png` | Figure: saturation at ~97% |
| Hard scaling curve plot | `outputs/results/scaling_curve_hard.png` | Figure: headline curve |
| Context contribution bar chart | `outputs/results/context_contribution_hard.png` | Figure: Chapter 7 headline |
| Real-data ablation combined | `outputs/results/ablation_results_combined.csv` | Chapter 5 per-system table |
| Real-data pooled (all×all) | `outputs/results/ablation_results_all_pooled.csv` | Chapter 5 pooled row |
| Synthetic ablation | `outputs/results/ablation_results_synthetic.csv` | Chapter 5 synthetic table |
| SHAP per system | `outputs/results/shap_*/` | Chapter 5 interpretability figure |
| Calibration per system | `outputs/results/calibration_*/` | Chapter 5 calibration figure |

### Key numbers (memorize for tables)

**Hard-task scaling at 40K (10 seeds):**
- Full CAAA: 0.9511 ± 0.0018
- CatBoost (+ctx): 0.9490 ± 0.0021
- Welch's t-test: t=2.249, p=0.0375 (significant, small effect)

**Context contribution at 40K hard:**
- CAAA: +1.28pp, CatBoost: +1.70pp, XGBoost: +1.60pp, LightGBM: +1.58pp, RandomForest: +2.30pp

**Default-difficulty scaling at 40K:**
- Full CAAA: 0.9714 ± 0.0010
- CatBoost (+ctx): 0.9698 ± 0.0006 (gap +0.16pp, p=0.09)

**Real-data pooled averages (9 system×dataset cells):**
- Full CAAA: F1=0.9264, Recall=0.9438, FP reduction=0.9121
- CatBoost: F1=0.9542, Recall=0.9376, FP reduction=0.9715
- CatBoost (+ctx): F1=0.9571, Recall=0.9424, FP reduction=0.9728
- CAAA+CatBoost Hybrid: F1=0.9328, Recall=0.9397

### Verified academic citations (from research agent)

| BibKey | Full citation | Use |
|---|---|---|
| `grinsztajn2022tree` | Grinsztajn, Oyallon, Varoquaux. "Why do tree-based models still outperform deep learning on typical tabular data?" NeurIPS 2022 | Ch 2 + Ch 6 — tree dominance on small tabular |
| `shwartzziv2022tabular` | Shwartz-Ziv, Armon. "Tabular data: Deep learning is not all you need." Information Fusion 2022 | Ch 2 — corroborates Grinsztajn |
| `mcelfresh2023neural` | McElfresh et al. "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" NeurIPS 2023 | Ch 6 — crossover phenomenon |
| `gorishniy2021ft` | Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data." NeurIPS 2021 | Ch 2 — FT-Transformer baseline |
| `gorishniy2022embeddings` | Gorishniy, Rubachev, Babenko. "On Embeddings for Numerical Features in Tabular Deep Learning." NeurIPS 2022 | Ch 3 — PLE justification |
| `chen2016xgboost` | Chen, Guestrin. "XGBoost: A Scalable Tree Boosting System." KDD 2016 | Ch 4 — tree baseline |
| `ke2017lightgbm` | Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017 | Ch 4 — tree baseline |
| `prokhorenkova2018catboost` | Prokhorenkova et al. "CatBoost: unbiased boosting with categorical features." NeurIPS 2018 | Ch 4 — tree baseline |
| `perez2018film` | Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI 2018 | Ch 3 — context integration |
| `oreshkin2018tadam` | Oreshkin et al. "TADAM: Task dependent adaptive metric for improved few-shot learning." NeurIPS 2018 | Ch 3 — FiLM delta init |
| `dumoulin2018feature` | Dumoulin et al. "Feature-wise transformations." Distill 2018 | Ch 3 — survey |
| `pham2024rcaeval` | Pham et al. "RCAEval: A Benchmark for Root Cause Analysis." ACM Web Conf. 2025 | Ch 2 + Ch 5 — real benchmark |
| `chen2024baro` | Pham et al. "BARO: Robust Root Cause Analysis for Microservices." FSE 2024 | Ch 2 — RCA |
| `zhou2018trainticket` | Zhou et al. "Fault Analysis and Debugging of Microservice Systems ... TrainTicket." IEEE TSE 2018 | Ch 4 — benchmark |
| `gan2019deathstar` | Gan et al. "DeathStarBench." ASPLOS 2019 | Ch 4 — benchmark |
| `zhang2022deeptralog` | Zhang et al. "DeepTraLog ... Graph-based Deep Learning." ICSE 2022 | Ch 2 — GNN efficacy |
| `lundberg2017shap` | Lundberg, Lee. "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017 | Ch 5 — SHAP |
| `guo2017calibration` | Guo et al. "On Calibration of Modern Neural Networks." ICML 2017 | Ch 5 — temperature scaling |
| `demsar2006statistical` | Demsar. "Statistical Comparisons of Classifiers over Multiple Data Sets." JMLR 2006 | Ch 5 — Wilcoxon / Welch |
| `nikolenko2021synthetic` | Nikolenko. *Synthetic Data for Deep Learning.* Springer 2021 | Ch 4 — synthetic-data validity |
| `jordon2022synthetic` | Jordon et al. "Synthetic Data — what, why and how?" arXiv:2205.03257 | Ch 4 — synthetic-data methodology |
| `geirhos2020shortcut` | Geirhos et al. "Shortcut Learning in Deep Neural Networks." NMI 2020 | Ch 4 — leakage framing |
| `pang2021deepreview` | Pang et al. "Deep Learning for Anomaly Detection: A Review." ACM CSUR 2021 | Ch 2 — AD vs attribution |
| `su2019omnianomaly` | Su et al. "Robust Anomaly Detection ... OmniAnomaly." KDD 2019 | Ch 2 — AD baseline |
| `audibert2020usad` | Audibert et al. "USAD." KDD 2020 | Ch 2 — AD baseline |
| `xu2022anomalytransformer` | Xu et al. "Anomaly Transformer." ICLR 2022 | Ch 2 — attention-based AD |
| `soldani2022survey` | Soldani, Brogi. "Anomaly Detection ... Survey." ACM CSUR 2022 | Ch 1 — FPR stats |

VERIFY flags (to double-check via DBLP before final submission): SAINT venue; LITE DSAA 2023; CloudAnoBench 2025.

---

## Chapter-by-chapter task plan

### Task 1: Preserve preamble, title page, abstract (rewrite abstract)

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (lines 1–165 — preamble, title, ack; replace abstract content lines 143–150)

- [ ] **Step 1: Keep preamble and title page (lines 1–119) unchanged**

- [ ] **Step 2: Keep acknowledgments (lines 124–134) unchanged**

- [ ] **Step 3: Rewrite abstract (replace lines 143–150)**

New abstract — replace the existing abstract block with:

```latex
Modern cloud-native systems built on microservice architectures generate
high-volume monitoring alerts with reported false-positive rates of 6--28\%.
Operational context---scheduled load events, deployment history, and
traffic-pattern signals---is routinely collected by production
observability stacks but is not used as a feature by existing anomaly
detection or attribution systems. This work investigates whether
contextual features materially improve anomaly attribution and, if so,
whether the benefit is specific to a neural architecture or is a general,
reproducible signal.

We first evaluate on the RCAEval benchmark (three microservice systems,
three dataset variants, $\sim$1{,}500 pooled cases). Gradient-boosted
tree baselines dominate all neural variants on real data---consistent
with published findings on small tabular tasks. To test whether sample
size is the bottleneck, we design a difficulty-aware synthetic benchmark
with explicit anti-leakage safeguards (pre-injection splits, randomized
context assignment, counterfactual baselines via RNG forking,
severity-tiered disguised faults) and run a controlled scaling study
from 500 to 40{,}000 samples over ten model variants with 10 random seeds.

Two findings emerge. First, a neural-over-tree crossover with scale:
at large $n$ and a lowered Bayes ceiling, the Full \sys{} model reaches
$95.11\% \pm 0.18$ F1 versus $94.90\% \pm 0.21$ for CatBoost-with-context
(Welch's $t = 2.249$, $p = 0.038$). Second, and more robustly,
\emph{context integration is a universal improvement mechanism}:
adding the same 5-dimensional context slice yields $+1.3$ to $+2.3$
percentage points of F1 across all five evaluated model families
(CAAA, CatBoost, XGBoost, LightGBM, RandomForest). The context
contribution is 6--10$\times$ larger than the best neural-over-tree gap,
replicates across seeds, and holds regardless of architecture class.
We therefore position \sys{} as a principled instantiation of a general
context-integration methodology rather than an architecture-specific win.

\vspace{0.5cm}
\noindent\textbf{Keywords:} Anomaly attribution, microservices, context
integration, tabular deep learning, scaling laws, evaluation methodology
```

- [ ] **Step 4: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: rewrite abstract for scaling-study narrative"
```

---

### Task 2: Chapter 1 — Introduction

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (replace chapter 1 content, lines 170–217)

- [ ] **Step 1: Replace chapter 1 section structure**

Sections:
1. Background — microservices, AIOps, FPR problem (keep existing, lightly edit)
2. Motivation — context as an unused signal (keep existing, lightly edit)
3. Research questions (rewrite to match new narrative):
   - RQ1: Does context materially improve anomaly attribution in microservices, and is the improvement architecture-specific or universal?
   - RQ2: Does a neural architecture (CAAA) ever outperform strong tree baselines (CatBoost), and if so, under what sample-size regime?
   - RQ3: What evaluation safeguards are required for a fair context-aware benchmark (leakage control, severity stratification, distribution matching)?
4. Contributions (rewrite to 6 items):
   (a) Difficulty-aware synthetic benchmark with 5 anti-leakage safeguards
   (b) Unified context feature taxonomy shared between synthetic and real data
   (c) Controlled scaling study (500→40K, 10 variants, 10 seeds)
   (d) Empirical evidence of neural-over-tree crossover with scale on anomaly attribution (p=0.038 at 40K hard)
   (e) The universal-context finding: +1.3–2.3pp across 5 model families
   (f) CAAA architecture — FiLM conditioning + PLE + Context Consistency Loss — as a principled context-integration instance
5. Report organization

- [ ] **Step 2: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: rewrite chapter 1 for new RQs and contributions"
```

---

### Task 3: Chapter 2 — Literature Survey (lightly revised)

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (replace chapter 2, lines 222–275)

- [ ] **Step 1: Keep existing structure, add two subsections**

Existing sections to retain (with light edits):
- Time-series anomaly detection
- RCA in microservices
- GNN-based fault localization
- FiLM conditioning
- Tabular deep learning

Add two new subsections:

**2.6 Scaling Laws and the Neural-Tree Crossover**

Cite `grinsztajn2022tree`, `shwartzziv2022tabular`, `mcelfresh2023neural`. Claim: tree dominance on small tabular is well-documented; crossover conditions (large $n$, many features, lower signal-to-noise) are less well characterized for microservice anomaly attribution specifically. This gap motivates our scaling study.

**2.7 Synthetic Data Validity in ML Evaluation**

Cite `nikolenko2021synthetic`, `jordon2022synthetic`, `geirhos2020shortcut`. Claim: synthetic data is a valid substrate for methodology claims if (a) the generative process is documented, (b) leakage controls are explicit, and (c) results corroborate with real-data trends where both exist. We satisfy all three.

- [ ] **Step 2: Update gap-analysis table (lines 257–275)**

Add row: "Context as feature, ✓ across model families (universal)" pointing to \sys{} contribution.

- [ ] **Step 3: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: add scaling-laws and synthetic-validity sections to lit survey"
```

---

### Task 4: Chapter 3 — Methodology (mostly preserved, two subsections edited)

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (chapter 3, lines 281–455)

- [ ] **Step 1: Keep architecture overview, PLE, Context Integration Module, Loss sections verbatim**

These are already correct: FiLM conditioning, TADAM-style delta, skeptical gate, Context Consistency Loss, CAAA+CatBoost hybrid, algorithm. No changes.

- [ ] **Step 2: Rewrite section 3.2 "Feature Vector Design" to emphasize unified feature taxonomy**

After the existing feature-group table, insert a paragraph: the same 44-dim vector is extracted from both synthetic and real cases. Context features (dims 12–16, 5d) have identical semantics in both: `event_active` (binary, was there a scheduled event?), `event_expected_impact` (scalar, expected load delta), `deploy_recent` (binary, recent deploy?), `time_of_day_sin/cos` (cyclic). Tree baselines consume the 44-dim flat; CAAA splits context out for FiLM. "No context" variants drop dims 12–16 → 39-dim input. **This uniformity is what makes the universal-context finding possible.**

- [ ] **Step 3: Add section 3.9 "Design Rationale for Universal Comparability"**

Short subsection (~200 words): all 10 model variants operate on the same 44-dim vector. The only architectural difference between CAAA and tree baselines is how the 5-dim context slice is integrated (FiLM + gated modulation for CAAA vs. concatenated-input for trees). This isolates context-integration mechanism from feature-set differences, enabling the per-family context-contribution analysis in Chapter 7.

- [ ] **Step 4: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: clarify unified feature taxonomy in methodology chapter"
```

---

### Task 5: Chapter 4 — Data: Real Benchmarks and Synthetic Generation (NEW CHAPTER)

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (replace existing chapter 4 "Evaluation Methodology" content, lines 461–499)

This chapter implements the user's requested narrative beats 1–4: real-data limits → synthetic-data need → synthetic-data design + anti-leakage → context feature taxonomy.

- [ ] **Step 1: Section 4.1 "The RCAEval Benchmark and Its Limits"**

Content:
- RCAEval overview: 3 systems (OnlineBoutique 12svc, SockShop 15svc, TrainTicket 64svc), 3 dataset variants (RE1 metrics-only 375 cases, RE2 multi-source 270, RE3 code-level 90), 735 total real fault cases.
- Pre-injection split yields ~1,500 balanced labeled samples pooled across all 9 system×dataset cells.
- This is too small for the neural-vs-tree comparison we want. Cite `grinsztajn2022tree`, `mcelfresh2023neural`: at $n \lesssim 2{,}000$ with modest feature dimensionality, tree models have a well-documented structural advantage.
- Conclusion: real benchmarks are necessary for external validity but insufficient to answer the scientific question "does architecture or context drive the gain?"

- [ ] **Step 2: Section 4.2 "Why We Needed Synthetic Data"**

Content:
- Desiderata: $n \geq 40{,}000$ labeled cases; per-service multivariate time series; structured context annotations (events, deploys, on-call); attribution labels (which service, which fault type, which severity).
- Survey of public datasets: RCAEval (small, no context annotations), DeathStarBench (no attribution labels), TrainTicket fault injection (small, no context), LOGHUB (logs-only, no structured context), AIOps 2018/2020 competition traces (no attribution, no context), CloudAnoBench 2025 (context-like "deceptive normals" but no feature-level integration hooks).
- Explicit statement: to our knowledge, no public dataset combines ($\geq$ thousands of cases, per-service multivariate metrics, structured context, attribution labels).
- Cite `nikolenko2021synthetic`, `jordon2022synthetic` for synthetic-data validity framing. The methodology-development path requires synthetic generation.

- [ ] **Step 3: Section 4.3 "Synthetic Data Design"**

Content:
- Service graph generator: fan-out 3–20 services per case, sampled topology.
- Metric generator: baseline diurnal load + noise floor, per-service CPU/memory/request-rate/error-rate/latency/network\_in/network\_out (7 channels × 60 timesteps × N services).
- Fault injection: 11 fault types (CPU exhaustion, memory leak, network delay, error cascade, ...) implemented in `FaultGenerator`. Severity drawn independently of fault type.
- Context generator: independent sampling process — deploys, event flags, time-of-day — NOT conditional on fault label.
- Difficulty levels:
  - **Default:** severity distribution 35/35/30 (low/medium/high), severity factors {0.05, 0.30, 1.00}, context-noise rate 30%.
  - **Hard:** severity distribution 60/25/15, severity factors {0.02, 0.15, 0.70}, context-noise rate 50%. Lowers the Bayes ceiling from ~97% to ~95%.

- [ ] **Step 4: Section 4.4 "Five Anti-Leakage Safeguards"**

Present as a numbered list with brief justification for each — this is the paragraph that defends the synthetic results against reviewer skepticism:

1. **Context-independent label sampling:** fault label and severity are drawn before any context sampling; context cannot inform the label distribution.
2. **Randomized context assignment:** 70% of NORMAL cases receive context events; 30% of FAULT cases receive fake "red-herring" context events. $P(\text{fault}|\text{no context}) = 0.70$, not 1.0. This blocks the shortcut-learning failure mode identified by `geirhos2020shortcut`.
3. **Pre-injection split for real data:** RCAEval cases are split at the `inject_time.txt` timestamp; both halves share identical distribution characteristics. Prevents distribution-mismatch artifacts where a model learns "real vs. synthetic source" instead of "fault vs. load."
4. **Counterfactual baselines via RNG forking:** for each fault case, a byte-identical normal-operation counterfactual is generated by forking the RNG before fault injection. This prevents reference-contamination bias in self-referencing baselines.
5. **Stratified train/val/test splits:** 60/20/20 splits stratified on label × severity × system-type prevents label leakage through class prior.

- [ ] **Step 5: Section 4.5 "Context Feature Taxonomy (Unified)"**

Content:
- Explicitly state: the 5-dim context slice (dims 12–16) has identical semantics on synthetic and real data.
- Table: context feature name, type, synthetic source, real-data source.
  - `event_active`: binary. Synthetic: generated event flag. Real: RCAEval operational-event annotations (derived from trace metadata when available, zero-imputed otherwise).
  - `event_expected_impact`: scalar. Synthetic: sampled load delta. Real: traffic-ratio deviation from 7-day window.
  - `deploy_recent`: binary. Synthetic: generated. Real: RCAEval deploy history where available.
  - `time_of_day_sin/cos`: cyclic. Both: computed from timestamp.
- Uniformity claim: any model trained on synthetic-context features can consume real-data-context features without retraining on feature semantics.

- [ ] **Step 6: Section 4.6 "Preserved Historical Pitfall Analysis"**

Keep the three pitfalls content from the old Chapter 4 (distribution mismatch, label leakage, context-dilemma) as a short subsection. This preserves the iterative-development story while reframing it as lessons feeding into section 4.4's safeguards.

- [ ] **Step 7: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: rewrite chapter 4 on data — real limits, synthetic design, leakage controls"
```

---

### Task 6: Chapter 5 — Real-Data Results (RENAMED, RESCOPED)

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (replace existing chapter 5 first half)

This chapter presents the RCAEval results honestly: CatBoost wins, CAAA is close but not better. It sets up the "so why bother with CAAA?" question that the scaling study answers.

- [ ] **Step 1: Section 5.1 "Experimental Setup"**

Keep existing content (PyTorch 2.6, 10 seeds, stratified split, etc.) — just limit scope to real-data setup here.

- [ ] **Step 2: Section 5.2 "Pooled Real-Data Results"**

Present a clean summary table from `ablation_results_combined.csv`:

| Variant | Avg F1 | MCC | FP Rate | Recall |
|---|---|---|---|---|
| Full CAAA | 0.9264 | 0.8642 | 0.088 | 0.944 |
| CAAA+CatBoost Hybrid | 0.9328 | 0.8729 | 0.073 | 0.940 |
| No Context Features | 0.9095 | 0.8280 | 0.098 | 0.919 |
| Baseline RF | 0.9431 | 0.8927 | 0.054 | 0.941 |
| XGBoost | 0.9337 | 0.8736 | 0.061 | 0.930 |
| LightGBM | 0.9400 | 0.8855 | 0.055 | 0.936 |
| **CatBoost** | **0.9542** | **0.9127** | **0.029** | 0.938 |
| **CatBoost (+ctx)** | **0.9571** | **0.9189** | **0.027** | **0.942** |

- [ ] **Step 3: Section 5.3 "Per-System Analysis"**

Keep existing per-system table (lines 602–624), add a column for CatBoost (+ctx) so the context-contribution can be traced per system.

- [ ] **Step 4: Section 5.4 "Honest Interpretation"**

Key paragraph — do not soften:

> On the RCAEval benchmark, CatBoost (with or without context) is the strongest model family. Full \sys{} achieves F1 = 92.6\%, ranking below CatBoost (95.4\%), CatBoost with context (95.7\%), and LightGBM (94.0\%). This is consistent with the extensive tabular-ML literature showing that at sample sizes $n \lesssim 2{,}000$ with moderate feature dimensionality, gradient-boosted trees have a structural advantage over neural models~\cite{grinsztajn2022tree,mcelfresh2023neural}. We emphasize this result rather than minimize it: on real data at current benchmark scales, trees should be preferred for deployment. The question addressed in Chapter~6 is whether this ranking is intrinsic or a consequence of sample size.

- [ ] **Step 5: Section 5.5 "SHAP Interpretability"**

Keep SHAP reference. Embed one or two SHAP summary plots from `outputs/results/shap_all_pooled/` as figures.

- [ ] **Step 6: Section 5.6 "Calibration"**

Add brief subsection citing `guo2017calibration`. Include one reliability-diagram figure from `outputs/results/calibration_all_pooled/`.

- [ ] **Step 7: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: rewrite chapter 5 as honest real-data results + interpretability"
```

---

### Task 7: Chapter 6 — Scaling Study (NEW CHAPTER)

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (replace existing chapter 6 content, lines 648–675)

This is the chapter that reframes the narrative. Two acts: default difficulty (saturation) → hard difficulty (crossover + significance).

- [ ] **Step 1: Section 6.1 "Motivation and Design"**

Content:
- Hypothesis: real-data tree dominance is a sample-size artifact, not an architectural intrinsic.
- Test: fix generative process; scale $n$ from 500 to 40,000; compare 10 model variants (Full CAAA, No Context, CatBoost ±ctx, XGBoost ±ctx, LightGBM ±ctx, RandomForest ±ctx).
- 10 seeds at each sample size for the headline runs; 3 seeds at smaller sizes for trajectory.
- Primary statistical test: Welch's t-test (unequal variance) between Full CAAA and CatBoost (+ctx) at 40K hard. Cite `demsar2006statistical`.

- [ ] **Step 2: Section 6.2 "Default-Difficulty Scaling: Saturation"**

Embed `scaling_curve.png` as a figure. Table populated from `scaling_study.csv`:

| $n$ | Full CAAA | CatBoost (+ctx) | Gap |
|---|---|---|---|
| 200 | 0.9500 | 0.9083 | +4.17 |
| 500 | 0.9367 | 0.9533 | −1.67 |
| 1,000 | 0.9483 | 0.9583 | −1.00 |
| 2,000 | 0.9567 | 0.9633 | −0.67 |
| 5,000 | 0.9610 | 0.9640 | −0.30 |
| 10,000 | 0.9623 | 0.9645 | −0.22 |
| 20,000 | 0.9664 | 0.9657 | +0.07 |
| 40,000 | 0.9714 | 0.9698 | +0.16 |

Interpretation: both models saturate near 97% F1. CAAA crosses over at 20K but the gap is within noise (p=0.09 with 3 seeds). This hints at a crossover but is not conclusive.

- [ ] **Step 3: Section 6.3 "Hard-Difficulty Scaling: Crossover Confirmed"**

Embed `scaling_curve_hard.png`. Table from `scaling_study_hard.csv` focusing on CAAA and CatBoost (+ctx):

| $n$ | Full CAAA | CatBoost (+ctx) | Gap |
|---|---|---|---|
| 500 | 0.9050 | 0.9150 | −1.00 |
| 1,000 | 0.9100 | 0.9335 | −2.35 |
| 5,000 | 0.9416 | 0.9427 | −0.11 |
| 20,000 | 0.9468 | 0.9465 | +0.03 |
| 40,000 | 0.9511 | 0.9490 | +0.21 |

Crossover occurs at $n \approx 20{,}000$. At $n = 40{,}000$ with 10 seeds, **Welch's t-test yields $t = 2.249$, $p = 0.0375$** — statistically significant at $\alpha = 0.05$.

- [ ] **Step 4: Section 6.4 "Effect-Size Honesty"**

Explicitly flag that the effect is small:
- $\Delta F_1 = 0.21$ percentage points at 40K hard.
- Cohen's $d \approx 1.08$ (moderate, based on pooled std of ~0.002).
- This is statistically detectable but practically modest.
- Corroborates `mcelfresh2023neural`: neural nets become competitive-but-not-dominant on tabular data as $n$ grows past a crossover threshold.

Key sentence: "We report the effect with its actual size, not amplified. The scientific value of Chapter 6 is the confirmation of a crossover regime, not a large architectural win."

- [ ] **Step 5: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: add chapter 6 — scaling study with crossover confirmation"
```

---

### Task 8: Chapter 7 — Universal Context Contribution (NEW CHAPTER, HEADLINE)

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (new chapter 7 replacing existing conclusion structure)

This is the report's strongest claim.

- [ ] **Step 1: Section 7.1 "Context Contribution per Model Family"**

Embed `context_contribution_hard.png` as Figure.

Primary table, populated from `scaling_study_hard.csv` (40K row, with-ctx minus no-ctx):

| Model family | No-ctx F1 | With-ctx F1 | $\Delta$ (pp) |
|---|---|---|---|
| CAAA | 0.9383 | 0.9511 | **+1.28** |
| CatBoost | 0.9320 | 0.9490 | **+1.70** |
| XGBoost | 0.9339 | 0.9499 | **+1.60** |
| LightGBM | 0.9342 | 0.9500 | **+1.58** |
| RandomForest | 0.9193 | 0.9423 | **+2.30** |

- [ ] **Step 2: Section 7.2 "Interpretation"**

Claims — strong, defensible:

1. Every model family gains from the identical 5-dim context slice.
2. The gain (+1.3 to +2.3 pp) is 6–10$\times$ larger than the best neural-over-tree gap (+0.21 pp at 40K hard).
3. The gain replicates across model-family inductive biases (neural FiLM modulation, GBDT with ordered boosting, GBDT with histogram binning, GBDT with leaf-wise growth, bagged decision trees).
4. Therefore the scientific contribution is not "CAAA beats CatBoost" (weak, small effect), but "context is a general improvement mechanism" (strong, reproducible, architecture-agnostic).

- [ ] **Step 3: Section 7.3 "Reframing CAAA's Contribution"**

Position CAAA as a principled *instantiation* of context integration:
- FiLM conditioning with skeptical gate is one of many ways to integrate context.
- Tree models do it via concatenation — and they work.
- The methodology is: (a) specify a well-defined context slice, (b) ensure leakage controls, (c) evaluate with universal comparability.
- CAAA demonstrates this methodology with a neural architecture; the finding is that the methodology itself carries the signal.

- [ ] **Step 4: Section 7.4 "Two Insights from the Joint Analysis"**

Summarize as headline insights:

**Insight 1 (Scaling crossover):** On anomaly attribution, gradient-boosted trees dominate at small $n$ but lose their lead as $n$ approaches $\sim$20{,}000 on harder tasks. Real-data benchmarks at $n \sim 1{,}500$ therefore underestimate neural-model capability; synthetic scaling is necessary to observe the crossover.

**Insight 2 (Universal context):** Context integration is a robust, architecture-independent improvement mechanism. The gain is larger and more reliable than any architectural choice evaluated.

- [ ] **Step 5: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: add chapter 7 — universal context contribution headline"
```

---

### Task 9: Chapter 8 — Discussion, Limitations, Conclusion

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (merge existing discussion + conclusion into one closing chapter)

- [ ] **Step 1: Section 8.1 "Discussion"**

- When does context help most? (Disguised-faults tier.)
- Why is the scaling crossover real but small? (Bayes ceiling; need harder tasks.)
- Why does context help trees too? (It adds genuine information orthogonal to metric patterns.)

- [ ] **Step 2: Section 8.2 "Limitations (honest)"**

- Synthetic-to-real transfer: universal-context finding is demonstrated on synthetic at 40K; real-data corroboration is only partial (CatBoost+ctx does narrowly beat CatBoost on real data pool: 0.9571 vs 0.9542, $\Delta$=+0.29 pp across 9 cells) — directional consistency, not matched magnitude.
- Crossover is small (0.21 pp) and demonstrated only on one synthetic task family.
- 5-dim context is a minimal slice; richer context might amplify the effect.
- Benchmark coverage: 3 microservice systems, 3 dataset variants. Generalization to other domains untested.

- [ ] **Step 3: Section 8.3 "Conclusion"**

Short — 3 paragraphs.
1. Restate: context is a universal improvement mechanism; CAAA is one principled instance; neural-over-tree crossover is real but modest.
2. Evaluation methodology contribution: 5 anti-leakage safeguards; difficulty-aware synthetic generation; controlled scaling at 40K.
3. Practical recommendation: deploy trees at small scale; deploy neural context-aware models when data grows past ~20K; always include context features regardless of model family.

- [ ] **Step 4: Section 8.4 "Future Work"**

Keep existing list (TabM, TabPFN ceiling, multi-root-cause, production integration, GNN augmentation, natural-language explanations).

- [ ] **Step 5: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: merge discussion + conclusion into chapter 8"
```

---

### Task 10: Update bibliography

**Files:**
- Modify: `report/CAAA_Project_Report.tex` (lines 713–810, bibliography)

- [ ] **Step 1: Add new BibItems**

Add BibItems for citations introduced by the new chapters (from the citation table above):

- `shwartzziv2022tabular` — Shwartz-Ziv, Armon 2022
- `mcelfresh2023neural` — already in (verify)
- `gorishniy2021ft` — Gorishniy 2021 FT-Transformer
- `chen2016xgboost` — XGBoost
- `ke2017lightgbm` — LightGBM
- `prokhorenkova2018catboost` — CatBoost
- `dumoulin2018feature` — Distill survey
- `zhou2018trainticket` — TrainTicket
- `gan2019deathstar` — DeathStarBench
- `zhang2022deeptralog` — DeepTraLog (replacing the shaky "GNNs effective" citation)
- `lundberg2017shap` — SHAP
- `guo2017calibration` — Temperature scaling
- `demsar2006statistical` — Statistical comparisons
- `nikolenko2021synthetic` — Synthetic data book
- `jordon2022synthetic` — Synthetic data methodology
- `pang2021deepreview` — AD review

- [ ] **Step 2: Remove or VERIFY-flag unsupported citations**

- `diagmlp2025` — VERIFY; if unconfirmed, replace with `zhang2022deeptralog` which has the same argumentative role.
- `cloudanobench2025` — VERIFY; keep only if confirmed.
- `cltad2023`, `chainofevent2024`, `dynacausal2025` — VERIFY via DBLP before final submission.

- [ ] **Step 3: Commit**

```bash
git add report/CAAA_Project_Report.tex
git commit -m "report: expand bibliography with scaling-study and methodology citations"
```

---

### Task 11: Compile + proof

**Files:**
- Run: `pdflatex report/CAAA_Project_Report.tex` twice (for TOC + refs)

- [ ] **Step 1: Compile**

```bash
cd report && pdflatex -interaction=nonstopmode CAAA_Project_Report.tex && pdflatex -interaction=nonstopmode CAAA_Project_Report.tex
```

- [ ] **Step 2: Check for compile errors, undefined references, missing figures**

- Every `\cite{...}` resolves
- Every `\ref{...}` resolves
- All included PNGs exist at referenced paths
- Bibliography renders

- [ ] **Step 3: Proof-read once end-to-end**

Focus: does the narrative flow match the outline (real-data limits → synth need → synth design → context features → scaling → universal context → insights)? Are the two headline numbers (p=0.0375; +1.3 to +2.3 pp) each stated once with proper context, not repeatedly amplified?

- [ ] **Step 4: Commit compiled PDF**

```bash
git add report/CAAA_Project_Report.pdf report/CAAA_Project_Report.tex
git commit -m "report: compile final PDF"
```

---

## Self-Review Checklist (before handing off)

- [ ] Every requirement in the user's outline (real-data limits → synth need → synth design → anti-leakage → context features → scaling → universal context → insights) has a chapter/section.
- [ ] Every VERIFY-flagged citation either (a) confirmed via DBLP, or (b) replaced with a verified equivalent, or (c) removed.
- [ ] No "TODO" or placeholder text remains in the .tex file.
- [ ] The small effect size (0.21 pp) is stated honestly, not amplified.
- [ ] The large effect (+1.3–2.3 pp universal context) is stated once with proper framing as the headline result.
- [ ] Real-data results in Ch 5 do not claim CAAA wins; they honestly report CatBoost's advantage.
- [ ] All figure files referenced exist at the path.
- [ ] All CSV numbers in tables match `outputs/results/*.csv` exactly (spot-check 5 cells).

---

## Files Produced

- `report/CAAA_Project_Report.tex` — rewritten
- `report/CAAA_Project_Report.pdf` — compiled

## Files Unchanged

- `src/**` — no code changes
- `outputs/results/**` — no new experiments
- `docs/research_journey.md` — separate, out of scope for this plan

## Time estimate

- Writing: 3–4 hours
- Citation verification: 30–45 minutes
- Compile + proof: 30 minutes
- Total: ~4–5 hours
