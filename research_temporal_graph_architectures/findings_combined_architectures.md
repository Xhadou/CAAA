# Combined Temporal-Graph Neural Network Architectures for AIOps / Microservice Anomaly Detection

**Research Date:** 2026-04-09
**Scope:** Spatio-temporal GNNs, LSTM/GRU+GNN hybrids, graph attention temporal networks for root cause analysis in microservice systems. Focus on architecture design, temporal-graph interaction patterns, parameter efficiency, and performance vs. simpler baselines.

---

## 1. Architecture Taxonomy

Three dominant patterns emerge for combining temporal and graph components:

| Pattern | Temporal | Graph | Interaction | Representative Systems |
|---------|----------|-------|-------------|----------------------|
| **Parallel Fusion** | TCN or LSTM branch | GAT/GCN branch | Concatenate outputs | GTAD, MTAD-GAT |
| **Sequential Encoder** | LSTM/GRU after GNN | GAT layers first | GNN feeds into RNN | GAL-MAD, TraceGra |
| **Unified Cell** | GraphLSTM cell | GNN gates replace FC | Fused at cell level | D-GNN+LSTM, OmniAnomaly |
| **Diffusion-Based** | Temporal attention | Graph diffusion | Propagation over time | ServiceGraph-FM, ST-GraphRCA |

---

## 2. Key Systems and Architectures

### 2.1 GTAD (Graph and Temporal Neural Network for Anomaly Detection)

**Source:** Entropy 2022 ([PMC9222957](https://pmc.ncbi.nlm.nih.gov/articles/PMC9222957/))

**Architecture:**
- **Graph component:** GATv2 (Graph Attention Network v2) with sigmoid activation and learnable attention scores. Learns the adjacency matrix from data without prior topology knowledge.
- **Temporal component:** Temporal Convolutional Network (TCN) with causal dilated convolutions (dilation factors 1, 2, 4), residual connections via 1x1 convolution, weight normalization + ReLU + spatial dropout.
- **Interaction:** Parallel -- TCN and GATv2 extract features independently; outputs are concatenated to form H in R^(2k x L). Multi-head attention then processes the combined representation for reconstruction.
- **Reconstruction:** Forecasting-based + reconstruction-based dual objective.

**Parameter count:** Not reported. Training overhead: ~3.73 seconds/epoch on MSL dataset.

**Performance (F1):**
| Dataset | GTAD | GDN | MTAD-GAT | OmniAnomaly |
|---------|------|-----|----------|-------------|
| MSL | 0.954 | 0.888 | 0.906 | 0.898 |
| SMAP | 0.962 | 0.921 | 0.914 | 0.869 |
| SMD | 0.960 | 0.917 | 0.949 | 0.940 |

**Ablation results:**
- Removing GATv2: ~6% F1 decrease
- Removing TCN: ~2% F1 decrease
- Removing multi-head attention: ~10% F1 decrease
- Single-dimension loss vs. all-dimension: ~23% F1 decrease

**Takeaway:** Graph attention contributes more than the temporal convolution component. The attention mechanism over the fused representation is the single most important component.

---

### 2.2 MTAD-GAT (Multivariate Time-series Anomaly Detection via Graph Attention Network)

**Source:** Zhao et al. 2020 ([arXiv:2009.02040](https://arxiv.org/abs/2009.02040)), PyTorch impl: [ML4ITS/mtad-gat-pytorch](https://github.com/ML4ITS/mtad-gat-pytorch)

**Architecture:**
- **Two parallel GAT branches:** (1) Feature-oriented GAT capturing inter-feature dependencies, (2) Time-oriented GAT capturing temporal correlations across timestamps.
- **Sequential fusion:** Outputs of both GATs are concatenated and passed through a GRU layer to capture longer sequential patterns.
- **Dual decoder:** Forecasting decoder (FC layers) + reconstruction decoder (VAE-based).

**Parameter count:** From reference implementation -- approximately 200K-400K parameters depending on configuration (hidden_dim=150, n_heads=1, default settings).

**Performance:** Consistently strong on MSL, SMAP, SMD benchmarks. Widely used as a baseline in subsequent work.

**Takeaway:** Seminal work demonstrating that parallel graph attention branches (feature-oriented + time-oriented) combined with a sequential GRU produces strong anomaly detection. The dual-branch GAT design is the key architectural innovation.

---

### 2.3 GAL-MAD (Graph Attention Layers for Microservice Anomaly Detection)

**Source:** 2025 ([arXiv:2504.00058](https://arxiv.org/html/2504.00058))

**Architecture:**
- **Encoder:** Two GAT layers process spatial dependencies across services -> reshape -> bidirectional LSTM captures temporal patterns.
- **Decoder:** Mirrors encoder in reverse (LSTM -> GAT layers).
- **Input:** 3D tensor (t=24 timesteps x n=12 services x k=22 features). Total 264 features.
- **Embedding dimension:** dz = 1 (extremely compressed latent space).
- **Post-hoc RCA:** SHAP values aggregated across services to localize anomalous microservices.

**Parameter count:** Not explicitly reported. Architecture is relatively lightweight given dz=1 bottleneck.

**Ablation results (Recall at 95:5 anomaly ratio):**
| Variant | Recall |
|---------|--------|
| **GAL-MAD (full)** | **0.988** |
| GAT-AE (no LSTM) | 0.884 |
| LSTM-AE (no GAT) | 0.903 |
| Linear-AE (no GAT, no LSTM) | 0.890 |

**Baseline comparison (Recall at 95:5):**
| Model | Recall |
|-------|--------|
| **GAL-MAD** | **0.988** |
| GDN | 0.809 |
| MAD-GAN | 0.863 |
| Kitsune | 0.901 |
| Transformer | 0.825 |

**Takeaway:** Sequential GAT->LSTM design with explicit adjacency matrix encoding of microservice topology outperforms learned-graph approaches (GDN) and non-graph temporal models. The ablation clearly shows both GAT and LSTM contribute, with GAT providing ~10% recall improvement and LSTM providing ~8.5% over the linear baseline.

---

### 2.4 ServiceGraph-FM (Graph Foundation Model for RCA)

**Source:** Mathematics 2026 ([MDPI 2227-7390/14/2/236](https://www.mdpi.com/2227-7390/14/2/236))

**Architecture:**
- **Pretraining:** Masked graph autoencoding learns transferable service-dependency embeddings from unlabeled topology data.
- **Temporal relational diffusion:** Models anomaly propagation as graph diffusion on dynamic service graphs with learnable edge propagation strengths. This replaces static adjacency with time-varying propagation.
- **Causal attention mechanism:** Leverages multi-hop path signals to identify root cause services.
- **Transfer learning:** Pretrained embeddings allow adaptation to new service topologies with minimal labeled data.

**Parameter count:** Not reported, but designed for parameter efficiency through pretraining + fine-tuning paradigm.

**Key innovation:** Treats fault propagation as a diffusion process over the service graph, where edge weights (propagation strengths) are learned and vary over time. This is more physically motivated than static GNN message passing.

**Takeaway:** Foundation-model approach to RCA. Addresses the small-dataset problem through self-supervised pretraining on graph structure. The temporal diffusion formulation is a compelling alternative to LSTM/GRU for modeling fault propagation dynamics.

---

### 2.5 ST-GraphRCA (Spatio-Temporal Graph Propagation for RCA)

**Source:** Sensors 2026 ([MDPI 1424-8220/26/5/1474](https://www.mdpi.com/1424-8220/26/5/1474))

**Architecture:**
- **Feature extraction:** PCA-DTW hybrid method for handling time-series asynchrony across distributed multi-source metrics.
- **Dynamic alignment:** Addresses the fundamental problem that metrics from different nodes are not temporally synchronized.
- **Graph propagation:** Spatio-temporal graph propagation model for root cause localization in IoT edge environments.
- **Designed for real-time:** Meets latency requirements for Industrial IoT.

**Takeaway:** Addresses a practical problem often ignored -- temporal misalignment between metrics from different services/nodes. The PCA-DTW alignment preprocessing is relevant for any temporal-graph approach on real microservice data.

---

### 2.6 TraceGra (Trace-based Graph Anomaly Detection)

**Source:** Computer Communications 2023 ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0140366423001135))

**Architecture:**
- **Unified graph representation:** Combines distributed traces and container performance metrics into a single graph.
- **Encoder:** GNN extracts topology features from the trace/dependency graph; LSTM extracts temporal features from metric time series.
- **Unsupervised:** Encoder-decoder architecture trained on normal data only.

**Takeaway:** Interesting for its multi-modal input (traces + metrics) unified into a graph representation, rather than treating them as separate data streams.

---

### 2.7 D-GNN + LSTM (Dynamic Graph Neural Network with LSTM)

**Source:** Symmetry 2024 ([MDPI 2073-8994/18/1/87](https://www.mdpi.com/2073-8994/18/1/87))

**Architecture:**
- **Dynamic GNN:** Graph structure evolves over time to capture changing microservice dependencies.
- **LSTM integration:** Combined at the cell level -- GNN gates replace fully connected layers within LSTM cells (GraphLSTM pattern).
- **CPU-only design:** Explicitly designed to run without GPU, targeting resource-constrained monitoring environments.

**Takeaway:** The GraphLSTM cell design (replacing FC layers in LSTM gates with GNN message passing) is the most tightly coupled temporal-graph integration pattern. CPU-only constraint forces parameter efficiency.

---

### 2.8 MULAN (Multi-modal Causal Structure Learning)

**Source:** KDD 2024 ([Paper PDF](https://zhengzhangchen.github.io/paper/MULAN-%20Multi-modal%20Causal%20Structure%20Learning%20and%20Root%20Cause%20Analysis%20for%20Microservice%20Systems.pdf))

**Architecture (4 modules):**
1. **Representation extraction:** Log-tailored language model encodes log data.
2. **Contrastive multi-modal causal structure learning:** Learns causal graph from metrics + logs using contrastive learning.
3. **Causal graph fusion:** KPI-aware attention merges discovered causal structures.
4. **Network propagation-based root cause localization:** Random walk on the causal graph to rank root cause candidates.

**Takeaway:** Multi-modal (metrics + logs) approach to causal graph discovery, rather than assuming a fixed service dependency graph. The contrastive learning approach for graph structure learning is relevant for settings where the true dependency graph is unknown or dynamic.

---

### 2.9 CIRCA and CloudRCA (Established Baselines)

**CIRCA:** Constructs a causal graph among monitoring metrics using system architecture knowledge and causal assumptions. Uses intervention-based causal inference rather than neural networks. Limited by static graph assumption.

**CloudRCA:** Neural network-based but treats the service graph as static. Known limitation: cannot adapt to topology changes or capture temporal evolution of dependencies.

**Takeaway:** Both are widely cited baselines but fundamentally limited by static graph assumptions. The temporal-graph architectures above specifically aim to address this limitation.

---

### 2.10 GAT-BN (Graph Attention Bayesian Network)

**Source:** Scientific Reports 2026 ([Nature](https://www.nature.com/articles/s41598-026-36883-7))

**Architecture:**
- Integrates Graph Attention mechanism into Bayesian Network structure learning.
- **Adaptive Prior algorithm:** Hierarchical attention mechanism calibrates prior strength to compensate for data scarcity.
- **GAT module:** Autonomously learns node criticality scores, focusing on low-frequency but high-consequence root causes.
- **Domain constraints:** Incorporates hierarchical domain-specific constraints during structure learning.

**Takeaway:** Specifically designed for small/imbalanced datasets. The adaptive prior with GAT-learned criticality scores is a principled approach to the sparse-label problem common in fault diagnosis.

---

## 3. Comparison of Temporal-Graph Interaction Patterns

### 3.1 Parallel Fusion (GTAD, MTAD-GAT)
- **Pros:** Simple to implement; each branch can be tuned independently; clear ablation of each component's contribution.
- **Cons:** No information flow between temporal and graph branches during feature extraction; late fusion may miss temporal-spatial correlations.
- **Best for:** General multivariate time-series anomaly detection where graph structure is not known a priori.

### 3.2 Sequential (GAL-MAD, TraceGra)
- **Pros:** Graph-enriched representations feed directly into temporal modeling; captures how graph-propagated signals evolve over time.
- **Cons:** Information flows one direction only (graph -> temporal or temporal -> graph); ordering matters and optimal order is task-dependent.
- **Best for:** Microservice RCA where the service dependency graph is known and stable.

### 3.3 Unified Cell (GraphLSTM / D-GNN+LSTM)
- **Pros:** Tightest integration; every temporal update step considers graph neighborhood; naturally models message passing over time.
- **Cons:** Higher implementation complexity; harder to ablate individual contributions; can be computationally expensive.
- **Best for:** Systems requiring real-time online detection where temporal and graph signals are deeply entangled.

### 3.4 Diffusion-Based (ServiceGraph-FM, ST-GraphRCA)
- **Pros:** Physically motivated (fault propagation as diffusion); learnable propagation speeds; handles temporal misalignment naturally.
- **Cons:** Assumes specific propagation dynamics; may not capture non-diffusive failure modes.
- **Best for:** Cascading failure scenarios where fault propagation timing matters.

---

## 4. Parameter Efficiency and Small Dataset Considerations

| Strategy | Used By | Mechanism |
|----------|---------|-----------|
| **Pretraining + fine-tuning** | ServiceGraph-FM | Masked graph autoencoding on unlabeled topology data |
| **Contrastive learning** | MULAN | Multi-modal contrastive loss for causal graph discovery |
| **Adaptive priors** | GAT-BN | Hierarchical attention calibrates prior strength for sparse data |
| **Extreme bottleneck** | GAL-MAD | dz=1 latent dimension forces compression |
| **CPU-only constraint** | D-GNN+LSTM | GraphLSTM designed for resource-constrained environments |
| **Dual-objective training** | GTAD, MTAD-GAT | Forecasting + reconstruction loss provides richer gradients |
| **DTW alignment** | ST-GraphRCA | PCA-DTW handles temporal misalignment, reducing need for large aligned datasets |

### Recommendations for Small Datasets (~100s of samples):
1. **Use known graph structure** rather than learning it (GAL-MAD approach with explicit adjacency matrix).
2. **Dual-objective training** (forecast + reconstruct) provides more supervisory signal per sample.
3. **Pretrain on unlabeled normal data** using masked autoencoding (ServiceGraph-FM pattern).
4. **Constrain model capacity** aggressively -- GAL-MAD's dz=1 bottleneck is surprisingly effective.
5. **Avoid learning the graph from scratch** on small datasets -- use domain knowledge or causal priors.

---

## 5. Key Findings and Implications for CAAA

### What consistently works:
- **Graph attention > graph convolution** for RCA tasks. GAT/GATv2 used in nearly all top-performing systems.
- **Known topology outperforms learned graphs** when available (GAL-MAD vs. GDN comparison: 0.988 vs. 0.809 recall).
- **Both temporal and graph components contribute**, but graph attention typically provides larger gains (GTAD ablation: -6% for GAT vs. -2% for TCN).
- **Multi-head attention over fused representations** is the single most impactful component in GTAD (-10% when removed).

### What to watch:
- **Diffusion-based temporal propagation** (ServiceGraph-FM) is a promising alternative to RNN/TCN for modeling fault cascades.
- **Foundation model pretraining** for service graphs enables transfer across topologies.
- **Multi-modal inputs** (metrics + logs + traces) consistently outperform single-modal approaches.

### Practical architecture recommendation:
For a parameter-efficient temporal-graph model on small microservice datasets:
1. Encode known service dependency graph as explicit adjacency matrix.
2. Use 1-2 GAT layers (not GCN) with small hidden dimensions (32-64).
3. Follow with a lightweight temporal model (small GRU or 1D conv, not full Transformer).
4. Train with dual objective (forecasting + reconstruction).
5. Use SHAP or attention weights for post-hoc root cause localization.
6. Expected parameter count: 50K-200K parameters, trainable on CPU.

---

## Sources

- [GTAD - Graph and Temporal Neural Network for Anomaly Detection (PMC9222957)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9222957/)
- [MTAD-GAT - Multivariate Time-Series Anomaly Detection via Graph Attention Network](https://arxiv.org/abs/2009.02040)
- [MTAD-GAT PyTorch Implementation](https://github.com/ML4ITS/mtad-gat-pytorch)
- [GAL-MAD - Graph Attention for Microservice Anomaly Detection](https://arxiv.org/html/2504.00058)
- [ServiceGraph-FM - Temporal Relational Diffusion for RCA](https://www.mdpi.com/2227-7390/14/2/236)
- [ST-GraphRCA - Spatio-Temporal Graph Propagation for RCA](https://www.mdpi.com/1424-8220/26/5/1474)
- [D-GNN+LSTM - CPU-Only Spatiotemporal Anomaly Detection](https://www.mdpi.com/2073-8994/18/1/87)
- [TraceGra - Trace-based Graph Anomaly Detection](https://www.sciencedirect.com/science/article/abs/pii/S0140366423001135)
- [MULAN - Multi-modal Causal Structure Learning](https://zhengzhangchen.github.io/paper/MULAN-%20Multi-modal%20Causal%20Structure%20Learning%20and%20Root%20Cause%20Analysis%20for%20Microservice%20Systems.pdf)
- [GAT-BN - Graph Attention Bayesian Network for Fault RCA](https://www.nature.com/articles/s41598-026-36883-7)
- [Survey: GNNs for Microservice-Based Cloud Applications](https://pmc.ncbi.nlm.nih.gov/articles/PMC9738439/)
- [Comprehensive Survey on RCA in Microservices](https://arxiv.org/html/2408.00803v1)
- [Awesome GNN for Time Series (GitHub)](https://github.com/KimMeen/Awesome-GNN4TS)
- [ICLR 2025 - Root Cause Analysis of Anomalies in Multi-Service Systems](https://proceedings.iclr.cc/paper_files/paper/2025/file/6fde96479648d71e4fd9724374bf76eb-Paper-Conference.pdf)
- [Spatio-temporal Causal Graph Attention Network](https://arxiv.org/pdf/2203.10749)
