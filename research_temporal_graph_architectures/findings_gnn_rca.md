# GNN/GAT Architectures for Microservice Root Cause Analysis

Research survey conducted 2026-04-09. Covers papers from 2020-2025 on graph neural
networks applied to service dependency graphs for fault localization and root cause
analysis in microservice systems.

---

## 1. MicroRCA (NOMS 2020)

**Paper**: "MicroRCA: Root Cause Localization of Performance Issues in Microservices"
(Wu et al., IEEE/IFIP NOMS 2020)

| Aspect | Detail |
|--------|--------|
| **Graph construction** | Attributed graph with service nodes and host nodes. Edges from trace-derived call relationships. Node attributes = response time anomaly scores + resource utilization metrics (CPU, memory, network). |
| **GNN variant** | None -- uses **personalized PageRank / random walk** on the attributed graph. Not a learned GNN; purely graph-algorithmic. |
| **Model size** | N/A (no trainable parameters; graph algorithm). |
| **Dataset** | Sock Shop (5 microservices on Kubernetes). ~100 fault injection experiments across CPU hog, memory leak, network delay. |
| **vs. tree-based** | Compared against MonitorRank and CloudRanger (also graph-walk methods). Achieves 89% precision, MAP 97% -- at least 13% and 15% higher than baselines. No tree-based ML baselines evaluated. |

**Key takeaway**: Foundational work. Graph is constructed at inference time from live
traces + metrics, not learned. Subsequent works (OpsKG, MicroHECL) build on this
random-walk paradigm.

---

## 2. DiagFusion (ICSE-SEIP 2023 / arXiv 2302.10512)

**Paper**: "Robust Failure Diagnosis of Microservice System through Multimodal Data"
(Zhang et al.)

| Aspect | Detail |
|--------|--------|
| **Graph construction** | Dependency graph built from **traces + deployment configuration**. Nodes = service instances. Edges = invocation relationships extracted from distributed traces. |
| **GNN variant** | **GCN** (Graph Convolutional Network) applied on the dependency graph. Multimodal embeddings (metrics, logs, traces) are node features; GCN aggregates neighborhood info for root cause instance localization + fault type classification. |
| **Model size** | Not explicitly reported. Embedding dim likely 64-128 based on follow-up evaluations. Estimated <500K parameters given the graph sizes. |
| **Dataset** | Train-Ticket system: 27 services, thousands of fault injection samples. Also tested on proprietary industrial datasets. |
| **vs. tree-based** | Not directly compared against tree-based classifiers. Compared against CloudRanger, MicroRCA, MicroRank. Outperforms on localization accuracy. |

**Key takeaway**: First major work to fuse logs + metrics + traces into GCN-based
diagnosis. The "Are GNNs Effective?" paper (below) later showed that much of
DiagFusion's gain comes from the multimodal preprocessing, not the GCN itself.

---

## 3. CHASE (WWW 2025 / arXiv 2406.19711)

**Paper**: "CHASE: A Causal Heterogeneous Graph based Framework for Root Cause
Analysis in Multimodal Microservice Systems" (Wang et al.)

| Aspect | Detail |
|--------|--------|
| **Graph construction** | **Heterogeneous hypergraph** from traces (invocation DAG), metrics (time-series linked to instance nodes), and logs (text events linked to instances). Each hyperedge represents a causality propagation path along an invocation chain. |
| **GNN variant** | **Hypergraph neural network** with heterogeneous message passing. 3 attention layers, 8 attention heads per layer, 1 hypergraph convolution layer. Not standard GCN/GAT -- custom hypergraph convolution. |
| **Model size** | Not explicitly reported. Embedding dim = 128. With 3 layers x 8 heads, estimated ~1-2M parameters. |
| **Dataset** | (1) GAIA: 10 services, 1,099 traces, 160 training traces, 4 anomaly types. (2) AIOps 2020 Challenge: hundreds of instances, 68 manually injected failures over 3 months, dynamic topology. |
| **vs. tree-based** | Not compared against tree-based. Compared against PC, GES, CloudRanger, MicroRCA, DiagFusion, Nezha, REASON. Achieves **36.2% gain on A@1** on GAIA and **29.4% improvement** on AIOps 2020 vs. best baseline. |

**Key takeaway**: Hypergraph structure captures multi-hop causality propagation that
standard pairwise GNNs miss. Strong results but on relatively small datasets. The
hypergraph approach is architecturally more complex than GAT/GCN.

---

## 4. DynaCausal (arXiv 2510.22613, Oct 2025)

**Paper**: "DynaCausal: Dynamic Causality-Aware Root Cause Analysis for Distributed
Microservices"

| Aspect | Detail |
|--------|--------|
| **Graph construction** | **Dynamic call graph** per time window. Edge weight = sigmoid(alpha * Norm(request_count) + (1-alpha) * Norm(error_rate)). Topology changes per window based on live trace data. |
| **GNN variant** | **H-GAT (Hybrid-Aware Graph Attention Network)** -- a novel dual-mechanism GAT combining (a) standard attention for feature-based similarity and (b) causal propagation weighting using dynamic edge weights from request counts and error rates. Preceded by a Transformer encoder for temporal modeling per service. |
| **Model size** | Not explicitly reported. Architecture: Transformer encoder -> H-GAT (K layers) -> MLP head. Estimated ~2-5M parameters given Transformer + multi-layer GAT. |
| **Dataset** | D1 (RCAEval-RE2-OB / Online Boutique): 90 fault cases, 12 services, 6 fault types. D2 (proprietary): 1,430 fault cases, 50 services, 25 fault types. |
| **vs. tree-based** | No tree-based baselines. Compared against Eadro, ART, Nezha, Baro, MicroRank. AC@1 = **0.769** on D1 (vs. Eadro 0.459), **0.481** on D2 (vs. Eadro 0.301). |

**Key takeaway**: The Transformer+GAT hybrid is the most architecturally sophisticated
approach found. Dynamic graph construction per time window is well-suited for
production systems where topology changes. The dual attention mechanism (learned +
causal) is a notable design choice.

---

## 5. GALR (Electronics, Jan 2025)

**Paper**: "GALR: Graph-Based Root Cause Localization and LLM-Assisted Recovery for
Microservice Systems"

| Aspect | Detail |
|--------|--------|
| **Graph construction** | Multimodal service call graph fusing time-series metrics, structured logs, and trace-derived topology. |
| **GNN variant** | **GAT** with temporal-aware edge attention to model failure propagation. Combined with LLM for recovery recommendation. |
| **Model size** | Not reported. |
| **Dataset** | Customer Service, Power Grid Resource, and Sock Shop benchmarks. |
| **vs. tree-based** | Compared against GNN baselines and recent RCA methods. Specific numbers not available from abstracts. |

---

## 6. "Are GNNs Actually Effective?" (ICSE 2025 / arXiv 2501.02766)

**Paper**: "Are GNNs Actually Effective for Multimodal Fault Diagnosis in Microservice
Systems?" (Yu et al., ICSE 2025)

This is a **critical negative result** paper that systematically evaluates GNN
contributions in microservice RCA.

### GNN Variants Benchmarked
- GCN, GAT, GraphSAGE (used inside DiagFusion, Eadro, DGERCL, DeepHunt)

### Datasets Evaluated

| Dataset | Services | Train | Val | Test |
|---------|----------|-------|-----|------|
| Social Network (SN) | 12 | 316 | 78 | 169 |
| Train Tickets (TT) | 27 | 3,256 | 813 | 1,744 |
| GAIA | 10 | 128 | 32 | 939 |
| D1 (industrial) | 46 | 63 | -- | 147 |
| D2 (industrial) | 18 | 40 | -- | 93 |

### Key Results: DiagMLP (no graph) vs. GNN Methods

| Task | Dataset | DiagMLP (no graph) | Best GNN Method |
|------|---------|-------------------|-----------------|
| Fault Detection F1 | SN | **96.7%** | 92.1% (Eadro) |
| Fault Detection F1 | TT | **90.8%** | 90.7% (Eadro) |
| Fault Localization Top-1 | SN | **80.2%** | 41.8% (Eadro) |
| Fault Localization Top-1 | TT | **98.5%** | 91.1% (baseline) |

### Conclusion
> "Improvements are largely driven by multimodal data preprocessing and embedding
> techniques, rather than GNN-based dependency modeling."

**Key takeaway for CAAA project**: This is the most relevant paper. It shows that on
current benchmarks, **a simple MLP with good feature engineering matches or beats
GNN/GAT approaches**. The graph structure provides minimal discriminative value when
the preprocessing pipeline already encodes dependency information. This validates
CAAA's tree-based approach.

---

## 7. Additional Notable Papers

### ServiceGraph-FM (Mathematics, Jan 2026)
- Pretrained graph-based foundation model for RCA
- Uses masked graph autoencoding pretraining + temporal relational diffusion
- Causal attention mechanism on dynamic service graphs
- Targets large-scale payment systems

### MicroEGRCL (ICSOC 2022)
- Edge-attention-based GNN for root cause localization
- Constructs graph from trace call chains
- Uses edge attention (similar to GAT but on edge features)

### MTG_CD (J. Cloud Computing, 2024)
- Multi-scale learnable transformation graph for fault classification
- Combines CNN and GCN on service dependency graphs

### DeepHunt (ACM TOSEM, 2024)
- Graph autoencoder for interpretable failure localization
- Reconstruction-based anomaly detection on service graphs

---

## Summary Table

| Method | Year | Venue | GNN Type | Graph Source | Beat Tree Baselines? | Dataset Scale |
|--------|------|-------|----------|-------------|---------------------|---------------|
| MicroRCA | 2020 | NOMS | PageRank (no GNN) | Traces + metrics | N/A | ~100 faults, 5 services |
| DiagFusion | 2023 | ICSE-SEIP | GCN | Traces + deploy config | Not tested | ~5K samples, 27 services |
| CHASE | 2025 | WWW | Hypergraph NN | Traces + metrics + logs | Not tested | 1,099 traces, 10 services |
| DynaCausal | 2025 | arXiv | Transformer + H-GAT | Dynamic call graph | Not tested | 90-1430 faults, 12-50 services |
| GALR | 2025 | Electronics | GAT | Traces + metrics + logs | Not tested | Sock Shop + 2 others |
| DiagMLP | 2025 | ICSE | **None (MLP)** | No graph | **Beats GNNs** | 5 datasets, 10-46 services |

---

## Implications for CAAA

1. **GNNs are not clearly superior to simpler methods.** The ICSE 2025 study (Yu et
   al.) demonstrates that topology-agnostic MLPs match or beat GNN-based methods
   when multimodal features are well-engineered. This strongly supports CAAA's
   tree-based (Random Forest, XGBoost) approach.

2. **Feature engineering dominates.** Across all papers, the quality of metric/trace/log
   preprocessing matters more than the graph architecture. CAAA's SHAP-driven feature
   analysis and severity scaling are well-aligned with this finding.

3. **Dataset sizes are small.** The largest benchmark (Train Tickets) has ~5,800
   samples across 27 services. Most have <1,000 fault cases. Tree-based methods
   excel in this low-data regime.

4. **If exploring GNN augmentation**: DynaCausal's dual-mechanism H-GAT (learned
   attention + causal propagation weights) is the most principled design. The dynamic
   graph construction per time window is also relevant. However, given the DiagMLP
   findings, the expected marginal gain over CAAA's current approach is likely small.

5. **Hypergraph approaches** (CHASE) capture multi-hop causality but add significant
   architectural complexity. Worth monitoring but premature to adopt given current
   evidence.

---

## Sources

- [MicroRCA (HAL/INRIA)](https://inria.hal.science/hal-02441640/document)
- [DiagFusion (arXiv 2302.10512)](https://arxiv.org/abs/2302.10512)
- [CHASE (arXiv 2406.19711)](https://arxiv.org/html/2406.19711v1)
- [DynaCausal (arXiv 2510.22613)](https://arxiv.org/html/2510.22613)
- [GALR (MDPI Electronics)](https://www.mdpi.com/2079-9292/15/1/243)
- [Are GNNs Effective? (arXiv 2501.02766)](https://arxiv.org/html/2501.02766v2)
- [Comprehensive RCA Survey (arXiv 2408.00803)](https://arxiv.org/html/2408.00803v1)
- [ServiceGraph-FM (MDPI Mathematics)](https://www.mdpi.com/2227-7390/14/2/236)
- [RCA Causal Inference Survey (arXiv 2408.13729)](https://arxiv.org/html/2408.13729v1)
