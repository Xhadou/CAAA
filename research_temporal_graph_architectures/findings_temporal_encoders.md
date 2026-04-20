# Temporal Encoder Architectures for Microservice Anomaly Detection

Research survey conducted 2026-04-09. Focus: practical temporal encoders for
small-dataset microservice fault classification, with reported parameter counts.

---

## 1. Lightweight 1D-CNN: LITE (Light Inception with boosTing tEchniques)

**Source:** Ismail-Fawaz et al., DSAA 2023; extended journal version 2024
([arXiv 2409.02869](https://arxiv.org/html/2409.02869v1),
[Springer](https://link.springer.com/article/10.1007/s41060-024-00708-5))

### Architecture
- Three-layer 1D-CNN using DepthWise Separable Convolutions (DWSC)
- Layer 1: standard convolutions with multiplexing (6 parallel branches,
  kernels {2, 4, 8, 16, 32, 64}) + hand-crafted custom filters
- Layers 2-3: DWSC with increasing dilation rates (D0=1, D1=2, D2=4), N=32 filters
- Final: Global Average Pooling over time axis -> FC classification head
- Boosted by: multiplexing, custom filters (moving average, diff), dilated convolutions

### Parameter counts
| Model        | Parameters  | Relative to LITE |
|--------------|-------------|------------------|
| **LITE**     | **9,814**   | 1x               |
| FCN          | 264,704     | 27x              |
| ResNet       | 504,000     | 51x              |
| InceptionTime| 420,192     | 43x              |

LITE = 2.34% of InceptionTime's parameters.

### Performance
- UCR archive (128 univariate datasets): statistically not significantly
  different from ResNet; ensemble LITETime ranks 3rd overall
- Multivariate (Kimore rehab, 71 samples/exercise): LITEMVTime 80% avg
  accuracy vs InceptionTime 77.33% -- **better on very small datasets**
- Faster training and inference, lower CO2/power than InceptionTime

### Relevance to CAAA
- With ~9.8K params, LITE is viable for <2000-sample regimes without overfitting
- Depthwise separable convolutions with multi-scale kernels are directly
  applicable as a per-service temporal feature extractor
- The multiplexing (parallel branches at different scales) captures both
  short transients (kernel=2) and longer trends (kernel=64)

---

## 2. TSRM: Lightweight CNN-based Temporal Feature Encoding

**Source:** Li et al., 2025 ([arXiv 2504.18878](https://arxiv.org/html/2504.18878))

### Architecture
- K independent 1D CNN layers with varying kernel sizes for multi-scale extraction:
  - Small kernels without dilation -> basic features (spikes, steps)
  - Medium kernels with minimal dilation -> intermediate patterns
  - Large kernels with significant dilation -> trends, seasonality
- Followed by attention-based feature aggregation layers
- Designed for both forecasting and imputation (encoder is task-agnostic)

### Key design choices
- Uses CNN instead of FFT for temporal representation -> lower memory, fewer params
- Outperforms state-of-the-art on most benchmarks while "significantly reducing
  complexity in the form of learnable parameters"
- No exact parameter count published, but described as "low memory profile"

### Relevance to CAAA
- Confirms that multi-scale 1D-CNN with attention aggregation is a
  competitive temporal encoder pattern
- The attention aggregation over CNN outputs is analogous to what we might
  use to weight different temporal scales before graph propagation

---

## 3. Hybrid CNN-LSTM with Attention for Anomaly Detection

**Source:** Lu et al., 2023 -- Multiscale C-LSTM
([Wiley](https://onlinelibrary.wiley.com/doi/10.1155/2023/6597623));
also Transformer+1D-CNN hybrid
([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197623001483))

### Pattern: CNN reduces frequency, LSTM extracts temporal features
- 1D-CNN with multiple kernel sizes (3, 5, 7) captures local correlations
- Bidirectional LSTM (128 hidden units) processes CNN output sequences
- Temporal Attention Gates (TAGs) fine-tune which time steps matter
- Feature and location attention on CNN outputs enhance key feature effects

### Typical configurations for microservices
- CNN kernels: {3, 5, 7} with 32-64 filters each
- BiLSTM: 128 hidden units (2 layers)
- Attention: single-head or multi-head (4 heads) over LSTM outputs
- Approximate parameter range: 50K-200K depending on input dimensionality

### Performance
- Multi-head self-attention CNN-LSTM: 85.3% accuracy on temporal pattern detection
- DeepAnt (CNN-only): effective with minimal training data, handles <5%
  contamination in unsupervised setup

---

## 4. Spatio-Temporal Graph Models for Microservice RCA

### 4a. D-GNN + LSTM for Microservice Anomaly Detection

**Source:** Feng et al., Symmetry 2026
([MDPI](https://www.mdpi.com/2073-8994/18/1/87))

- Dynamic Graph Neural Network combined with LSTM
- Models structural dependencies (service call graph) AND temporal evolution
- CPU-only inference (no GPU required) -- relevant for production deployment
- Per-service temporal encoding via LSTM, then message passing on service graph

### 4b. Grace: Spatial-Temporal GCN for Microservices

**Source:** Referenced in comprehensive RCA survey
([arXiv 2408.00803](https://arxiv.org/html/2408.00803v1))

- Spatial-temporal graph convolutional network
- Binary classifies each microservice as faulty/healthy
- Includes interpreter model for explainability
- Uses GCN layers for spatial (graph topology) + temporal convolutions

### 4c. ServiceGraph-FM: Pretrained Graph Encoder for RCA

**Source:** [ResearchGate](https://www.researchgate.net/publication/399590835)

- Self-supervised graph encoder pretrained on large-scale production traces
- Temporal relational diffusion for capturing time-varying service interactions
- Foundation model approach: pretrain on cluster traces, fine-tune for diagnosis

### 4d. MetricSage: GNN for Per-Service Metrics

**Source:** MicroIRC, [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0164121224001900)

- GNN model that fits metrics features within each anomaly window
- Instance-level RCA: 93.1% top-5 accuracy
- >17% improvement at service level, >11.5% at instance level vs SOTA

### 4e. ST-GraphRCA for IoT/Edge (transferable pattern)

**Source:** [MDPI Sensors](https://www.mdpi.com/1424-8220/26/5/1474)

- Spatio-temporal graph for root cause analysis with propagation modeling
- Relevant architectural pattern: temporal encoder per node -> graph attention
  across nodes -> temporal decoder for anomaly scoring

---

## 5. Temporal Attention for Log/Trace Anomaly Detection

**Source:** TLA-Net, PMC 2024
([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679089/))

- Temporal Logical Attention Network for log-based anomaly detection
- Captures sequential and logical dependencies in distributed system logs
- Lightweight attention mechanism designed for streaming/online detection

Also: integrated model combining Temporal Graph Attention with
Transformer-augmented RNNs
([Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-85822-5))

---

## 6. Tree Baselines vs Neural Temporal Models on Small Data

### Key findings from surveys

No single paper provides a clean head-to-head comparison of temporal neural
models vs tree baselines on small (<2000 sample) tabular time-series data.
However, converging evidence from multiple sources:

1. **LITE on small datasets (71 samples):** Outperforms InceptionTime,
   suggesting parameter-efficient CNNs can work on very small datasets

2. **Latent representation + RF/SVM pattern:** Generative models learn a
   time-series self-predictor; latent representation is then fed to Random
   Forest or SVM for classification. This hybrid approach is competitive
   ([arXiv survey on deep learning for TSC](https://arxiv.org/pdf/1809.04356))

3. **General small-data regime consensus:** Tree models (XGBoost, RF) remain
   strong baselines on tabular features derived from time series. Neural
   approaches need either:
   - Very few parameters (<10K, like LITE)
   - Strong inductive bias (1D-CNN with proper kernel sizes)
   - Pretraining/transfer from related tasks
   - Ensemble strategies to reduce variance

4. **Practical guidance from AIOps literature:** Most production microservice
   RCA systems use tree models or shallow neural nets on engineered features,
   reserving deep temporal models for settings with abundant telemetry data
   (>10K samples) or when temporal patterns cannot be captured by summary
   statistics.

---

## 7. Architectural Recommendations for CAAA

Based on this survey, candidate temporal encoder architectures ranked by
suitability for CAAA's regime (~500-2000 samples, 5-50 services, per-service
metric time windows):

### Option A: LITE-style Multi-scale 1D-CNN (Recommended)
- ~10K parameters per service encoder
- Parallel depthwise-separable conv branches at scales {3, 5, 7, 11}
- Global average pooling -> fixed-dim embedding per service
- Pros: proven on small datasets, interpretable scales, very fast
- Cons: no explicit recurrence (may miss long-range order effects)

### Option B: Tiny Temporal Attention (1D-CNN + Single-Head Attention)
- CNN backbone (3 layers, 32 filters, kernels {3,5,7}) -> attention pooling
- ~15-25K parameters
- Attention weights provide temporal interpretability
- Pros: learns which time steps matter; works with variable-length windows
- Cons: slightly more parameters; attention on short sequences may not help

### Option C: Shallow BiLSTM with Attention
- BiLSTM(64 hidden) -> single-head attention -> FC
- ~30-50K parameters
- Pros: captures sequential ordering explicitly
- Cons: more parameters, slower, harder to parallelize across services

### Option D: Hybrid CNN-encoder + Graph Propagation
- Per-service: LITE-style 1D-CNN producing d-dim embedding
- Cross-service: 1-2 layers of GAT/GCN on service dependency graph
- Total: ~15K (temporal) + ~5K (graph) = ~20K parameters
- Pros: captures both temporal and structural patterns
- Cons: requires known/inferred service graph; more complex training

### Summary Table

| Architecture | Params | Small-data fit | Captures ordering | Graph-aware |
|---|---|---|---|---|
| LITE 1D-CNN | ~10K | Excellent | Partial (via kernel) | No |
| CNN+Attention | ~20K | Good | Yes (attention) | No |
| Shallow BiLSTM | ~40K | Moderate | Yes (recurrence) | No |
| CNN+GAT hybrid | ~20K | Good | Partial | Yes |
| Current CAAA (RF) | ~N/A | Excellent | No | Via features |

---

## Sources

- [LITE: Light Inception with boosTing tEchniques (arXiv)](https://arxiv.org/html/2409.02869v1)
- [LITE journal version (Springer)](https://link.springer.com/article/10.1007/s41060-024-00708-5)
- [TSRM: Lightweight Temporal Feature Encoding (arXiv)](https://arxiv.org/html/2504.18878)
- [Multiscale C-LSTM for Anomaly Detection (Wiley)](https://onlinelibrary.wiley.com/doi/10.1155/2023/6597623)
- [Transformer + 1D-CNN for Time Series Anomaly (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0952197623001483)
- [D-GNN + LSTM Microservice Anomaly Detection (MDPI)](https://www.mdpi.com/2073-8994/18/1/87)
- [Comprehensive RCA Survey (arXiv)](https://arxiv.org/html/2408.00803v1)
- [ServiceGraph-FM (ResearchGate)](https://www.researchgate.net/publication/399590835)
- [MicroIRC / MetricSage (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0164121224001900)
- [ST-GraphRCA (MDPI Sensors)](https://www.mdpi.com/1424-8220/26/5/1474)
- [TLA-Net Temporal Logical Attention (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679089/)
- [Temporal Graph Attention + Transformer RNN (Nature)](https://www.nature.com/articles/s41598-025-85822-5)
- [Deep Learning for TSC Survey (arXiv)](https://arxiv.org/pdf/1809.04356)
- [Deep Learning for Time Series Anomaly Detection Survey (ACM)](https://dl.acm.org/doi/full/10.1145/3691338)
- [GNN for Anomaly Detection Systematic Review (Springer)](https://link.springer.com/article/10.1007/s10462-026-11532-7)
- [InceptionTime (arXiv)](https://arxiv.org/abs/1909.04939)
