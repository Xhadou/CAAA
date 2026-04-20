# Research Plan: Temporal and Graph Architectures for Microservice Anomaly Attribution

## Main Question
What neural architectures can process raw time-series and service graph structure for anomaly attribution in microservices, giving an advantage over tree-based models that can only operate on flat feature vectors?

## Subtopics

### 1. GNN/GAT for Service Dependency Graph Fault Localization
- Papers: MicroRCA, TraceRCA, DiagFusion, CausIL
- How they construct service dependency graphs
- Message passing approaches for fault propagation
- How they handle small graphs (5-15 nodes)

### 2. Temporal Encoders for Microservice Anomaly Detection
- LSTM, GRU, Temporal Attention, 1D-CNN approaches
- How they encode per-service time series
- Handling variable-length sequences
- Lightweight approaches for small datasets

### 3. Combined Temporal-Graph Architectures
- Architectures that fuse temporal encoding with graph structure
- Spatio-temporal GNNs (ST-GNN) for AIOps
- How they handle the service × timestep × metric tensor
- Practical implementations and parameter counts

## Expected Synthesis
- Which architecture gives the best advantage over trees on small datasets
- Parameter-efficient designs suitable for <2000 training samples
- How to integrate with existing FiLM context conditioning
