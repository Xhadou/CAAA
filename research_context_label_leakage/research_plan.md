# Research Plan: Context Label Leakage in Anomaly Detection Evaluation

## Main Research Question
How do anomaly detection and fault attribution papers prevent context/metadata features from becoming perfect label proxies in evaluation? When one class always has operational context (e.g., "scheduled event") and the other never does, models learn the metadata signal rather than the underlying metric pattern. How is this handled in the literature?

## Subtopics

### 1. Label Leakage Prevention in ML/Time-Series
- How is feature/label leakage defined and prevented in general ML?
- Specific techniques for time-series classification
- Standard evaluation practices when auxiliary features correlate with labels

### 2. Context-Aware AIOps and Fault Detection Methodology
- How do AIOps papers (DejaVu, Lumos, BARO, Eadro) handle operational context?
- Do they assign context to both classes or only one?
- How do they prevent context from being a trivial classifier?

### 3. Counterfactual and Causal Evaluation in Anomaly Detection
- Causal inference approaches: how to evaluate when context is an intervention
- Counterfactual evaluation: testing "what if the context were different?"
- Partial context assignment strategies in evaluation protocols
