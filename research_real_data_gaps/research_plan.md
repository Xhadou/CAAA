# Research Plan: Improving CAAA on Real Data

## Problem
On RCAEval real data, tree baselines (CatBoost 93.2%) beat Full CAAA (84.5%) by 8.7 points. Worse, No Context Features (87.2%) beats Full CAAA — meaning the context consistency loss HURTS on real data. The core issue: context is synthetically assigned to real traces, creating a noisy signal that misleads the neural model while trees ignore it.

## Subtopics to Research

### 1. Neural vs Tree Baselines on Tabular Data
Why do trees consistently beat neural networks on tabular/structured data? What recent advances (TabPFN, FT-Transformer, TabNet, SAINT) close this gap? Is this a known problem in the ML literature and what do papers recommend?

### 2. Context Feature Engineering from Real Telemetry
Instead of synthetically assigning context, can we DERIVE context features from the real metrics themselves? E.g., detecting load patterns, periodicity, cross-service correlation as proxy context. How do self-supervised and contrastive approaches extract context from unlabeled data?

### 3. Transfer Learning: Synthetic Pre-training → Real Fine-tuning
Can we pre-train CAAA on synthetic data (where context works well) and fine-tune on real data? Domain adaptation techniques for time-series anomaly detection. How do papers handle the synthetic-to-real transfer gap?
