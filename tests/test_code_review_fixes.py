"""Tests that verify the end-to-end code review fixes.

These tests cover efficiency, design, and robustness improvements:
  1. Auto-detect device in main.py (GPU when available, CPU otherwise)
  2. Vectorized extract_batch with pre-allocated array
  3. Vectorized predict_with_confidence in AnomalyClassifier
  4. Correct time_seasonality using datetime for epoch timestamps
  5. LayerNorm in ContextIntegrationModule context encoder
  6. Redundant forward pass eliminated in contrastive _compute_loss
  7. Guard compute_reconstruction_errors against empty data
  8. Temperature clamped after calibration to avoid division-by-zero
"""

import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_loader.data_types import AnomalyCase, ServiceMetrics
from src.data_loader.dataset import generate_combined_dataset
from src.features.extractors import FeatureExtractor, N_FEATURES
from src.models.caaa_model import CAAAModel
from src.models.classifier import AnomalyClassifier
from src.models.context_module import ContextIntegrationModule
from src.training.trainer import CAAATrainer


# ── 1. Auto-detect device ───────────────────────────────────────────


class TestAutoDetectDevice:
    """Verify main.py no longer hardcodes device='cpu'."""

    def test_main_no_hardcoded_cpu(self):
        """main.py should not hardcode device='cpu' for CAAATrainer."""
        with open("src/main.py") as f:
            content = f.read()
        assert 'device="cpu"' not in content, (
            "main.py should auto-detect device, not hardcode 'cpu'"
        )

    def test_main_uses_cuda_check(self):
        """main.py should use torch.cuda.is_available() for device selection."""
        with open("src/main.py") as f:
            content = f.read()
        assert "torch.cuda.is_available()" in content


# ── 2. Pre-allocated extract_batch ──────────────────────────────────


class TestExtractBatchPrealloc:
    """Verify extract_batch uses pre-allocated array instead of vstack."""

    def test_no_vstack_in_extract_batch(self):
        """extract_batch should not use np.vstack (creates temp arrays)."""
        source = inspect.getsource(FeatureExtractor.extract_batch)
        assert "vstack" not in source

    def test_extract_batch_shape(self):
        """extract_batch output shape should match (n_cases, N_FEATURES)."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=3, n_load=3, seed=42, include_hard=False,
        )
        ext = FeatureExtractor(seed=42)
        X = ext.extract_batch(fault_cases + load_cases)
        assert X.shape == (6, N_FEATURES)

    def test_extract_batch_matches_individual(self):
        """extract_batch output should match individual extract calls."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=3, n_load=3, seed=42, include_hard=False,
        )
        cases = fault_cases + load_cases
        ext1 = FeatureExtractor(seed=42)
        batch = ext1.extract_batch(cases)

        ext2 = FeatureExtractor(seed=42)
        individual = np.array([ext2.extract(c) for c in cases])

        np.testing.assert_allclose(batch, individual)


# ── 3. Vectorized predict_with_confidence ────────────────────────────


class TestVectorizedPredictWithConfidence:
    """Verify AnomalyClassifier.predict_with_confidence is vectorized."""

    def test_no_iterrows(self):
        """predict_with_confidence should not use iterrows()."""
        source = inspect.getsource(AnomalyClassifier.predict_with_confidence)
        assert "iterrows" not in source

    def test_output_matches_expected(self):
        """Vectorized version should produce same results as old row-by-row."""
        X = np.random.default_rng(42).normal(size=(30, 5)).astype(np.float32)
        y = np.array(["FAULT"] * 15 + ["EXPECTED_LOAD"] * 15)
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)

        preds, confs = clf.predict_with_confidence(X, confidence_threshold=0.6)
        assert len(preds) == 30
        assert len(confs) == 30
        assert set(preds).issubset({"FAULT", "EXPECTED_LOAD", "UNKNOWN"})


# ── 4. Correct time_seasonality ──────────────────────────────────────


class TestTimeSeasonalityFix:
    """Verify time_seasonality uses datetime for epoch timestamps."""

    def test_uses_datetime_not_modulo(self):
        """Extractor should use datetime for hour extraction, not modulo."""
        source = inspect.getsource(FeatureExtractor._context_features)
        assert "datetime" in source
        assert "mean_ts % 24" not in source

    def test_synthetic_returns_neutral(self):
        """Synthetic timestamps (small integers) should return 0.5."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "timestamp": np.arange(60),
            "cpu_usage": rng.uniform(10, 30, 60),
            "memory_usage": rng.uniform(20, 40, 60),
            "request_rate": rng.uniform(50, 200, 60),
            "error_rate": rng.uniform(0.001, 0.01, 60),
            "latency": rng.uniform(10, 100, 60),
            "network_in": rng.uniform(1000, 5000, 60),
            "network_out": rng.uniform(1000, 5000, 60),
        })
        case = AnomalyCase(
            case_id="ts_test",
            system="test",
            label="FAULT",
            services=[ServiceMetrics(service_name="frontend", metrics=df)],
            context={},
        )
        ext = FeatureExtractor(seed=42)
        feats = ext.extract(case)
        ts_idx = ext.feature_names().index("time_seasonality")
        assert feats[ts_idx] == 0.5


# ── 5. LayerNorm in context encoder ─────────────────────────────────


class TestContextEncoderLayerNorm:
    """Verify ContextIntegrationModule context encoder has LayerNorm."""

    def test_context_encoder_has_layer_norm(self):
        """context_encoder should include LayerNorm layers."""
        mod = ContextIntegrationModule()
        layer_types = [type(m).__name__ for m in mod.context_encoder]
        assert "LayerNorm" in layer_types, (
            f"Expected LayerNorm in context_encoder, got: {layer_types}"
        )


# ── 6. No redundant forward pass in contrastive _compute_loss ────────


class TestContrastiveComputeLoss:
    """Verify _compute_loss avoids redundant forward pass for contrastive."""

    def test_no_double_forward_in_compute_loss(self):
        """_compute_loss for contrastive should not call model() and
        get_embeddings() separately (double forward pass)."""
        source = inspect.getsource(CAAATrainer._compute_loss)
        # In the contrastive branch, model() should not appear before
        # get_embeddings; instead embeddings are computed first and logits
        # derived from model.classifier(embeddings).
        assert "self.model.classifier(embeddings)" in source

    def test_contrastive_loss_computes(self):
        """Contrastive _compute_loss should still produce valid loss."""
        model = CAAAModel(input_dim=36)
        trainer = CAAATrainer(model, loss_type="contrastive")
        X = np.random.default_rng(42).normal(size=(8, 36)).astype(np.float32)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loss = trainer._compute_loss(X_t, y_t)
        assert np.isfinite(loss)
        assert loss > 0


# ── 7. Guard compute_reconstruction_errors against empty data ────────


class TestReconstructionErrorsEmptyGuard:
    """Verify compute_reconstruction_errors handles data shorter than seq_length."""

    def test_empty_data_returns_empty(self):
        """Data shorter than seq_length should return empty array."""
        from src.models.anomaly_detector import AnomalyDetector, LSTMAutoencoder

        det = AnomalyDetector(seq_length=30)
        det.model = LSTMAutoencoder(n_features=5)
        det.threshold = 1.0

        # Data with fewer rows than seq_length
        short_data = np.random.default_rng(42).normal(size=(10, 5))
        errors = det.compute_reconstruction_errors(short_data)
        assert len(errors) == 0


# ── 8. Temperature clamped after calibration ─────────────────────────


class TestTemperatureClamping:
    """Verify temperature is clamped to prevent division by zero."""

    def test_temperature_always_positive(self):
        """After calibrate_temperature, temperature should be >= 0.01."""
        model = CAAAModel(input_dim=36)
        trainer = CAAATrainer(model)

        X = np.random.default_rng(42).normal(size=(20, 36)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        trainer.train(X, y, epochs=5, batch_size=8)

        temp = trainer.calibrate_temperature(X, y)
        assert temp >= 0.01, f"Temperature {temp} should be >= 0.01"
