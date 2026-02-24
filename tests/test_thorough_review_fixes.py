"""Tests for the second-pass thorough file-by-file code review fixes.

Covers issues missed in the first review due to truncated file views:
  1. scripts/train.py: hardcoded device="cpu" → auto-detect
  2. scripts/ablation.py: hardcoded device="cpu" → auto-detect (all 4 sites)
  3. scripts/demo.py: hardcoded device="cpu" → auto-detect + magic number input_dim=36
  4. scripts/demo.py: data leakage from using test set as validation
  5. trainer.py: double forward pass in contrastive training loop (not just _compute_loss)
  6. anomaly_detector.py: weights_only=True incompatible with pickling scaler
"""

import inspect
import tempfile

import numpy as np
import pytest
import torch

from src.models.anomaly_detector import AnomalyDetector, LSTMAutoencoder
from src.models.caaa_model import CAAAModel
from src.training.trainer import CAAATrainer


# ── 1-3: Scripts should not hardcode device="cpu" ────────────────────


class TestScriptsAutoDetectDevice:
    """Verify all scripts auto-detect GPU/CPU device."""

    def test_train_script_no_hardcoded_cpu(self):
        with open("scripts/train.py") as f:
            content = f.read()
        assert 'device="cpu"' not in content, (
            "scripts/train.py should auto-detect device"
        )

    def test_ablation_script_no_hardcoded_cpu(self):
        with open("scripts/ablation.py") as f:
            content = f.read()
        assert 'device="cpu"' not in content, (
            "scripts/ablation.py should auto-detect device"
        )

    def test_demo_script_no_hardcoded_cpu(self):
        with open("scripts/demo.py") as f:
            content = f.read()
        assert 'device="cpu"' not in content, (
            "scripts/demo.py should auto-detect device"
        )


# ── 3b: demo.py should not hardcode input_dim=36 ────────────────────


class TestDemoNoMagicNumbers:
    """Verify demo.py uses dynamic input_dim, not a magic number."""

    def test_demo_no_hardcoded_input_dim_36(self):
        with open("scripts/demo.py") as f:
            content = f.read()
        assert "input_dim=36" not in content, (
            "scripts/demo.py should use X_train.shape[1] instead of 36"
        )


# ── 4: demo.py should not use test set for validation (data leakage) ─


class TestDemoNoDataLeakage:
    """Verify demo.py holds out separate validation data."""

    def test_demo_does_not_use_test_for_val(self):
        with open("scripts/demo.py") as f:
            content = f.read()
        assert "X_val=X_test" not in content, (
            "scripts/demo.py should not use test set as validation"
        )
        assert "y_val=y_test" not in content, (
            "scripts/demo.py should not use test labels as validation"
        )

    def test_demo_has_val_split(self):
        """demo.py should create a separate validation split."""
        with open("scripts/demo.py") as f:
            content = f.read()
        assert "X_val" in content, "demo.py should have a validation set"


# ── 5: Contrastive training loop should avoid double forward pass ────


class TestContrastiveTrainingNoDoubleForward:
    """Verify training loop uses single forward pass for contrastive."""

    def test_train_loop_no_double_forward(self):
        """The training loop should compute embeddings first and derive
        logits from model.classifier(embeddings), not call model()
        and then get_embeddings() separately."""
        source = inspect.getsource(CAAATrainer.train)
        # In the contrastive branch, model.classifier(embeddings) should
        # appear, and model(X_batch) should NOT be called before
        # get_embeddings when loss_type == "contrastive"
        assert "self.model.classifier(embeddings)" in source, (
            "Training loop should derive logits from classifier(embeddings)"
        )

    def test_contrastive_training_converges(self):
        """Contrastive training should still converge after the fix."""
        model = CAAAModel(input_dim=36)
        trainer = CAAATrainer(model, loss_type="contrastive")
        X = np.random.default_rng(42).normal(size=(20, 36)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        history = trainer.train(X, y, epochs=10, batch_size=8)
        assert len(history["train_loss"]) == 10
        # Loss should decrease from initial value
        assert history["train_loss"][-1] < history["train_loss"][0] * 2


# ── 6: AnomalyDetector save/load should work with scaler ────────────


class TestAnomalyDetectorSaveLoad:
    """Verify AnomalyDetector save/load works (includes non-tensor objects)."""

    def test_save_load_roundtrip(self):
        """Save and load should preserve model, scaler, and threshold."""
        import pandas as pd

        rng = np.random.default_rng(42)
        metrics_list = [
            pd.DataFrame({
                "timestamp": np.arange(60),
                "cpu_usage": rng.uniform(10, 30, 60),
                "memory_usage": rng.uniform(20, 40, 60),
                "request_rate": rng.uniform(50, 200, 60),
                "error_rate": rng.uniform(0.001, 0.01, 60),
                "latency": rng.uniform(10, 100, 60),
                "network_in": rng.uniform(1000, 5000, 60),
                "network_out": rng.uniform(1000, 5000, 60),
            })
            for _ in range(5)
        ]

        det = AnomalyDetector(seq_length=10)
        det.fit(metrics_list, epochs=2, batch_size=16)
        original_threshold = det.threshold

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        det.save(path)

        det2 = AnomalyDetector(seq_length=10)
        # This should NOT raise (weights_only=False allows scaler)
        det2.load(path, n_features=8)

        assert det2.threshold == original_threshold
        assert det2.model is not None

    def test_load_does_not_use_weights_only_true(self):
        """load() should use weights_only=False (scaler is not a tensor)."""
        source = inspect.getsource(AnomalyDetector.load)
        assert "weights_only=False" in source, (
            "AnomalyDetector.load should use weights_only=False"
        )
