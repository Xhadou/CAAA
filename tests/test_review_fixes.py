"""Tests verifying code review fixes and performance improvements.

Consolidates tests from the three separate review-fix files into a
single organized module.  Each test class documents which fix it
validates.

Sections:
  1. Auto-detect device (main.py and scripts/)
  2. Feature extraction optimizations (vectorization, pre-allocation)
  3. Time seasonality fix (datetime for epoch timestamps)
  4. Context module architecture (LayerNorm)
  5. Contrastive loss (no double forward pass)
  6. Anomaly detector (save/load, empty data guard)
  7. Temperature calibration (clamping)
  8. Classifier type hints and exports
  9. AD pre-filter (all services)
  10. Demo script (data leakage, magic numbers)
  11. Evaluate method (contrastive loss path)
  12. Statistical features (numpy optimization)
"""

import inspect
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_loader.data_types import AnomalyCase, ServiceMetrics
from src.data_loader.dataset import generate_combined_dataset
from src.evaluation.metrics import compute_all_metrics
from src.features.extractors import FeatureExtractor, N_FEATURES
from src.features.feature_schema import N_FEATURES as SCHEMA_N_FEATURES
from src.models.anomaly_detector import AnomalyDetector, LSTMAutoencoder
from src.models.caaa_model import CAAAModel
from src.models.classifier import AnomalyClassifier
from src.models.context_module import ContextIntegrationModule
from src.training.trainer import CAAATrainer


# ── 1. Auto-detect device ───────────────────────────────────────────


class TestAutoDetectDevice:
    """All entry points should auto-detect GPU/CPU, not hardcode device."""

    def test_main_no_hardcoded_cpu(self):
        with open("src/main.py") as f:
            content = f.read()
        assert 'device="cpu"' not in content

    def test_main_uses_cuda_check(self):
        with open("src/main.py") as f:
            content = f.read()
        assert "torch.cuda.is_available()" in content

    def test_train_script_no_hardcoded_cpu(self):
        with open("scripts/train.py") as f:
            content = f.read()
        assert 'device="cpu"' not in content

    def test_ablation_script_no_hardcoded_cpu(self):
        with open("scripts/ablation.py") as f:
            content = f.read()
        assert 'device="cpu"' not in content

    def test_demo_script_no_hardcoded_cpu(self):
        with open("scripts/demo.py") as f:
            content = f.read()
        assert 'device="cpu"' not in content


# ── 2. Feature extraction optimizations ─────────────────────────────


class TestExtractBatchPrealloc:
    """extract_batch should use pre-allocated array instead of vstack."""

    def test_no_vstack_in_extract_batch(self):
        source = inspect.getsource(FeatureExtractor.extract_batch)
        assert "vstack" not in source

    def test_extract_batch_shape(self):
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=3, n_load=3, seed=42, include_hard=False,
        )
        ext = FeatureExtractor(seed=42)
        X = ext.extract_batch(fault_cases + load_cases)
        assert X.shape == (6, N_FEATURES)

    def test_extract_batch_matches_individual(self):
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=3, n_load=3, seed=42, include_hard=False,
        )
        cases = fault_cases + load_cases
        ext1 = FeatureExtractor(seed=42)
        batch = ext1.extract_batch(cases)

        ext2 = FeatureExtractor(seed=42)
        individual = np.array([ext2.extract(c) for c in cases])

        np.testing.assert_allclose(batch, individual)


class TestVectorizedPredictWithConfidence:
    """AnomalyClassifier.predict_with_confidence should be vectorized."""

    def test_no_iterrows(self):
        source = inspect.getsource(AnomalyClassifier.predict_with_confidence)
        assert "iterrows" not in source

    def test_output_matches_expected(self):
        X = np.random.default_rng(42).normal(size=(30, 5)).astype(np.float32)
        y = np.array(["FAULT"] * 15 + ["EXPECTED_LOAD"] * 15)
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)

        preds, confs = clf.predict_with_confidence(X, confidence_threshold=0.6)
        assert len(preds) == 30
        assert len(confs) == 30
        assert set(preds).issubset({"FAULT", "EXPECTED_LOAD", "UNKNOWN"})


class TestCrossServiceSyncVectorized:
    """cross_service_sync should use vectorized numpy correlation."""

    def test_vectorized_correlation_in_source(self):
        source = inspect.getsource(FeatureExtractor._workload_features)
        assert "np.corrcoef" in source
        assert "for j in range(i + 1, n):" not in source

    def test_cross_service_sync_value_range(self):
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=5, n_load=5, seed=42, include_hard=False,
        )
        ext = FeatureExtractor(seed=42)
        names = ext.feature_names()
        css_idx = names.index("cross_service_sync")
        for case in fault_cases + load_cases:
            feats = ext.extract(case)
            assert -1.0 <= feats[css_idx] <= 1.0

    def test_cross_service_sync_single_service(self):
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
            case_id="test_single",
            system="test",
            label="FAULT",
            services=[ServiceMetrics(service_name="frontend", metrics=df)],
        )
        ext = FeatureExtractor(seed=42)
        feats = ext.extract(case)
        css_idx = ext.feature_names().index("cross_service_sync")
        assert feats[css_idx] == 0.0


# ── 3. Time seasonality fix ─────────────────────────────────────────


class TestTimeSeasonalityFix:
    """time_seasonality should use datetime for epoch timestamps."""

    def test_uses_datetime_not_modulo(self):
        source = inspect.getsource(FeatureExtractor._context_features)
        assert "datetime" in source
        assert "mean_ts % 24" not in source

    def test_synthetic_returns_neutral(self):
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


# ── 4. Context module architecture ──────────────────────────────────


class TestContextEncoderLayerNorm:
    """ContextIntegrationModule context encoder should include LayerNorm."""

    def test_context_encoder_has_layer_norm(self):
        mod = ContextIntegrationModule()
        layer_types = [type(m).__name__ for m in mod.context_encoder]
        assert "LayerNorm" in layer_types


# ── 5. Contrastive loss — no double forward pass ────────────────────


class TestContrastiveNoDoubleForward:
    """Both _compute_loss and train() should avoid redundant forward pass."""

    def test_no_double_forward_in_compute_loss(self):
        source = inspect.getsource(CAAATrainer._compute_loss)
        assert "self.model.classifier(embeddings)" in source

    def test_no_double_forward_in_train_loop(self):
        source = inspect.getsource(CAAATrainer.train)
        assert "self.model.classifier(embeddings)" in source

    def test_contrastive_loss_computes(self):
        model = CAAAModel(input_dim=36)
        trainer = CAAATrainer(model, loss_type="contrastive")
        X = np.random.default_rng(42).normal(size=(8, 36)).astype(np.float32)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loss = trainer._compute_loss(X_t, y_t)
        assert np.isfinite(loss)
        assert loss > 0


# ── 6. Anomaly detector ─────────────────────────────────────────────


class TestAnomalyDetectorSaveLoad:
    """save/load should work with non-tensor objects (scaler, threshold)."""

    def test_save_load_roundtrip(self):
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
        det2.load(path, n_features=8)
        assert det2.threshold == original_threshold
        assert det2.model is not None

    def test_load_does_not_use_weights_only_true(self):
        source = inspect.getsource(AnomalyDetector.load)
        assert "weights_only=False" in source


class TestReconstructionErrorsEmptyGuard:
    """compute_reconstruction_errors should handle data shorter than seq_length."""

    def test_empty_data_returns_empty(self):
        det = AnomalyDetector(seq_length=30)
        det.model = LSTMAutoencoder(n_features=5)
        det.threshold = 1.0
        short_data = np.random.default_rng(42).normal(size=(10, 5))
        errors = det.compute_reconstruction_errors(short_data)
        assert len(errors) == 0


# ── 7. Temperature calibration ──────────────────────────────────────


class TestTemperatureClamping:
    """Temperature should be clamped to prevent division by zero."""

    def test_temperature_always_positive(self):
        model = CAAAModel(input_dim=36)
        trainer = CAAATrainer(model)
        X = np.random.default_rng(42).normal(size=(20, 36)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        trainer.train(X, y, epochs=5, batch_size=8)
        temp = trainer.calibrate_temperature(X, y)
        assert temp >= 0.01


# ── 8. Classifier type hints and exports ────────────────────────────


class TestClassifierTypeHints:
    """AnomalyClassifier should use np.ndarray type hints."""

    def test_fit_accepts_ndarray(self):
        X = np.random.default_rng(42).normal(size=(20, 5)).astype(np.float32)
        y = np.array(["FAULT"] * 10 + ["EXPECTED_LOAD"] * 10)
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        assert clf.is_fitted

    def test_fit_still_accepts_dataframe(self):
        X = pd.DataFrame(
            np.random.default_rng(42).normal(size=(20, 5)),
            columns=[f"f{i}" for i in range(5)],
        )
        y = np.array(["FAULT"] * 10 + ["EXPECTED_LOAD"] * 10)
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        assert clf.is_fitted

    def test_predict_accepts_ndarray(self):
        X = np.random.default_rng(42).normal(size=(20, 5)).astype(np.float32)
        y = np.array(["FAULT"] * 10 + ["EXPECTED_LOAD"] * 10)
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        preds = clf.predict(X)
        assert len(preds) == 20

    def test_fit_type_hint_is_ndarray(self):
        sig = inspect.signature(AnomalyClassifier.fit)
        x_annotation = sig.parameters["X"].annotation
        assert x_annotation is np.ndarray


class TestInitExports:
    """__init__.py files should define __all__."""

    def test_evaluation_init_has_all(self):
        import src.evaluation
        assert hasattr(src.evaluation, "__all__")

    def test_training_init_has_all(self):
        import src.training
        assert hasattr(src.training, "__all__")


# ── 9. AD pre-filter (all services) ────────────────────────────────


class TestADPreFilterAllServices:
    """Anomaly detector pre-filter should check all services."""

    def test_train_script_checks_all_services(self):
        with open("scripts/train.py") as f:
            content = f.read()
        assert "case.services[0]" not in content
        assert "for svc in case.services" in content

    def test_main_checks_all_services(self):
        with open("src/main.py") as f:
            content = f.read()
        assert "case.services[0]" not in content


# ── 10. Demo script (data leakage, magic numbers) ───────────────────


class TestDemoScript:
    """demo.py should not leak test data or use magic numbers."""

    def test_demo_no_hardcoded_input_dim_36(self):
        with open("scripts/demo.py") as f:
            content = f.read()
        assert "input_dim=36" not in content

    def test_demo_does_not_use_test_for_val(self):
        with open("scripts/demo.py") as f:
            content = f.read()
        assert "X_val=X_test" not in content
        assert "y_val=y_test" not in content

    def test_demo_has_val_split(self):
        with open("scripts/demo.py") as f:
            content = f.read()
        assert "X_val" in content


# ── 11. Evaluate method (contrastive loss path) ────────────────────


class TestEvaluateContrastiveLoss:
    """evaluate() should handle all loss types, including contrastive."""

    def test_evaluate_with_contrastive_loss(self):
        model = CAAAModel(input_dim=36)
        trainer = CAAATrainer(model, loss_type="contrastive")
        X = np.random.default_rng(42).normal(size=(20, 36)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        trainer.train(X, y, epochs=5, batch_size=8)
        result = trainer.evaluate(X, y)
        assert np.isfinite(result["loss"])
        assert 0 <= result["accuracy"] <= 1

    def test_evaluate_with_context_consistency_loss(self):
        model = CAAAModel(input_dim=36)
        trainer = CAAATrainer(model, use_context_loss=True)
        X = np.random.default_rng(42).normal(size=(20, 36)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        trainer.train(X, y, epochs=5, batch_size=8)
        result = trainer.evaluate(X, y)
        assert np.isfinite(result["loss"])
        assert 0 <= result["accuracy"] <= 1

    def test_evaluate_with_cross_entropy_loss(self):
        model = CAAAModel(input_dim=36)
        trainer = CAAATrainer(model)
        X = np.random.default_rng(42).normal(size=(20, 36)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        trainer.train(X, y, epochs=5, batch_size=8)
        result = trainer.evaluate(X, y)
        assert np.isfinite(result["loss"])
        assert 0 <= result["accuracy"] <= 1


# ── 12. Statistical features (numpy optimization) ──────────────────


class TestStatisticalFeaturesNumpy:
    """_statistical_features should use numpy instead of pd.concat."""

    def test_no_pd_concat_in_statistical_features(self):
        source = inspect.getsource(FeatureExtractor._statistical_features)
        # Strip comments to only check executable code
        code_lines = [
            line for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "pd.concat" not in code_only
        assert "np.concatenate" in source

    def test_statistical_features_all_finite(self):
        from src.features.feature_schema import STATISTICAL_RANGE
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=5, n_load=5, seed=42, include_hard=False,
        )
        ext = FeatureExtractor(seed=42)
        start, end = STATISTICAL_RANGE
        for case in fault_cases + load_cases:
            feats = ext.extract(case)
            stat_feats = feats[start:end]
            assert np.all(np.isfinite(stat_feats))
