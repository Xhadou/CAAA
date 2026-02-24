"""Tests that verify the remaining code review fixes.

These tests cover issues 2.2, 2.4, 2.5, 3.2, and 3.6 from the code review.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from src.data_loader.dataset import generate_combined_dataset
from src.evaluation.metrics import compute_all_metrics
from src.features.extractors import FeatureExtractor, N_FEATURES
from src.features.feature_schema import N_FEATURES as SCHEMA_N_FEATURES
from src.models.classifier import AnomalyClassifier


# ── 2.2: Vectorized cross-service sync ──────────────────────────────

class TestCrossServiceSyncVectorized:
    """Verify that cross_service_sync uses vectorized numpy correlation."""

    def test_vectorized_correlation_in_source(self):
        """The O(n²) pairwise loop should be replaced with np.corrcoef."""
        source = inspect.getsource(FeatureExtractor._workload_features)
        assert "np.corrcoef" in source, "Should use vectorized np.corrcoef"
        assert "for j in range(i + 1, n):" not in source, (
            "Should not have O(n²) nested loop"
        )

    def test_cross_service_sync_value_range(self):
        """cross_service_sync should be in [-1, 1]."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=5, n_load=5, seed=42, include_hard=False,
        )
        ext = FeatureExtractor(seed=42)
        names = ext.feature_names()
        css_idx = names.index("cross_service_sync")

        for case in fault_cases + load_cases:
            feats = ext.extract(case)
            assert -1.0 <= feats[css_idx] <= 1.0, (
                f"cross_service_sync out of range: {feats[css_idx]}"
            )

    def test_cross_service_sync_single_service(self):
        """With a single service, cross_service_sync should be 0."""
        from src.data_loader.data_types import AnomalyCase, ServiceMetrics

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


# ── 2.5: AnomalyClassifier type hint standardization ────────────────

class TestClassifierTypeHints:
    """Verify that AnomalyClassifier uses np.ndarray type hints."""

    def test_fit_accepts_ndarray(self):
        """fit() should accept np.ndarray without errors."""
        X = np.random.default_rng(42).normal(size=(20, 5)).astype(np.float32)
        y = np.array(["FAULT"] * 10 + ["EXPECTED_LOAD"] * 10)
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        assert clf.is_fitted

    def test_fit_still_accepts_dataframe(self):
        """fit() should still accept DataFrames for backward compat."""
        X = pd.DataFrame(
            np.random.default_rng(42).normal(size=(20, 5)),
            columns=[f"f{i}" for i in range(5)],
        )
        y = np.array(["FAULT"] * 10 + ["EXPECTED_LOAD"] * 10)
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        assert clf.is_fitted

    def test_predict_accepts_ndarray(self):
        """predict() should accept np.ndarray."""
        X = np.random.default_rng(42).normal(size=(20, 5)).astype(np.float32)
        y = np.array(["FAULT"] * 10 + ["EXPECTED_LOAD"] * 10)
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        preds = clf.predict(X)
        assert len(preds) == 20

    def test_fit_type_hint_is_ndarray(self):
        """fit() type hint should be np.ndarray, not pd.DataFrame."""
        sig = inspect.signature(AnomalyClassifier.fit)
        x_annotation = sig.parameters["X"].annotation
        assert x_annotation is np.ndarray, (
            f"Expected np.ndarray, got {x_annotation}"
        )


# ── 3.2: __all__ in __init__.py files ──────────────────────────────

class TestInitExports:
    """Verify __all__ exports in previously empty __init__.py files."""

    def test_evaluation_init_has_all(self):
        import src.evaluation
        assert hasattr(src.evaluation, "__all__"), (
            "src/evaluation/__init__.py should define __all__"
        )

    def test_training_init_has_all(self):
        import src.training
        assert hasattr(src.training, "__all__"), (
            "src/training/__init__.py should define __all__"
        )


# ── 3.6: AD pre-filter checks all services ─────────────────────────

class TestADPreFilterAllServices:
    """Verify that anomaly detector pre-filter checks all services."""

    def test_train_script_checks_all_services(self):
        """train.py should check all services, not just services[0]."""
        with open("scripts/train.py") as f:
            content = f.read()
        assert "case.services[0]" not in content, (
            "train.py should not only check first service"
        )
        assert "for svc in case.services" in content, (
            "train.py should iterate over all services"
        )

    def test_main_checks_all_services(self):
        """main.py should check all services, not just services[0]."""
        with open("src/main.py") as f:
            content = f.read()
        assert "case.services[0]" not in content, (
            "main.py should not only check first service"
        )
