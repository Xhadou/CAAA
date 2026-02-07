"""Tests for feature extraction."""

import pytest
import numpy as np
import numpy.testing as npt

from src.data_loader.dataset import generate_combined_dataset
from src.data_loader.data_types import AnomalyCase
from src.features.extractors import FeatureExtractor, N_FEATURES


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    """Generate a small dataset for feature tests."""
    fault_cases, load_cases = generate_combined_dataset(
        n_fault=5, n_load=5, seed=42
    )
    return fault_cases, load_cases


@pytest.fixture
def extractor():
    return FeatureExtractor()


# ── Single extraction ─────────────────────────────────────────────────

class TestFeatureExtractorSingle:
    def test_feature_extractor_single(self, extractor, dataset):
        fault_cases, _ = dataset
        feats = extractor.extract(fault_cases[0])
        assert feats.shape == (N_FEATURES,)
        assert feats.shape == (36,)

    def test_features_are_finite(self, extractor, dataset):
        fault_cases, load_cases = dataset
        for case in fault_cases + load_cases:
            feats = extractor.extract(case)
            assert np.all(np.isfinite(feats)), "Feature vector contains NaN or Inf"


# ── Batch extraction ──────────────────────────────────────────────────

class TestFeatureExtractorBatch:
    def test_feature_extractor_batch(self, extractor, dataset):
        fault_cases, load_cases = dataset
        all_cases = fault_cases + load_cases
        feats = extractor.extract_batch(all_cases)
        assert feats.shape == (len(all_cases), N_FEATURES)
        assert feats.shape == (10, 36)


# ── Feature names ─────────────────────────────────────────────────────

class TestFeatureNames:
    def test_feature_names(self, extractor):
        names = extractor.feature_names()
        assert len(names) == N_FEATURES
        assert len(names) == 36
        assert all(isinstance(n, str) for n in names)
        # Spot check some known names
        assert "global_load_ratio" in names
        assert "event_active" in names
        assert "max_error_rate" in names
        assert "n_services" in names


# ── Fault vs Load feature differences ────────────────────────────────

class TestFaultVsLoad:
    def test_fault_vs_load_features_differ(self, extractor, dataset):
        fault_cases, load_cases = dataset
        fault_feats = extractor.extract_batch(fault_cases)
        load_feats = extractor.extract_batch(load_cases)
        names = extractor.feature_names()

        # error_rate_delta should be higher for fault cases on average
        err_idx = names.index("error_rate_delta")
        assert np.mean(fault_feats[:, err_idx]) > np.mean(load_feats[:, err_idx])

        # event_active should differ: load=1.0, fault=0.0
        event_idx = names.index("event_active")
        npt.assert_array_equal(load_feats[:, event_idx], 1.0)
        npt.assert_array_equal(fault_feats[:, event_idx], 0.0)


# ── Context features ─────────────────────────────────────────────────

class TestContextFeatures:
    def test_context_features_for_load(self, extractor, dataset):
        _, load_cases = dataset
        names = extractor.feature_names()
        event_idx = names.index("event_active")
        conf_idx = names.index("context_confidence")

        for case in load_cases:
            feats = extractor.extract(case)
            assert feats[event_idx] == 1.0
            assert feats[conf_idx] > 0

    def test_context_features_for_fault(self, extractor, dataset):
        fault_cases, _ = dataset
        names = extractor.feature_names()
        event_idx = names.index("event_active")

        for case in fault_cases:
            feats = extractor.extract(case)
            assert feats[event_idx] == 0.0
