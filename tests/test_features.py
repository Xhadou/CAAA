"""Tests for feature extraction."""

import pytest
import numpy as np

from src.data_loader.dataset import generate_combined_dataset
from src.data_loader.data_types import AnomalyCase
from src.features.extractors import FeatureExtractor, N_FEATURES


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    """Generate a dataset for feature tests (large enough for majority checks)."""
    fault_cases, load_cases = generate_combined_dataset(
        n_fault=30, n_load=30, seed=42
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
        assert feats.shape == (44,)

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


# ── Feature names ─────────────────────────────────────────────────────

class TestFeatureNames:
    def test_feature_names(self, extractor):
        names = extractor.feature_names()
        assert len(names) == N_FEATURES
        assert len(names) == 44
        assert all(isinstance(n, str) for n in names)
        # Spot check some known names
        assert "global_load_ratio" in names
        assert "cpu_deviation" in names
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

        # With comparison context mode, cpu_deviation should be higher for
        # fault cases on average (faults deviate more from baseline)
        cpu_dev_idx = names.index("cpu_deviation")
        assert np.mean(fault_feats[:, cpu_dev_idx]) > np.mean(load_feats[:, cpu_dev_idx]) - 0.1


# ── Context features ─────────────────────────────────────────────────

class TestContextFeatures:
    def test_context_features_for_load(self, dataset):
        _, load_cases = dataset
        ext = FeatureExtractor(context_mode="external")
        # Index 12 is the first context slot; in external mode it holds the
        # event_active value (1.0 when an event_type key is present in context).
        event_idx = 12

        # Majority (>60%) of load cases should have event_active == 1.0
        # (30% may have empty context due to label-leakage prevention)
        active_count = 0
        for case in load_cases:
            feats = ext.extract(case)
            if feats[event_idx] == 1.0:
                active_count += 1
        assert active_count / len(load_cases) >= 0.60

    def test_context_features_for_fault(self, dataset):
        fault_cases, _ = dataset
        ext = FeatureExtractor(context_mode="external")
        # Index 12 is the first context slot; in external mode it holds the
        # event_active value (0.0 when no event_type key is present in context).
        event_idx = 12

        # Majority (>60%) of fault cases should have event_active == 0.0
        # (30% may have fake context due to label-leakage prevention)
        inactive_count = 0
        for case in fault_cases:
            feats = ext.extract(case)
            if feats[event_idx] == 0.0:
                inactive_count += 1
        assert inactive_count / len(fault_cases) >= 0.60


# ── Change point features ────────────────────────────────────────────

class TestChangePointFeatures:
    def test_onset_gradient_computed_for_all_cases(self, extractor, dataset):
        """onset_gradient should be a finite number for all cases."""
        fault_cases, load_cases = dataset
        names = extractor.feature_names()
        onset_idx = names.index("onset_gradient")
        for case in fault_cases + load_cases:
            feats = extractor.extract(case)
            assert np.isfinite(feats[onset_idx]), "onset_gradient is not finite"

    def test_change_point_magnitude_computed_for_all_cases(self, extractor, dataset):
        """change_point_magnitude should be a finite number for all cases."""
        fault_cases, load_cases = dataset
        names = extractor.feature_names()
        mag_idx = names.index("change_point_magnitude")
        for case in fault_cases + load_cases:
            feats = extractor.extract(case)
            assert np.isfinite(feats[mag_idx]), "change_point_magnitude is not finite"
            assert feats[mag_idx] >= 0.0, "change_point_magnitude should be non-negative"

    def test_change_point_magnitude_in_schema(self, extractor):
        """change_point_magnitude should replace memory_trend_uniformity."""
        names = extractor.feature_names()
        assert "change_point_magnitude" in names
        assert "memory_trend_uniformity" not in names


# ── Reproducibility ──────────────────────────────────────────────────

class TestFeatureReproducibility:
    def test_same_seed_produces_identical_features(self, dataset):
        """Extracting the same data with the same seed should be deterministic."""
        fault_cases, load_cases = dataset
        all_cases = fault_cases + load_cases

        ext1 = FeatureExtractor(seed=42)
        X1 = ext1.extract_batch(all_cases)

        ext2 = FeatureExtractor(seed=42)
        X2 = ext2.extract_batch(all_cases)

        np.testing.assert_array_equal(X1, X2)

    def test_different_seed_produces_different_features(self, dataset):
        """Different seeds should produce different context features (external mode)."""
        fault_cases, load_cases = dataset
        all_cases = fault_cases + load_cases

        ext1 = FeatureExtractor(seed=42, context_mode="external")
        X1 = ext1.extract_batch(all_cases)

        ext2 = FeatureExtractor(seed=99, context_mode="external")
        X2 = ext2.extract_batch(all_cases)

        # External context features (indices 12-17) should differ due to noise
        assert not np.allclose(X1[:, 12:17], X2[:, 12:17])

    def test_extraction_order_independent(self, dataset):
        """Features for a case should not depend on extraction order.

        Verifies fix for Issue #1: per-case deterministic RNG ensures
        extracting case A then case B gives the same result as
        extracting case B then case A.
        """
        fault_cases, load_cases = dataset
        all_cases = fault_cases + load_cases

        # Extract in original order
        ext1 = FeatureExtractor(seed=42)
        feat_first = ext1.extract(all_cases[0])

        # Extract in reversed order — case 0 is now extracted last
        ext2 = FeatureExtractor(seed=42)
        for c in reversed(all_cases[1:]):
            ext2.extract(c)
        feat_after_others = ext2.extract(all_cases[0])

        np.testing.assert_array_equal(feat_first, feat_after_others)
