"""Tests for IDS components: feature extractor, detectors, pipeline."""
import numpy as np
import pytest

from cps_defender.core.models import AttackType, FlowRecord, N_FEATURES
from cps_defender.ids.feature_extractor import FeatureExtractor
from cps_defender.ids.detectors.rule_based import RuleBasedDetector
from cps_defender.ids.detectors.statistical import StatisticalDetector
from cps_defender.ids.detectors.ml_detector import MLDetector
from cps_defender.ids.pipeline import IDSPipeline
from cps_defender.testbed.traffic_sim import TrafficSimulator


# ── Feature extractor ─────────────────────────────────────────────────────────

class TestFeatureExtractor:
    def test_fit_transform_shape(self, small_dataset, fitted_extractor):
        X = fitted_extractor.transform(small_dataset)
        assert X.shape == (len(small_dataset), N_FEATURES)
        assert X.dtype == np.float32

    def test_no_nan_or_inf(self, feature_matrix):
        assert not np.any(np.isnan(feature_matrix))
        assert not np.any(np.isinf(feature_matrix))

    def test_normalisation(self, large_dataset):
        ext = FeatureExtractor()
        X = ext.fit_transform(large_dataset)
        col_means = np.abs(X.mean(axis=0))
        assert col_means.max() < 1.0, "Scaled means should be near 0"

    def test_transform_one(self, fitted_extractor, small_dataset):
        flow = small_dataset[0]
        vec = fitted_extractor.transform_one(flow)
        assert vec.shape == (N_FEATURES,)

    def test_save_load_roundtrip(self, fitted_extractor, small_dataset, tmp_path):
        path = str(tmp_path / "extractor.joblib")
        fitted_extractor.save(path)
        loaded = FeatureExtractor.load(path)
        X1 = fitted_extractor.transform(small_dataset[:10])
        X2 = loaded.transform(small_dataset[:10])
        np.testing.assert_allclose(X1, X2, rtol=1e-5)

    def test_feature_names_length(self, fitted_extractor):
        names = fitted_extractor.get_feature_names()
        assert len(names) == N_FEATURES

    def test_fit_transform_consistency(self, small_dataset):
        ext = FeatureExtractor()
        X1 = ext.fit_transform(small_dataset)
        X2 = ext.transform(small_dataset)
        np.testing.assert_allclose(X1, X2, rtol=1e-6)


# ── Rule-based detector ───────────────────────────────────────────────────────

class TestRuleBasedDetector:
    @pytest.fixture
    def det(self):
        return RuleBasedDetector()

    def test_no_exception_on_normal(self, det, small_dataset):
        normal = [f for f in small_dataset if f.label == AttackType.NORMAL]
        assert len(normal) > 0, "Expected normal flows in dataset"
        for flow in normal[:20]:
            score = det.predict_score(flow)
            assert 0.0 <= score <= 1.0

    def test_low_fp_on_normal(self, det, small_dataset):
        normal = [f for f in small_dataset if f.label == AttackType.NORMAL][:30]
        fp = sum(1 for f in normal if det.predict(f) is not None)
        assert fp / len(normal) < 0.25, "FP rate too high on normal traffic"

    def test_score_range(self, det, small_dataset):
        for flow in small_dataset[:50]:
            score = det.predict_score(flow)
            assert 0.0 <= score <= 1.0

    def test_flood_score_higher_than_normal(self, det, small_dataset):
        normal  = [f for f in small_dataset if f.label == AttackType.NORMAL]
        attacks = [f for f in small_dataset if f.label == AttackType.FLOODING]
        if not normal or not attacks:
            pytest.skip("Need both normal and flooding flows")
        avg_normal  = np.mean([det.predict_score(f) for f in normal[:20]])
        avg_attack  = np.mean([det.predict_score(f) for f in attacks[:20]])
        assert avg_attack >= avg_normal, \
            f"Flooding score ({avg_attack:.3f}) not > normal ({avg_normal:.3f})"

    def test_rule_stats_dict(self, det, small_dataset):
        for f in small_dataset[:50]:
            det.predict(f)
        stats = det.get_rule_stats()
        assert isinstance(stats, dict)
        assert all(isinstance(v, int) for v in stats.values())

    def test_reset_stats(self, det, small_dataset):
        for f in small_dataset[:10]:
            det.predict(f)
        det.reset_stats()
        stats = det.get_rule_stats()
        assert all(v == 0 for v in stats.values())


# ── Statistical detector ──────────────────────────────────────────────────────

class TestStatisticalDetector:
    @pytest.fixture
    def fitted_det(self, large_dataset):
        det = StatisticalDetector()
        det.fit(large_dataset)
        return det

    def test_score_range(self, fitted_det, small_dataset):
        for flow in small_dataset[:30]:
            score = fitted_det.predict_score(flow)
            assert 0.0 <= score <= 1.0

    def test_returns_optional_alert(self, fitted_det, small_dataset):
        for flow in small_dataset[:20]:
            result = fitted_det.predict(flow)
            assert result is None or hasattr(result, "confidence")

    def test_fit_returns_self(self, small_dataset):
        det = StatisticalDetector()
        result = det.fit(small_dataset)
        assert result is det

    def test_attack_scores_elevated(self, fitted_det, small_dataset):
        normal  = [f for f in small_dataset if f.label == AttackType.NORMAL][:20]
        attacks = [f for f in small_dataset
                   if f.label != AttackType.NORMAL][:20]
        if not normal or not attacks:
            pytest.skip("Insufficient data")
        avg_normal = np.mean([fitted_det.predict_score(f) for f in normal])
        avg_attack = np.mean([fitted_det.predict_score(f) for f in attacks])
        # Attacks should on average score higher
        assert avg_attack >= avg_normal - 0.1  # lenient bound


# ── ML detector ───────────────────────────────────────────────────────────────

class TestMLDetector:
    @pytest.fixture
    def trained_det(self, large_dataset):
        det = MLDetector()
        det.train(large_dataset)
        return det

    def test_train_returns_report(self, large_dataset):
        det = MLDetector()
        result = det.train(large_dataset)
        assert isinstance(result, dict)
        assert "accuracy" in result

    def test_predict_returns_optional_alert(self, trained_det, small_dataset):
        for flow in small_dataset[:20]:
            result = trained_det.predict(flow)
            assert result is None or hasattr(result, "confidence")

    def test_score_range(self, trained_det, small_dataset):
        for flow in small_dataset[:30]:
            score = trained_det.predict_score(flow)
            assert 0.0 <= score <= 1.0

    def test_is_trained_flag(self, large_dataset):
        det = MLDetector()
        assert not det.is_trained
        det.train(large_dataset)
        assert det.is_trained

    def test_feature_importance(self, trained_det):
        fi = trained_det.get_feature_importance()
        assert isinstance(fi, dict)
        assert len(fi) > 0
        total = sum(fi.values())
        assert abs(total - 1.0) < 0.01

    def test_top_features(self, trained_det):
        top = trained_det.top_features(n=5)
        assert isinstance(top, list)
        assert len(top) <= 5

    def test_save_load_roundtrip(self, trained_det, small_dataset, tmp_path):
        path = str(tmp_path / "ml_detector.joblib")
        trained_det.save(path)
        loaded = MLDetector.load(path)
        assert loaded.is_trained
        # Predictions should match
        for flow in small_dataset[:10]:
            s1 = trained_det.predict_score(flow)
            s2 = loaded.predict_score(flow)
            assert abs(s1 - s2) < 0.01

    def test_evaluate_returns_metrics(self, trained_det, small_dataset):
        labels = [f.label for f in small_dataset]
        metrics = trained_det.evaluate(small_dataset, labels)
        assert "accuracy" in metrics

    def test_accuracy_above_baseline(self, trained_det, large_dataset):
        labels = [f.label for f in large_dataset]
        metrics = trained_det.evaluate(large_dataset, labels)
        # Should beat random (1/6 ≈ 0.17) significantly
        assert metrics["accuracy"] > 0.50


# ── IDS Pipeline ──────────────────────────────────────────────────────────────

class TestIDSPipeline:
    @pytest.fixture
    def trained_pipeline(self, large_dataset):
        pipe = IDSPipeline()
        pipe.train(large_dataset)
        return pipe

    def test_train_returns_dict(self, large_dataset):
        pipe = IDSPipeline()
        result = pipe.train(large_dataset)
        assert isinstance(result, dict)
        assert "accuracy" in result

    def test_analyze_returns_optional_alert(self, trained_pipeline, small_dataset):
        for flow in small_dataset[:20]:
            result = trained_pipeline.analyze(flow)
            assert result is None or hasattr(result, "confidence")

    def test_analyze_batch(self, trained_pipeline, small_dataset):
        results = trained_pipeline.analyze_batch(small_dataset[:30])
        assert len(results) == 30
        for r in results:
            assert r is None or hasattr(r, "confidence")

    def test_alerts_fired_on_attacks(self, trained_pipeline, small_dataset):
        attacks = [f for f in small_dataset if f.label != AttackType.NORMAL][:20]
        if not attacks:
            pytest.skip("No attack flows in small_dataset")
        alerts = [trained_pipeline.analyze(f) for f in attacks]
        fired = sum(1 for a in alerts if a is not None)
        # At least 50% detection rate on training-distribution attacks
        assert fired / len(attacks) > 0.50, \
            f"Too few alerts on attack flows: {fired}/{len(attacks)}"

    def test_low_fp_on_normal(self, trained_pipeline, small_dataset):
        normal = [f for f in small_dataset if f.label == AttackType.NORMAL][:30]
        if not normal:
            pytest.skip("No normal flows in small_dataset")
        fps = sum(1 for f in normal
                  if trained_pipeline.analyze(f) is not None)
        assert fps / len(normal) < 0.30, "FP rate too high"

    def test_pipeline_stats(self, trained_pipeline, small_dataset):
        for f in small_dataset[:30]:
            trained_pipeline.analyze(f)
        stats = trained_pipeline.stats()
        assert isinstance(stats, dict)

    def test_save_load_roundtrip(self, trained_pipeline, small_dataset, tmp_path):
        path = str(tmp_path / "pipeline_dir")
        trained_pipeline.save(path)
        loaded = IDSPipeline.load(path)
        for flow in small_dataset[:10]:
            a1 = trained_pipeline.analyze(flow)
            a2 = loaded.analyze(flow)
            # Both should agree on alert/no-alert
            assert (a1 is None) == (a2 is None)

    def test_create_default(self):
        pipe = IDSPipeline.create_default()
        assert isinstance(pipe, IDSPipeline)

    def test_custom_weights(self, large_dataset, small_dataset):
        # Rule-only ensemble
        pipe = IDSPipeline(weights={"rule": 1.0, "stat": 0.0, "ml": 0.0})
        pipe.train(large_dataset)
        for flow in small_dataset[:10]:
            result = pipe.analyze(flow)
            assert result is None or hasattr(result, "confidence")
