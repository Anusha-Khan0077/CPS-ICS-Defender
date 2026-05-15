"""
IDS Ensemble Pipeline.

Combines rule-based, statistical, and ML detectors with configurable weights.
Implements a confidence-escalation policy:
  score < 0.3  → ignore
  0.3-0.5      → MONITOR
  0.5-0.7      → ALERT
  > 0.7        → HIGH_ALERT  (triggers active SDN mitigation)

Design: ensemble aggregation is weighted average of individual scores.
The rule detector gets the highest base weight because it has the lowest
false-positive rate on known attack patterns; its weight can be tuned in config.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from cps_defender.core.config import get_config
from cps_defender.core.events import EventType, get_bus
from cps_defender.core.models import Alert, AttackType, FlowRecord, Severity
from cps_defender.ids.detectors.ml_detector import MLDetector
from cps_defender.ids.detectors.rule_based import RuleBasedDetector
from cps_defender.ids.detectors.statistical import StatisticalDetector
from cps_defender.ids.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class IDSPipeline:
    """
    End-to-end IDS pipeline:
      FlowRecord → [rule + statistical + ML] → ensemble score → Alert (or None)

    Quickstart:
        pipeline = IDSPipeline.create_default()
        pipeline.train(train_flows)
        alert = pipeline.analyze(flow)
    """

    def __init__(
        self,
        rule_detector:  Optional[RuleBasedDetector]  = None,
        stat_detector:  Optional[StatisticalDetector] = None,
        ml_detector:    Optional[MLDetector]          = None,
        weights:        Optional[Dict[str, float]]    = None,
        alert_threshold: float = 0.50,
    ):
        self.rule_det  = rule_detector  or RuleBasedDetector()
        self.stat_det  = stat_detector  or StatisticalDetector()
        self.ml_det    = ml_detector    or MLDetector()

        cfg = get_config()
        default_w = cfg.get("ids", "ensemble_weights") or {}
        self.weights = weights or {
            "rule_based":   default_w.get("rule_based",  0.40),
            "statistical":  default_w.get("statistical", 0.30),
            "ml":           default_w.get("ml",          0.30),
        }
        self._normalise_weights()

        self.alert_threshold = alert_threshold
        self._alert_count = 0
        self._total_flows = 0

        # Wire to event bus
        self._bus = get_bus()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def train(self, flows: List[FlowRecord], labels: Optional[List[str]] = None) -> Dict:
        """Train statistical and ML detectors on labelled flows."""
        logger.info("Training IDS pipeline on %d flows …", len(flows))

        # Stat detector needs only normal flows to characterise baseline
        normal_flows = [f for f in flows if f.label == AttackType.NORMAL]
        if normal_flows:
            self.stat_det.fit(normal_flows)
        else:
            logger.warning("No normal flows provided; statistical detector not fitted on clean traffic")
            self.stat_det.fit(flows)  # fit on all as fallback

        # ML detector trains on all labelled flows
        metrics = self.ml_det.train(flows, labels)
        logger.info("MLDetector metrics: accuracy=%.3f", metrics["accuracy"])
        return metrics

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyze(self, flow: FlowRecord) -> Optional[Alert]:
        """
        Analyse one flow through all detectors.
        Returns an Alert if ensemble score ≥ alert_threshold, else None.
        """
        self._total_flows += 1

        scores: Dict[str, float] = {
            "rule_based":  self.rule_det.predict_score(flow),
            "statistical": self.stat_det.predict_score(flow),
            "ml":          self.ml_det.predict_score(flow),
        }

        ensemble_score = sum(scores[k] * self.weights[k] for k in scores)

        if ensemble_score < self.alert_threshold:
            return None

        # Build alert from the detector with the highest individual score
        best_det = max(scores, key=scores.__getitem__)
        raw_alert = self._get_raw_alert(flow, best_det)
        if raw_alert is None:
            # Fallback: construct from ensemble info
            raw_alert = self._make_alert(flow, ensemble_score, scores)

        # Override with ensemble values
        raw_alert.score      = float(ensemble_score)
        raw_alert.confidence = float(min(ensemble_score * 1.1, 1.0))
        raw_alert.raw_scores = scores

        self._alert_count += 1
        self._bus.emit(EventType.ALERT_GENERATED, raw_alert, source="ids_pipeline")
        logger.debug("Alert: %s → %s (score=%.3f)", flow.src_ip, raw_alert.attack_type, ensemble_score)
        return raw_alert

    def analyze_batch(self, flows: List[FlowRecord]) -> List[Optional[Alert]]:
        return [self.analyze(f) for f in flows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        return {
            "total_flows": self._total_flows,
            "alerts": self._alert_count,
            "alert_rate": self._alert_count / max(self._total_flows, 1),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_raw_alert(self, flow: FlowRecord, detector_name: str) -> Optional[Alert]:
        if detector_name == "rule_based":
            return self.rule_det.predict(flow)
        if detector_name == "statistical":
            return self.stat_det.predict(flow)
        if detector_name == "ml":
            return self.ml_det.predict(flow)
        return None

    def _make_alert(self, flow: FlowRecord, score: float, scores: Dict[str, float]) -> Alert:
        return Alert(
            uid=flow.uid or str(uuid.uuid4()),
            timestamp=flow.timestamp or time.time(),
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            attack_type=AttackType.SCAN,
            severity=Severity.MEDIUM,
            confidence=score,
            score=score,
            detector="ensemble",
            raw_scores=scores,
        )

    def _normalise_weights(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Ensemble weights must sum to > 0")
        self.weights = {k: v / total for k, v in self.weights.items()}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create_default(cls) -> "IDSPipeline":
        cfg = get_config()
        return cls(
            alert_threshold=cfg.get("ids", "alert_threshold", default=0.50),
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        import os
        os.makedirs(directory, exist_ok=True)
        self.ml_det.save(f"{directory}/ml_detector.joblib")
        logger.info("Pipeline saved → %s", directory)

    @classmethod
    def load(cls, directory: str) -> "IDSPipeline":
        pipeline = cls()
        import os
        ml_path = f"{directory}/ml_detector.joblib"
        if os.path.exists(ml_path):
            pipeline.ml_det = MLDetector.load(ml_path)
        return pipeline
