"""
Statistical anomaly detector.

Two complementary methods:
  1. Rolling Z-score — per-feature drift detection (low latency, interpretable).
  2. Isolation Forest — multivariate anomaly (handles correlated features).

Design: IsolationForest chosen over One-Class SVM because it scales O(n log n),
handles high-dimensional data well, and is available in scikit-learn (no extra dep).
Rolling stats are computed in O(1) space using Welford's online algorithm.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

from cps_defender.core.models import (
    FEATURE_NAMES,
    N_FEATURES,
    Alert,
    AttackType,
    FlowRecord,
    Severity,
)

logger = logging.getLogger(__name__)


# ── Welford online statistics ─────────────────────────────────────────────────

class WelfordStats:
    """Incremental mean / variance (numerically stable, O(1) space)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def zscore(self, x: float) -> float:
        return (x - self.mean) / (self.std + 1e-9)


# ── Statistical Detector ──────────────────────────────────────────────────────

class StatisticalDetector:
    """
    Per-feature rolling Z-score + Isolation Forest ensemble.

    Usage:
        det = StatisticalDetector(window=200, zscore_thresh=3.5, if_contamination=0.05)
        det.fit(normal_flows)          # fit IF + warm-up rolling stats
        alert = det.predict(new_flow)  # returns Alert or None
    """

    def __init__(
        self,
        window: int = 200,
        zscore_thresh: float = 3.5,
        if_contamination: float = 0.05,
        if_n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.window = window
        self.zscore_thresh = zscore_thresh
        self.if_contamination = if_contamination

        # Rolling stats per feature
        self._welford: List[WelfordStats] = [WelfordStats() for _ in range(N_FEATURES)]
        self._buffer: deque = deque(maxlen=window)

        # Isolation Forest
        self._iforest = IsolationForest(
            n_estimators=if_n_estimators,
            contamination=if_contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._iforest_fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, flows: List[FlowRecord]) -> "StatisticalDetector":
        if not flows:
            logger.warning("StatisticalDetector.fit called with 0 flows")
            return self

        X = np.vstack([f.to_feature_vector() for f in flows]).astype(np.float32)

        # Warm up rolling stats on normal (training) data
        for vec in X:
            self._update_rolling(vec)

        # Fit Isolation Forest
        self._iforest.fit(X)
        self._iforest_fitted = True
        logger.info(
            "StatisticalDetector fitted: %d samples, IF contamination=%.2f",
            len(flows), self.if_contamination,
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, flow: FlowRecord) -> Optional[Alert]:
        vec = flow.to_feature_vector()
        score, anom_features = self._zscore_score(vec)
        if_score = self._iforest_score(vec)

        # Update rolling state (important: after scoring, not before)
        self._update_rolling(vec)

        # Combine: weighted average, higher weight on IF when fitted
        if self._iforest_fitted:
            combined = 0.4 * score + 0.6 * if_score
        else:
            combined = score

        if combined < 0.40:
            return None

        attack = self._infer_attack(flow, anom_features)
        sev = (
            Severity.HIGH   if combined >= 0.75 else
            Severity.MEDIUM if combined >= 0.55 else
            Severity.LOW
        )

        return Alert(
            uid=flow.uid,
            timestamp=flow.timestamp,
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            attack_type=attack,
            severity=sev,
            confidence=min(combined, 1.0),
            score=combined,
            detector="statistical",
            raw_scores={"zscore": score, "iforest": if_score, "combined": combined},
        )

    def predict_score(self, flow: FlowRecord) -> float:
        """Return 0-1 anomaly score (used by ensemble)."""
        vec = flow.to_feature_vector()
        zscore_s, _ = self._zscore_score(vec)
        if_s = self._iforest_score(vec)
        self._update_rolling(vec)
        return 0.4 * zscore_s + 0.6 * if_s if self._iforest_fitted else zscore_s

    # ── Internals ─────────────────────────────────────────────────────────────

    def _update_rolling(self, vec: np.ndarray) -> None:
        for i, v in enumerate(vec):
            self._welford[i].update(float(v))
        self._buffer.append(vec.copy())

    def _zscore_score(self, vec: np.ndarray) -> Tuple[float, List[str]]:
        if all(w.n < 5 for w in self._welford):
            return 0.0, []

        zscores = np.array([w.zscore(float(v)) for w, v in zip(self._welford, vec)])
        abs_z = np.abs(zscores)
        anom_mask = abs_z > self.zscore_thresh
        anom_features = [FEATURE_NAMES[i] for i in np.where(anom_mask)[0]]

        # Score = fraction of features anomalous, weighted by max z
        if anom_mask.any():
            frac = float(anom_mask.sum()) / N_FEATURES
            max_z_norm = float(np.clip(abs_z.max() / (self.zscore_thresh * 2), 0, 1))
            score = 0.5 * frac + 0.5 * max_z_norm
        else:
            score = 0.0

        return float(score), anom_features

    def _iforest_score(self, vec: np.ndarray) -> float:
        if not self._iforest_fitted:
            return 0.0
        # decision_function returns negative anomaly score; invert and normalise
        raw = float(self._iforest.decision_function(vec.reshape(1, -1))[0])
        # Raw range ~[-0.5, 0.5] depending on contamination; map to [0,1]
        score = float(np.clip(0.5 - raw, 0, 1))
        return score

    @staticmethod
    def _infer_attack(flow: FlowRecord, anom_features: List[str]) -> str:
        if "burst_count" in anom_features:
            return AttackType.FLOODING
        if "function_code" in anom_features or "unique_fc_count" in anom_features:
            return AttackType.CMD_INJECTION
        if "inter_arrival_mean" in anom_features:
            return AttackType.REPLAY
        if "pkt_count" in anom_features and "flow_duration" in anom_features:
            return AttackType.SCAN
        return AttackType.SCAN  # safe fallback for unknowns
