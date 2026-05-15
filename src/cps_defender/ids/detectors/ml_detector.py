"""
ML-based IDS detector — Random Forest classifier.

Design decisions:
  • Random Forest over SVM/NN: handles mixed feature types without scaling,
    naturally outputs class probabilities, is robust to missing values,
    and trains in seconds on thousands of samples (no GPU needed).
  • Calibration: sklearn's CalibratedClassifierCV wraps the RF for well-calibrated
    probabilities — important because the ensemble weights depend on confidence.
  • Feature importance exposed for interpretability / operator dashboards.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import joblib

from cps_defender.core.models import (
    FEATURE_NAMES,
    N_FEATURES,
    Alert,
    AttackType,
    FlowRecord,
    Severity,
)
from cps_defender.ids.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# Label index → attack type string
LABEL_MAP: Dict[int, str] = {
    0: AttackType.NORMAL,
    1: AttackType.SCAN,
    2: AttackType.REPLAY,
    3: AttackType.CMD_INJECTION,
    4: AttackType.FLOODING,
    5: AttackType.MITM,
}
INV_LABEL_MAP: Dict[str, int] = {v: k for k, v in LABEL_MAP.items()}
N_CLASSES = len(LABEL_MAP)


class MLDetector:
    """
    Random Forest–based multi-class classifier.

    Workflow:
        detector = MLDetector()
        detector.train(train_flows, labels)
        alert = detector.predict(flow)
        detector.save("data/models/detector.joblib")
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self._rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
            min_samples_leaf=2,
        )
        # CalibratedClassifierCV gives reliable probability estimates
        self._model: Optional[CalibratedClassifierCV] = None
        self._extractor: Optional[FeatureExtractor] = None
        self._classes: List[int] = list(LABEL_MAP.keys())
        self._is_trained = False
        self.feature_importances_: Optional[np.ndarray] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        flows: List[FlowRecord],
        labels: Optional[List[str]] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> Dict:
        """
        Train on a list of FlowRecords.  Labels may come from:
          • the `labels` argument (list of attack type strings), or
          • flow.label on each record (set by the simulator).

        Returns a dict of training metrics.
        """
        if labels is None:
            labels = [f.label for f in flows]
        if len(flows) != len(labels):
            raise ValueError(f"flows/labels length mismatch: {len(flows)} vs {len(labels)}")
        if len(flows) < 10:
            raise ValueError("Need at least 10 training samples")

        # Convert string labels to ints
        y = np.array([INV_LABEL_MAP.get(l, 0) for l in labels], dtype=int)

        # Feature extraction + normalisation
        self._extractor = extractor or FeatureExtractor()
        if extractor is None:
            X = self._extractor.fit_transform(flows)
        else:
            X = self._extractor.transform(flows)

        # Calibrated cross-validation (cv=3 is fast, cv=5 is more accurate)
        n_splits = min(3, self._min_samples_per_class(y))
        self._model = CalibratedClassifierCV(self._rf, cv=n_splits, method="isotonic")
        self._model.fit(X, y)

        # Feature importance from the inner RF
        inner_rf = self._model.calibrated_classifiers_[0].estimator
        self.feature_importances_ = inner_rf.feature_importances_

        self._is_trained = True
        metrics = self._eval_metrics(X, y)
        logger.info(
            "MLDetector trained: %d samples, %d classes, OOB-like accuracy=%.3f",
            len(flows), len(np.unique(y)), metrics["accuracy"],
        )
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, flow: FlowRecord) -> Optional[Alert]:
        if not self._is_trained or self._model is None:
            return None

        X = self._extractor.transform_one(flow).reshape(1, -1)
        proba = self._model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])
        attack = LABEL_MAP[pred_class]

        if attack == AttackType.NORMAL:
            return None
        if confidence < 0.35:
            return None

        sev = (
            Severity.CRITICAL if confidence >= 0.90 else
            Severity.HIGH     if confidence >= 0.75 else
            Severity.MEDIUM   if confidence >= 0.55 else
            Severity.LOW
        )

        return Alert(
            uid=flow.uid,
            timestamp=flow.timestamp,
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            attack_type=attack,
            severity=sev,
            confidence=confidence,
            score=confidence,
            detector="ml:random_forest",
            raw_scores={LABEL_MAP[i]: float(p) for i, p in enumerate(proba)},
        )

    def predict_score(self, flow: FlowRecord) -> float:
        if not self._is_trained or self._model is None:
            return 0.0
        X = self._extractor.transform_one(flow).reshape(1, -1)
        proba = self._model.predict_proba(X)[0]
        # "attack probability" = 1 - P(normal)
        normal_idx = self._classes.index(0) if 0 in self._classes else 0
        return float(1.0 - proba[normal_idx])

    def predict_batch(self, flows: List[FlowRecord]) -> List[Optional[Alert]]:
        return [self.predict(f) for f in flows]

    # ── Feature importance ────────────────────────────────────────────────────

    def get_feature_importance(self) -> Dict[str, float]:
        if self.feature_importances_ is None:
            return {}
        return {
            name: float(imp)
            for name, imp in zip(FEATURE_NAMES, self.feature_importances_)
        }

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        fi = self.get_feature_importance()
        return sorted(fi.items(), key=lambda x: -x[1])[:n]

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, flows: List[FlowRecord], labels: List[str]) -> Dict:
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        y = np.array([INV_LABEL_MAP.get(l, 0) for l in labels])
        X = self._extractor.transform(flows)
        return self._eval_metrics(X, y)

    def _eval_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self._model.predict(X)
        accuracy = float(np.mean(preds == y))
        labels_present = list(np.unique(np.concatenate([y, preds])))
        target_names = [LABEL_MAP.get(i, str(i)) for i in labels_present]
        report = classification_report(
            y, preds, labels=labels_present, target_names=target_names, zero_division=0, output_dict=True
        )
        return {
            "accuracy": accuracy,
            "report": report,
            "n_samples": len(y),
        }

    @staticmethod
    def _min_samples_per_class(y: np.ndarray) -> int:
        _, counts = np.unique(y, return_counts=True)
        return max(1, int(counts.min()))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "extractor": self._extractor,
            "classes": self._classes,
            "feature_importances": self.feature_importances_,
        }, path)
        logger.info("MLDetector saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "MLDetector":
        data = joblib.load(path)
        obj = cls()
        obj._model = data["model"]
        obj._extractor = data["extractor"]
        obj._classes = data["classes"]
        obj.feature_importances_ = data.get("feature_importances")
        obj._is_trained = True
        logger.info("MLDetector loaded ← %s", path)
        return obj

    @property
    def is_trained(self) -> bool:
        return self._is_trained
