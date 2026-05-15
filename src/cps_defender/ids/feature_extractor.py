"""
Feature extraction layer.

Accepts either:
  • FlowRecord objects (from the testbed simulator), or
  • Zeek conn.log / dnp3.log JSON lines (from real deployments).

Design: all feature engineering lives here so detectors stay stateless
and easy to swap. A StandardScaler is fitted once during training and
reused at inference (saved alongside the ML model).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from cps_defender.core.models import (
    FEATURE_NAMES,
    N_FEATURES,
    FlowRecord,
    Protocol,
)

logger = logging.getLogger(__name__)


# ── DNP3 function code taxonomy ───────────────────────────────────────────────
# Source: IEEE Std 1815-2012 (DNP3)
_FC_SAFE  = frozenset(range(0, 34))       # read/response/unsolicited — normal ops
_FC_CTRL  = frozenset([3, 4, 65, 66, 67]) # operate/direct-operate — requires auth
_FC_DANGEROUS = frozenset([
    19,   # Record Current Time (timing attack vector)
    129,  # Response (spoofable)
    130,  # Unsolicited Response
    131,  # Authenticate Request (probe)
    132,  # Authenticate Error
])
_FC_CONFIG = frozenset([14, 15, 22, 23, 24, 25])  # Write/Freeze — config change


def classify_function_code(fc: int) -> str:
    if fc in _FC_DANGEROUS:
        return "dangerous"
    if fc in _FC_CONFIG:
        return "config"
    if fc in _FC_CTRL:
        return "control"
    if fc in _FC_SAFE:
        return "safe"
    return "unknown"


# ── Zeek log parsers ──────────────────────────────────────────────────────────

def _parse_zeek_conn_line(line: str) -> Optional[Dict]:
    """Parse a single Zeek conn.log JSON line."""
    try:
        d = json.loads(line.strip())
        return d
    except (json.JSONDecodeError, ValueError):
        return None


def zeek_logs_to_flows(conn_log_path: str, dnp3_log_path: Optional[str] = None) -> List[FlowRecord]:
    """
    Load Zeek logs and produce FlowRecord objects.
    Joins conn.log and dnp3.log on uid when both are provided.
    """
    conn_records: Dict[str, Dict] = {}
    path = Path(conn_log_path)
    if not path.exists():
        raise FileNotFoundError(f"conn.log not found: {conn_log_path}")

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = _parse_zeek_conn_line(line)
            if rec and "uid" in rec:
                conn_records[rec["uid"]] = rec

    dnp3_records: Dict[str, Dict] = {}
    if dnp3_log_path:
        p2 = Path(dnp3_log_path)
        if p2.exists():
            with p2.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    rec = _parse_zeek_conn_line(line)
                    if rec and "uid" in rec:
                        dnp3_records[rec["uid"]] = rec

    flows: List[FlowRecord] = []
    for uid, c in conn_records.items():
        d = dnp3_records.get(uid, {})
        try:
            flow = FlowRecord(
                uid=uid,
                timestamp=float(c.get("ts", 0)),
                src_ip=c.get("id.orig_h", "0.0.0.0"),
                dst_ip=c.get("id.resp_h", "0.0.0.0"),
                src_port=int(c.get("id.orig_p", 0)),
                dst_port=int(c.get("id.resp_p", 0)),
                protocol=Protocol.DNP3 if int(c.get("id.resp_p", 0)) in (20000, 19999) else Protocol.UNKNOWN,
                flow_duration=float(c.get("duration", 0)),
                pkt_count=int(c.get("orig_pkts", 0)) + int(c.get("resp_pkts", 0)),
                byte_count=int(c.get("orig_bytes", 0)) + int(c.get("resp_bytes", 0)),
                function_code=int(d.get("fc_request", 0)) if d else 0,
                req_resp_ratio=_safe_ratio(int(c.get("orig_pkts", 0)),
                                           int(c.get("orig_pkts", 0)) + int(c.get("resp_pkts", 0))),
                is_broadcast=1 if c.get("id.resp_h", "").endswith(".255") else 0,
            )
            flow.bytes_per_pkt = _safe_ratio(flow.byte_count, flow.pkt_count)
            flows.append(flow)
        except (ValueError, KeyError) as exc:
            logger.debug("Skipping malformed record uid=%s: %s", uid, exc)

    logger.info("Parsed %d flows from Zeek logs", len(flows))
    return flows


def _safe_ratio(num: float, den: float, fallback: float = 0.0) -> float:
    return num / den if den > 0 else fallback


# ── Feature Extractor ────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Converts FlowRecord objects → normalised numpy arrays.

    Fit on training data, then transform at inference.
    Scaler is saved/loaded alongside the ML model.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    # ── Fit / transform ───────────────────────────────────────────────────────

    def fit(self, flows: List[FlowRecord]) -> "FeatureExtractor":
        X = self._raw_matrix(flows)
        self.scaler.fit(X)
        self._fitted = True
        logger.info("Scaler fitted on %d flows, %d features", len(flows), N_FEATURES)
        return self

    def transform(self, flows: List[FlowRecord]) -> np.ndarray:
        X = self._raw_matrix(flows)
        if self._fitted:
            X = self.scaler.transform(X)
        return X

    def fit_transform(self, flows: List[FlowRecord]) -> np.ndarray:
        return self.fit(flows).transform(flows)

    def transform_one(self, flow: FlowRecord) -> np.ndarray:
        return self.transform([flow])[0]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _raw_matrix(self, flows: List[FlowRecord]) -> np.ndarray:
        if not flows:
            return np.zeros((0, N_FEATURES), dtype=np.float32)
        rows = [self._enrich(f).to_feature_vector() for f in flows]
        return np.vstack(rows).astype(np.float32)

    @staticmethod
    def _enrich(flow: FlowRecord) -> FlowRecord:
        """Compute derived fields that require cross-field logic."""
        if flow.pkt_count > 0 and flow.bytes_per_pkt == 0:
            flow.bytes_per_pkt = flow.byte_count / flow.pkt_count
        return flow

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "fitted": self._fitted}, path)
        logger.info("FeatureExtractor saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        obj = cls()
        data = joblib.load(path)
        obj.scaler = data["scaler"]
        obj._fitted = data["fitted"]
        logger.info("FeatureExtractor loaded ← %s", path)
        return obj

    def get_feature_names(self) -> List[str]:
        return list(FEATURE_NAMES)
