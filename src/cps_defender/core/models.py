"""
Shared data models (dataclasses, no Pydantic dep).

FlowRecord — one network flow (from Zeek log or simulator).
Alert      — IDS output.
NetworkState — snapshot fed to the RL agent.

Feature layout (FEATURE_NAMES) is the single source of truth used by
feature extractor, ML detector, and RL environment alike.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np


# ── Protocol / attack enums ──────────────────────────────────────────────────

class Protocol(IntEnum):
    UNKNOWN = 0
    DNP3    = 1
    MODBUS  = 2
    IEC104  = 3
    S7COMM  = 4


class AttackType(str):
    NORMAL           = "normal"
    SCAN             = "scan"
    REPLAY           = "replay"
    CMD_INJECTION    = "command_injection"
    FLOODING         = "flooding"
    MITM             = "mitm"


class Severity(IntEnum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


# ── Feature schema ────────────────────────────────────────────────────────────
#  Every ML component uses this ordered list.  Add features here only.

FEATURE_NAMES: List[str] = [
    "flow_duration",        # seconds
    "pkt_count",
    "byte_count",
    "bytes_per_pkt",
    "src_port",
    "dst_port",
    "protocol_id",          # Protocol enum value
    "function_code",        # DNP3/Modbus FC (0-255); 0 if N/A
    "unique_fc_count",      # distinct FCs in the flow
    "req_resp_ratio",       # fraction of pkts that are requests
    "inter_arrival_mean",   # ms
    "inter_arrival_std",    # ms
    "is_broadcast",         # 0/1
    "direction",            # 0=master→RTU, 1=RTU→master, 2=unknown
    "burst_count",          # pkts in last 1 s
    "error_rate",           # error responses / total
]

N_FEATURES = len(FEATURE_NAMES)


# ── Core data classes ─────────────────────────────────────────────────────────

@dataclass
class FlowRecord:
    """One summarized network flow."""
    uid: str
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: Protocol = Protocol.DNP3

    # Feature fields (mirrors FEATURE_NAMES where possible)
    flow_duration: float = 0.0
    pkt_count: int = 0
    byte_count: int = 0
    bytes_per_pkt: float = 0.0
    function_code: int = 0
    unique_fc_count: int = 1
    req_resp_ratio: float = 0.5
    inter_arrival_mean: float = 0.0
    inter_arrival_std: float = 0.0
    is_broadcast: int = 0
    direction: int = 0
    burst_count: int = 0
    error_rate: float = 0.0

    # Ground-truth label (for simulated data)
    label: str = AttackType.NORMAL
    label_id: int = 0

    def to_feature_vector(self) -> np.ndarray:
        vec = np.array([
            self.flow_duration,
            float(self.pkt_count),
            float(self.byte_count),
            self.bytes_per_pkt,
            float(self.src_port),
            float(self.dst_port),
            float(int(self.protocol)),
            float(self.function_code),
            float(self.unique_fc_count),
            self.req_resp_ratio,
            self.inter_arrival_mean,
            self.inter_arrival_std,
            float(self.is_broadcast),
            float(self.direction),
            float(self.burst_count),
            self.error_rate,
        ], dtype=np.float32)
        assert len(vec) == N_FEATURES, f"Feature length mismatch: {len(vec)} vs {N_FEATURES}"
        return vec

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Alert:
    """IDS alert output."""
    uid: str
    timestamp: float
    src_ip: str
    dst_ip: str
    attack_type: str
    severity: Severity
    confidence: float          # 0-1
    score: float               # composite score (0-1)
    detector: str              # which detector fired
    features: Optional[np.ndarray] = None
    raw_scores: Dict[str, float] = field(default_factory=dict)
    mitigated: bool = False

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.7

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items() if k != "features"}
        d["severity"] = int(self.severity)
        return d


@dataclass
class NetworkState:
    """Snapshot of network health for the RL agent."""
    timestamp: float = field(default_factory=time.time)
    alert_count_1m: int = 0
    avg_severity: float = 0.0
    avg_confidence: float = 0.0
    attack_types: Dict[str, int] = field(default_factory=dict)
    current_action: int = 0     # last mitigation action applied
    latency_ms: float = 0.0
    packet_loss_pct: float = 0.0
    isolated_hosts: int = 0
    qos_score: float = 1.0     # 1.0 = perfect, 0.0 = degraded

    def to_vector(self) -> np.ndarray:
        return np.array([
            float(self.alert_count_1m) / 100.0,  # normalise
            self.avg_severity / 4.0,
            self.avg_confidence,
            float(self.current_action) / 4.0,
            self.latency_ms / 1000.0,
            self.packet_loss_pct / 100.0,
            float(self.isolated_hosts) / 20.0,
            self.qos_score,
        ], dtype=np.float32)

    N_STATE = 8  # dimension of to_vector()
