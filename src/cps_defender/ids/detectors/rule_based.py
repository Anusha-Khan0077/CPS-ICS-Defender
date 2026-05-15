"""
Rule-based IDS detector.

Design rationale: rules fire with high precision on known attack signatures
even before ML models are trained. They also serve as interpretable
ground truth for operators. Weighted alongside ML/statistical detectors
in the ensemble.

Each rule returns a (confidence, attack_type) pair or None.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from cps_defender.core.models import Alert, AttackType, FlowRecord, Protocol, Severity

logger = logging.getLogger(__name__)

RuleResult = Optional[Tuple[float, str]]   # (confidence 0-1, attack_type)
RuleFn = Callable[[FlowRecord], RuleResult]


# ── Individual rules ──────────────────────────────────────────────────────────

def rule_broadcast_write(flow: FlowRecord) -> RuleResult:
    """Broadcast DNP3 writes are almost always attacks (replay/injection)."""
    if flow.is_broadcast and flow.function_code in (3, 4, 65, 66, 67, 81):
        return 0.90, AttackType.CMD_INJECTION
    return None


def rule_unknown_function_code(flow: FlowRecord) -> RuleResult:
    """DNP3 FCs above 131 are reserved; their presence indicates probe/fuzzing."""
    if flow.protocol == Protocol.DNP3 and flow.function_code > 131:
        return 0.80, AttackType.SCAN
    return None


def rule_high_rate_scan(flow: FlowRecord) -> RuleResult:
    """
    Very short flows (<0.5 s) with many packets = port scan or device enumeration.
    Legitimate DNP3 polling is periodic and spaced at ≥ 1 s intervals.
    """
    if flow.flow_duration < 0.5 and flow.pkt_count > 20:
        return 0.75, AttackType.SCAN
    return None


def rule_config_change_fc(flow: FlowRecord) -> RuleResult:
    """Write/Freeze FCs (14,15,22-25) on unauthenticated sessions = injection risk."""
    CONFIG_FCS = {14, 15, 22, 23, 24, 25}
    if flow.function_code in CONFIG_FCS and flow.protocol == Protocol.DNP3:
        return 0.85, AttackType.CMD_INJECTION
    return None


def rule_timing_attack(flow: FlowRecord) -> RuleResult:
    """
    FC 19 (Record Current Time) outside scheduled sync windows.
    Attackers use it to desynchronise ICS clocks → trigger events.
    """
    if flow.function_code == 19 and flow.pkt_count > 1:
        return 0.70, AttackType.REPLAY
    return None


def rule_flooding(flow: FlowRecord) -> RuleResult:
    """Packet rate far above normal polling cadence = flooding."""
    if flow.burst_count > 500:
        return 0.85, AttackType.FLOODING
    if flow.pkt_count > 0 and flow.flow_duration > 0:
        pps = flow.pkt_count / flow.flow_duration
        if pps > 1000:
            return 0.80, AttackType.FLOODING
    return None


def rule_unusual_port(flow: FlowRecord) -> RuleResult:
    """Traffic to DNP3 master/outstation ports from unexpected source ports."""
    DNP3_PORTS = {20000, 19999}
    if flow.dst_port in DNP3_PORTS and flow.src_port < 1024 and flow.src_port not in {20000, 19999}:
        return 0.60, AttackType.MITM
    return None


def rule_error_storm(flow: FlowRecord) -> RuleResult:
    """High error rate = device malfunction or spoofed error injections."""
    if flow.error_rate > 0.5 and flow.pkt_count > 10:
        return 0.65, AttackType.CMD_INJECTION
    return None


def rule_duplicate_unsolicited(flow: FlowRecord) -> RuleResult:
    """FC 130 (unsolicited response) in bulk = replay attack."""
    if flow.function_code == 130 and flow.pkt_count > 5 and flow.req_resp_ratio < 0.1:
        return 0.80, AttackType.REPLAY
    return None


# ── Registry ──────────────────────────────────────────────────────────────────

_ALL_RULES: List[Tuple[str, RuleFn]] = [
    ("broadcast_write",          rule_broadcast_write),
    ("unknown_function_code",    rule_unknown_function_code),
    ("high_rate_scan",           rule_high_rate_scan),
    ("config_change_fc",         rule_config_change_fc),
    ("timing_attack",            rule_timing_attack),
    ("flooding",                 rule_flooding),
    ("unusual_port",             rule_unusual_port),
    ("error_storm",              rule_error_storm),
    ("duplicate_unsolicited",    rule_duplicate_unsolicited),
]


# ── Detector class ────────────────────────────────────────────────────────────

class RuleBasedDetector:
    """
    Evaluates all rules against a flow.
    Returns the max-confidence match (or None if clean).
    """

    def __init__(self, rules: Optional[List[Tuple[str, RuleFn]]] = None):
        self.rules = rules if rules is not None else _ALL_RULES
        self._stats: Dict[str, int] = {name: 0 for name, _ in self.rules}

    def predict(self, flow: FlowRecord) -> Optional[Alert]:
        """
        Run all rules.  Returns an Alert for the highest-confidence match,
        or None if no rule fires.
        """
        best_conf = 0.0
        best_attack = AttackType.NORMAL
        best_rule = ""

        for name, fn in self.rules:
            try:
                result = fn(flow)
            except Exception:
                logger.exception("Rule %s raised on flow %s", name, flow.uid)
                continue

            if result is not None:
                conf, attack = result
                if conf > best_conf:
                    best_conf = conf
                    best_attack = attack
                    best_rule = name
                self._stats[name] += 1

        if best_conf == 0.0:
            return None

        sev = (
            Severity.CRITICAL if best_conf >= 0.90 else
            Severity.HIGH     if best_conf >= 0.75 else
            Severity.MEDIUM   if best_conf >= 0.60 else
            Severity.LOW
        )

        return Alert(
            uid=flow.uid,
            timestamp=flow.timestamp,
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            attack_type=best_attack,
            severity=sev,
            confidence=best_conf,
            score=best_conf,
            detector=f"rule:{best_rule}",
        )

    def predict_score(self, flow: FlowRecord) -> float:
        """Return 0-1 risk score (for ensemble weighting)."""
        alert = self.predict(flow)
        return alert.score if alert else 0.0

    def get_rule_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = {name: 0 for name, _ in self.rules}
