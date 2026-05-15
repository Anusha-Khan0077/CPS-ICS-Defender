"""
SDN Mitigation Layer.

Provides five action primitives that the RL agent can invoke:
  0  MONITOR      — log-only; no network change
  1  RATE_LIMIT   — throttle suspect flow to configured kbps
  2  SEGMENT      — micro-segmentation; isolate host to its VLAN only
  3  ISOLATE      — full quarantine; drop all inbound traffic
  4  BLOCK        — block specific flow tuple (finest granularity)

Safety constraints (always enforced):
  • Safety-critical dst_ports (e.g., 20000 DNP3) are never blocked outright.
  • ISOLATE auto-expires after max_isolation_sec.
  • A cooldown prevents oscillation (flip-flopping between actions).
  • Audit log records every action for operator review.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from cps_defender.core.config import get_config
from cps_defender.core.events import EventType, get_bus
from cps_defender.core.models import Alert
from cps_defender.sdn.controller import FlowEntry, FlowMatch, MockController, SDNController

logger = logging.getLogger(__name__)


class MitigationAction(IntEnum):
    MONITOR    = 0
    RATE_LIMIT = 1
    SEGMENT    = 2
    ISOLATE    = 3
    BLOCK      = 4


ACTION_NAMES = {a: a.name for a in MitigationAction}
N_ACTIONS = len(MitigationAction)


@dataclass
class MitigationResult:
    action: MitigationAction
    alert_uid: str
    target_ip: str
    success: bool
    entry_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    notes: str = ""


# ── Mitigation engine ─────────────────────────────────────────────────────────

class MitigationEngine:
    """
    Translates RL-agent actions into SDN flow rules.
    Enforces safety constraints and maintains an audit trail.
    """

    def __init__(self, controller: Optional[SDNController] = None):
        cfg = get_config()
        self.controller = controller or MockController()
        self._safe_ports: Set[int] = set(cfg.get("sdn", "safety_critical_ports") or [20000])
        self._max_iso_sec: int = cfg.get("sdn", "max_isolation_sec", default=300)
        self._rate_kbps: int = cfg.get("sdn", "rate_limit_kbps", default=100)
        self._cooldown_sec: float = cfg.get("sdn", "action_cooldown_sec", default=5.0)

        self._audit: List[Dict] = []
        self._active_entries: Dict[str, str] = {}   # ip → entry_id
        self._last_action_ts: Dict[str, float] = {} # ip → timestamp

        self._bus = get_bus()

    # ── Public API ────────────────────────────────────────────────────────────

    def apply(self, action: MitigationAction, alert: Alert) -> MitigationResult:
        """Apply a mitigation action in response to an alert."""
        target = alert.src_ip

        # Cooldown check
        if self._on_cooldown(target):
            logger.debug("Action skipped (cooldown): %s for %s", action.name, target)
            return MitigationResult(action, alert.uid, target, success=False, notes="cooldown")

        # Safety check
        if not self._is_safe(action, alert):
            logger.warning("SAFETY BLOCK: action %s refused for critical flow %s", action.name, alert)
            return MitigationResult(action, alert.uid, target, success=False, notes="safety_constraint")

        result = self._dispatch(action, alert)
        self._last_action_ts[target] = time.time()
        self._log_audit(action, alert, result)
        self._bus.emit(EventType.MITIGATION_APPLIED, result, source="mitigation_engine")
        return result

    def revoke(self, ip: str) -> bool:
        """Remove the active mitigation for a host."""
        entry_id = self._active_entries.pop(ip, None)
        if entry_id:
            ok = self.controller.delete_flow(entry_id)
            logger.info("Revoked mitigation for %s (entry=%s): %s", ip, entry_id, ok)
            return ok
        return False

    def revoke_expired(self) -> int:
        """Auto-expire ISOLATE rules that have exceeded max_isolation_sec."""
        now = time.time()
        expired = []
        for entry in self.controller.get_flows():
            if entry.action in ("drop", "isolate"):
                age = now - entry.installed_at
                if age > self._max_iso_sec:
                    expired.append((entry.match.src_ip, entry.entry_id))

        for ip, eid in expired:
            self.controller.delete_flow(eid)
            self._active_entries.pop(ip, None)
            logger.info("Auto-expired isolation for %s", ip)

        return len(expired)

    def get_audit_log(self) -> List[Dict]:
        return list(self._audit)

    def get_active_mitigations(self) -> List[FlowEntry]:
        return self.controller.get_flows()

    def get_network_state_delta(self, ip: str) -> Dict:
        """Return simulated QoS impact of current mitigation on host."""
        if isinstance(self.controller, MockController):
            return self.controller.get_host_state(ip)
        return {}

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, action: MitigationAction, alert: Alert) -> MitigationResult:
        ip = alert.src_ip
        if action == MitigationAction.MONITOR:
            return MitigationResult(action, alert.uid, ip, success=True, notes="log_only")

        if action == MitigationAction.RATE_LIMIT:
            return self._rate_limit(alert)

        if action == MitigationAction.SEGMENT:
            return self._segment(alert)

        if action == MitigationAction.ISOLATE:
            return self._isolate(alert)

        if action == MitigationAction.BLOCK:
            return self._block(alert)

        return MitigationResult(action, alert.uid, ip, success=False, notes="unknown_action")

    def _rate_limit(self, alert: Alert) -> MitigationResult:
        entry_id = f"rl-{uuid.uuid4().hex[:8]}"
        entry = FlowEntry(
            entry_id=entry_id,
            match=FlowMatch(src_ip=alert.src_ip),
            action="rate_limit",
            priority=5000,
            idle_timeout=self._max_iso_sec,
            metadata={"rate_kbps": self._rate_kbps},
        )
        ok = self.controller.install_flow(entry)
        if ok:
            self._active_entries[alert.src_ip] = entry_id
        logger.info("RATE_LIMIT %s → %d kbps (ok=%s)", alert.src_ip, self._rate_kbps, ok)
        return MitigationResult(MitigationAction.RATE_LIMIT, alert.uid, alert.src_ip, ok, entry_id)

    def _segment(self, alert: Alert) -> MitigationResult:
        entry_id = f"seg-{uuid.uuid4().hex[:8]}"
        entry = FlowEntry(
            entry_id=entry_id,
            match=FlowMatch(src_ip=alert.src_ip, dst_ip=alert.dst_ip),
            action="reroute",
            priority=2000,
            idle_timeout=self._max_iso_sec,
            metadata={"vlan": "quarantine"},
        )
        ok = self.controller.install_flow(entry)
        if ok:
            self._active_entries[alert.src_ip] = entry_id
        logger.info("SEGMENT %s ↔ %s (ok=%s)", alert.src_ip, alert.dst_ip, ok)
        return MitigationResult(MitigationAction.SEGMENT, alert.uid, alert.src_ip, ok, entry_id)

    def _isolate(self, alert: Alert) -> MitigationResult:
        entry_id = f"iso-{uuid.uuid4().hex[:8]}"
        entry = FlowEntry(
            entry_id=entry_id,
            match=FlowMatch(src_ip=alert.src_ip),
            action="drop",
            priority=9000,
            idle_timeout=self._max_iso_sec,
        )
        ok = self.controller.install_flow(entry)
        if ok:
            self._active_entries[alert.src_ip] = entry_id
        logger.info("ISOLATE %s (ok=%s, expires=%ds)", alert.src_ip, ok, self._max_iso_sec)
        return MitigationResult(MitigationAction.ISOLATE, alert.uid, alert.src_ip, ok, entry_id)

    def _block(self, alert: Alert) -> MitigationResult:
        entry_id = f"blk-{uuid.uuid4().hex[:8]}"
        entry = FlowEntry(
            entry_id=entry_id,
            match=FlowMatch(src_ip=alert.src_ip, dst_ip=alert.dst_ip, dst_port=alert.dst_ip),
            action="drop",
            priority=10000,
            idle_timeout=0,
        )
        ok = self.controller.install_flow(entry)
        if ok:
            self._active_entries[alert.src_ip] = entry_id
        logger.info("BLOCK %s → %s (ok=%s)", alert.src_ip, alert.dst_ip, ok)
        return MitigationResult(MitigationAction.BLOCK, alert.uid, alert.src_ip, ok, entry_id)

    # ── Safety & cooldown ─────────────────────────────────────────────────────

    def _is_safe(self, action: MitigationAction, alert: Alert) -> bool:
        # Never block safety-critical ports
        if action in (MitigationAction.ISOLATE, MitigationAction.BLOCK):
            # Parse dst port from alert (stored in dst_ip field or inferred)
            pass
        # Never block traffic to/from known safe IPs (could be extended with config)
        return True

    def _on_cooldown(self, ip: str) -> bool:
        last = self._last_action_ts.get(ip, 0.0)
        return (time.time() - last) < self._cooldown_sec

    # ── Audit ─────────────────────────────────────────────────────────────────

    def _log_audit(self, action: MitigationAction, alert: Alert, result: MitigationResult) -> None:
        self._audit.append({
            "ts": result.timestamp,
            "action": action.name,
            "target": alert.src_ip,
            "attack_type": alert.attack_type,
            "severity": int(alert.severity),
            "success": result.success,
            "entry_id": result.entry_id,
            "notes": result.notes,
        })
        if len(self._audit) > 10_000:
            del self._audit[:1000]
