"""
SDN Controller abstraction layer.

Two backends:
  MockController  — in-memory flow table, works without Mininet/Ryu.
                    Used for unit tests, RL training, and offline demos.
  RyuController   — REST client for a running Ryu controller (ryu-manager).

Design: program to an interface (SDNController ABC) so swapping backends
needs zero changes in mitigation.py or the RL agent.

OpenFlow action priorities (higher = earlier match):
  10000 — block rules (highest)
   5000 — rate-limit / shape
   2000 — reroute / segment
   1000 — default forward
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class FlowMatch:
    """OpenFlow-style match fields."""
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    ip_proto: Optional[int] = None  # 6=TCP, 17=UDP

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class FlowEntry:
    """One installed flow rule."""
    entry_id: str
    match: FlowMatch
    action: str            # forward | drop | rate_limit | reroute
    priority: int = 1000
    idle_timeout: int = 0  # 0 = permanent
    installed_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


# ── Abstract interface ─────────────────────────────────────────────────────────

class SDNController(ABC):
    """Abstract SDN controller — implement this to add new backends."""

    @abstractmethod
    def install_flow(self, entry: FlowEntry) -> bool:
        ...

    @abstractmethod
    def delete_flow(self, entry_id: str) -> bool:
        ...

    @abstractmethod
    def get_flows(self) -> List[FlowEntry]:
        ...

    @abstractmethod
    def get_stats(self, dpid: str = "1") -> Dict:
        ...


# ── Mock controller ────────────────────────────────────────────────────────────

class MockController(SDNController):
    """
    In-memory controller with realistic semantics.
    Tracks flow rules and simulates network state changes.
    """

    def __init__(self):
        self._flows: Dict[str, FlowEntry] = {}
        self._stats: Dict[str, Dict] = {}
        self._install_count = 0
        # Simulated per-host latency / loss (modified by mitigation actions)
        self._host_state: Dict[str, Dict] = {}

    def install_flow(self, entry: FlowEntry) -> bool:
        self._flows[entry.entry_id] = entry
        self._install_count += 1
        # Simulate effect on host metrics
        host = entry.match.src_ip or "unknown"
        state = self._host_state.setdefault(host, {"latency_ms": 5.0, "loss_pct": 0.0, "rate_kbps": -1})
        if entry.action == "drop":
            state["rate_kbps"] = 0
        elif entry.action == "rate_limit":
            limit = entry.metadata.get("rate_kbps", 100)
            state["rate_kbps"] = limit
            state["latency_ms"] += 2.0   # rate limiting adds queuing latency
        elif entry.action == "reroute":
            state["latency_ms"] += 5.0
        logger.debug("Installed flow %s: %s → %s", entry.entry_id, entry.match, entry.action)
        return True

    def delete_flow(self, entry_id: str) -> bool:
        if entry_id not in self._flows:
            return False
        entry = self._flows.pop(entry_id)
        # Restore host state
        host = entry.match.src_ip or "unknown"
        if host in self._host_state:
            self._host_state[host] = {"latency_ms": 5.0, "loss_pct": 0.0, "rate_kbps": -1}
        logger.debug("Deleted flow %s", entry_id)
        return True

    def get_flows(self) -> List[FlowEntry]:
        return list(self._flows.values())

    def get_stats(self, dpid: str = "1") -> Dict:
        installed = len(self._flows)
        # Aggregate simulated metrics
        latencies = [s["latency_ms"] for s in self._host_state.values()]
        losses = [s["loss_pct"] for s in self._host_state.values()]
        return {
            "dpid": dpid,
            "n_flows": installed,
            "total_installs": self._install_count,
            "avg_latency_ms": float(sum(latencies) / len(latencies)) if latencies else 5.0,
            "avg_loss_pct": float(sum(losses) / len(losses)) if losses else 0.0,
            "host_states": dict(self._host_state),
        }

    def get_host_state(self, ip: str) -> Dict:
        return self._host_state.get(ip, {"latency_ms": 5.0, "loss_pct": 0.0, "rate_kbps": -1})


# ── Ryu REST controller ───────────────────────────────────────────────────────

class RyuController(SDNController):
    """
    REST client for a Ryu controller (ryu-manager with ofctl_rest app).
    Endpoint: http://<host>:8080

    Start Ryu: ryu-manager ryu.app.ofctl_rest ryu.app.rest_topology
    """

    def __init__(self, base_url: str = "http://localhost:8080", dpid: str = "1", timeout: int = 5):
        self.base_url = base_url.rstrip("/")
        self.dpid = dpid
        self.timeout = timeout
        self._installed: Dict[str, FlowEntry] = {}  # local registry

    def install_flow(self, entry: FlowEntry) -> bool:
        payload = self._entry_to_ryu(entry)
        try:
            self._post(f"/stats/flowentry/add", payload)
            self._installed[entry.entry_id] = entry
            return True
        except Exception as exc:
            logger.error("Ryu install_flow failed: %s", exc)
            return False

    def delete_flow(self, entry_id: str) -> bool:
        if entry_id not in self._installed:
            logger.warning("Flow %s not in local registry; cannot delete", entry_id)
            return False
        entry = self._installed[entry_id]
        payload = self._entry_to_ryu(entry)
        try:
            self._post(f"/stats/flowentry/delete", payload)
            del self._installed[entry_id]
            return True
        except Exception as exc:
            logger.error("Ryu delete_flow failed: %s", exc)
            return False

    def get_flows(self) -> List[FlowEntry]:
        return list(self._installed.values())

    def get_stats(self, dpid: Optional[str] = None) -> Dict:
        dpid = dpid or self.dpid
        try:
            data = self._get(f"/stats/flow/{dpid}")
            return {"dpid": dpid, "flows": data}
        except Exception as exc:
            logger.error("Ryu get_stats failed: %s", exc)
            return {"dpid": dpid, "error": str(exc)}

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _post(self, path: str, payload: Dict) -> Dict:
        url = self.base_url + path
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> Dict:
        url = self.base_url + path
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    # ── Payload builder ───────────────────────────────────────────────────────

    @staticmethod
    def _entry_to_ryu(entry: FlowEntry) -> Dict:
        """Convert FlowEntry to Ryu REST API payload."""
        match = {}
        m = entry.match
        if m.src_ip:   match["nw_src"] = m.src_ip
        if m.dst_ip:   match["nw_dst"] = m.dst_ip
        if m.src_port: match["tp_src"] = m.src_port
        if m.dst_port: match["tp_dst"] = m.dst_port

        if entry.action == "drop":
            actions = []
        elif entry.action == "rate_limit":
            # Ryu meter-based rate limiting (requires OF1.3+)
            actions = [{"type": "METER", "meter_id": 1}]
        else:
            actions = [{"type": "OUTPUT", "port": "NORMAL"}]

        return {
            "dpid": 1,
            "priority": entry.priority,
            "idle_timeout": entry.idle_timeout,
            "match": match,
            "actions": actions,
        }


# ── Factory ────────────────────────────────────────────────────────────────────

def create_controller(controller_type: str = "mock", **kwargs) -> SDNController:
    if controller_type == "ryu":
        url = kwargs.get("controller_url", "http://localhost:8080")
        return RyuController(base_url=url)
    return MockController()
