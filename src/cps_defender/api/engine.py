"""
engine.py — Background detection engine for the Flask dashboard.

Runs the full IDS → Mitigation → RL pipeline in a background thread.
Results are pushed into a thread-safe queue consumed by the SSE endpoint.

Design: single shared Engine instance (singleton via module-level object).
Flask routes call engine.start() / engine.stop() and read engine.metrics.
The SSE endpoint drains engine.event_queue.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
import datetime
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import numpy as np

from cps_defender.core.events import reset_bus
from cps_defender.core.models import AttackType
from cps_defender.ids.pipeline import IDSPipeline
from cps_defender.sdn.controller import create_controller
from cps_defender.sdn.mitigation import MitigationEngine, MitigationAction
from cps_defender.testbed.traffic_sim import TrafficSimulator

logger = logging.getLogger(__name__)


# ── Severity → action heuristic ───────────────────────────────────────────────
_SEV_ACTION = {
    "critical": MitigationAction.ISOLATE,
    "high":     MitigationAction.RATE_LIMIT,
    "medium":   MitigationAction.SEGMENT,
    "low":      MitigationAction.MONITOR,
}


@dataclass
class AlertEvent:
    """One alert pushed to the SSE stream."""
    id:          str
    timestamp:   str
    src_ip:      str
    dst_ip:      str
    attack_type: str
    severity:    str
    confidence:  float
    score:       float
    action:      str
    mitigated:   bool
    true_label:  str  # ground-truth label (from simulator)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Metrics:
    """Cumulative performance counters, thread-safe via a lock."""
    total:    int = 0
    alerts:   int = 0
    tp: int = 0; fp: int = 0; fn: int = 0; tn: int = 0
    by_severity:   Dict[str, int] = field(default_factory=dict)
    by_attack:     Dict[str, int] = field(default_factory=dict)
    by_action:     Dict[str, int] = field(default_factory=dict)
    timeline:      List[Dict]     = field(default_factory=list)  # last 60 ticks
    flows_per_sec: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, flow_label: str, alert: Optional[Any],
               action_name: str, ts: str) -> None:
        with self._lock:
            self.total += 1
            is_attack = flow_label != AttackType.NORMAL
            if alert:
                self.alerts += 1
                sev = alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity)
                self.by_severity[sev] = self.by_severity.get(sev, 0) + 1
                atype = (alert.attack_type.value
                         if hasattr(alert.attack_type, "value")
                         else str(alert.attack_type))
                self.by_attack[atype] = self.by_attack.get(atype, 0) + 1
                self.by_action[action_name] = self.by_action.get(action_name, 0) + 1
                if is_attack: self.tp += 1
                else:          self.fp += 1
            else:
                if is_attack: self.fn += 1
                else:          self.tn += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            tpr  = self.tp / max(1, self.tp + self.fn)
            fpr  = self.fp / max(1, self.fp + self.tn)
            prec = self.tp / max(1, self.tp + self.fp)
            f1   = 2 * prec * tpr / max(1e-9, prec + tpr)
            return {
                "total":    self.total,
                "alerts":   self.alerts,
                "tp": self.tp, "fp": self.fp,
                "fn": self.fn, "tn": self.tn,
                "tpr":  round(tpr,  4),
                "fpr":  round(fpr,  4),
                "precision": round(prec, 4),
                "f1":   round(f1,   4),
                "by_severity": dict(self.by_severity),
                "by_attack":   dict(self.by_attack),
                "by_action":   dict(self.by_action),
                "timeline":    list(self.timeline[-60:]),
                "flows_per_sec": round(self.flows_per_sec, 1),
            }

    def reset(self) -> None:
        with self._lock:
            self.total = self.alerts = self.tp = self.fp = self.fn = self.tn = 0
            self.by_severity.clear(); self.by_attack.clear()
            self.by_action.clear();   self.timeline.clear()
            self.flows_per_sec = 0.0


class DetectionEngine:
    """
    Singleton background engine.

    start()  → spawns worker thread, begins simulation loop
    stop()   → signals thread to exit cleanly
    Attributes:
      metrics     — live Metrics object (snapshot() for JSON)
      event_queue — queue.Queue of AlertEvent pushed for SSE
      status      — "idle" | "running" | "training" | "stopping"
    """

    def __init__(self) -> None:
        self.metrics      = Metrics()
        self.event_queue: queue.Queue[Dict] = queue.Queue(maxsize=500)
        self.status       = "idle"
        self._thread: Optional[threading.Thread] = None
        self._stop_evt    = threading.Event()
        self._pipeline:   Optional[IDSPipeline]    = None
        self._engine:     Optional[MitigationEngine] = None
        self._rl_agent    = None

        # Configurable via /api/config
        self.attack_prob  = 0.30
        self.flow_delay   = 0.15   # seconds between flows
        self.use_rl       = False
        self.scenario     = None   # None = mixed
        self.warmup_flows = 200    # flows before live display
        self.seed         = 42

    # ── Public API ─────────────────────────────────────────────────

    def configure(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def start(self) -> bool:
        if self.status == "running":
            return False
        self._stop_evt.clear()
        self.metrics.reset()
        self.status = "training"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_evt.set()
        self.status = "stopping"

    def load_rl_agent(self, path: str) -> bool:
        try:
            from cps_defender.rl.agent import DQNAgent
            self._rl_agent = DQNAgent.load(path)
            self.use_rl = True
            return True
        except Exception as e:
            logger.warning("RL agent load failed: %s", e)
            return False

    # ── Worker thread ───────────────────────────────────────────────

    def _run(self) -> None:
        reset_bus()
        try:
            # 1. Train IDS on warm-up batch
            self._push_status("Training IDS on warm-up data…")
            sim = TrafficSimulator(
                seed=self.seed, attack_probability=self.attack_prob)
            warmup = sim.generate(n_flows=self.warmup_flows)
            self._pipeline = IDSPipeline()
            self._pipeline.train(warmup)
            ctrl = create_controller("mock")
            self._engine = MitigationEngine(ctrl)
            self._push_status(f"Warm-up complete ({self.warmup_flows} flows). Running live…")
            self.status = "running"

            # 2. Stream live flows
            batch_size  = 50
            batch_start = time.monotonic()
            batch_count = 0

            while not self._stop_evt.is_set():
                flows = sim.generate(n_flows=batch_size)
                for flow in flows:
                    if self._stop_evt.is_set():
                        break
                    self._process(flow)
                    batch_count += 1
                    elapsed = time.monotonic() - batch_start
                    if elapsed >= 1.0:
                        self.metrics.flows_per_sec = batch_count / elapsed
                        self.metrics.timeline.append({
                            "t":       datetime.datetime.now().strftime("%H:%M:%S"),
                            "alerts":  self.metrics.alerts,
                            "total":   self.metrics.total,
                            "tpr":     round(self.metrics.tp / max(1, self.metrics.tp + self.metrics.fn), 3),
                        })
                        batch_count = 0; batch_start = time.monotonic()
                    if self.flow_delay > 0:
                        time.sleep(self.flow_delay)
        except Exception as exc:
            logger.exception("Engine worker crashed: %s", exc)
            self._push_status(f"ERROR: {exc}")
        finally:
            self.status = "idle"

    def _process(self, flow) -> None:
        alert = self._pipeline.analyze(flow)
        ts    = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

        action_name = "—"
        mitigated   = False

        if alert:
            if self.use_rl and self._rl_agent:
                s = np.array([
                    min(self.metrics.alerts / max(1, self.metrics.total), 1.0),
                    self.metrics.tp / max(1, self.metrics.tp + self.metrics.fn),
                    self.metrics.fp / max(1, self.metrics.fp + self.metrics.tn),
                    0.9, 0.0, alert.confidence, alert.score,
                    float(str(alert.severity) in ("4", "critical")),
                ], dtype=np.float32)
                action = list(MitigationAction)[
                    self._rl_agent.select_action(s, greedy=True)]
            else:
                action = _SEV_ACTION.get(
                    alert.severity.value
                    if hasattr(alert.severity, "value") else str(alert.severity),
                    MitigationAction.MONITOR)

            result = self._engine.apply(action, alert)
            action_name = action.name if result.success else "BLOCKED"
            mitigated   = result.success

            ev = AlertEvent(
                id=str(uuid.uuid4())[:8],
                timestamp=ts,
                src_ip=str(alert.src_ip),
                dst_ip=str(alert.dst_ip),
                attack_type=(alert.attack_type.value
                             if hasattr(alert.attack_type, "value")
                             else str(alert.attack_type)),
                severity=(alert.severity.value
                          if hasattr(alert.severity, "value")
                          else str(alert.severity)),
                confidence=round(alert.confidence, 3),
                score=round(alert.score, 3),
                action=action_name,
                mitigated=mitigated,
                true_label=flow.label,
            )
            try:
                self.event_queue.put_nowait(ev.to_dict())
            except queue.Full:
                pass  # drop oldest — dashboard will catch up

        self.metrics.record(flow.label, alert, action_name, ts)

    def _push_status(self, msg: str) -> None:
        try:
            self.event_queue.put_nowait({"__status__": msg,
                                         "timestamp": datetime.datetime.now().isoformat()})
        except queue.Full:
            pass


# Module-level singleton
engine = DetectionEngine()
