"""
Lightweight publish-subscribe event bus.

Design: chosen over external queues (Redis/Kafka) to eliminate runtime deps
while still decoupling IDS → SDN → RL signal flows.
Thread-safe; supports wildcard "*" subscriptions for logging/monitoring.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    FLOW_RECEIVED   = "flow_received"
    ALERT_GENERATED = "alert_generated"
    MITIGATION_APPLIED = "mitigation_applied"
    RL_ACTION       = "rl_action"
    TRAINING_STEP   = "training_step"
    SYSTEM_STATUS   = "system_status"


@dataclass
class Event:
    type: str
    payload: Any
    source: str = ""
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"Event(type={self.type!r}, source={self.source!r}, ts={self.timestamp:.3f})"


class EventBus:
    """Thread-safe pub/sub bus with history ring-buffer."""

    def __init__(self, history_limit: int = 500):
        self._subs: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._history: List[Event] = []
        self._history_limit = history_limit

    # ── Subscription ─────────────────────────────────────────────────────────

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subs[event_type].append(callback)
        logger.debug("Subscribed %s → %s", event_type, getattr(callback, "__qualname__", repr(callback)))

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            try:
                self._subs[event_type].remove(callback)
            except ValueError:
                pass

    # ── Publishing ───────────────────────────────────────────────────────────

    def publish(self, event: Event) -> int:
        """Deliver event synchronously. Returns count of notified subscribers."""
        with self._lock:
            callbacks = list(self._subs.get(event.type, []))
            callbacks.extend(self._subs.get("*", []))  # wildcards
            self._history.append(event)
            if len(self._history) > self._history_limit:
                del self._history[0]

        notified = 0
        for cb in callbacks:
            try:
                cb(event)
                notified += 1
            except Exception:
                logger.exception("Handler %s raised for event %s", getattr(cb, "__qualname__", repr(cb)), event)
        return notified

    def emit(self, event_type: str, payload: Any, source: str = "") -> int:
        """Shorthand publish."""
        return self.publish(Event(type=event_type, payload=payload, source=source))

    # ── History ───────────────────────────────────────────────────────────────

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
        since: float = 0.0,
    ) -> List[Event]:
        with self._lock:
            events = list(self._history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events[-limit:]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "history_size": len(self._history),
                "subscriber_count": sum(len(v) for v in self._subs.values()),
                "topics": len(self._subs),
            }


# ── Singleton ─────────────────────────────────────────────────────────────────
_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_bus() -> None:
    """Used in tests to get a fresh bus."""
    global _bus
    _bus = EventBus()
