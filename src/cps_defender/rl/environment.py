"""
Reinforcement Learning environment for adaptive IDS response.

Gym-like interface but with zero gym dependency — the agent only needs
numpy arrays for observation/action spaces. This avoids a heavy framework
for what is essentially a tabular/shallow-NN problem.

State (8-dim float32 vector): see NetworkState.to_vector()
Actions (discrete int 0-4): MitigationAction enum values

Reward design (tuned empirically on simulated scenarios):
  +2.0   attack successfully stopped (true positive mitigation)
  +1.0   correct MONITOR on benign traffic (true negative)
  -1.0   false positive (mitigated benign flow — operator cost)
  -2.0   false negative (attack not mitigated — damage cost)
  -0.5   QoS degradation per 10 ms added latency
  -0.3   unnecessary escalation (ISOLATE when RATE_LIMIT would suffice)

Episode terminates after max_steps or when the scenario ends.
"""
from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from cps_defender.core.models import Alert, AttackType, NetworkState
from cps_defender.sdn.mitigation import MitigationAction, MitigationEngine, N_ACTIONS

logger = logging.getLogger(__name__)

# Observation dimensionality (matches NetworkState.to_vector())
OBS_DIM = 8


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class CPSEnvironment:
    """
    Simulated CPS/ICS network environment for RL training.

    Generates alert sequences and measures the impact of mitigation decisions.
    The simulator is deliberately simple so that the RL algorithm can learn
    quickly (< 500 episodes) without a full Mininet emulation.
    """

    def __init__(
        self,
        engine: Optional[MitigationEngine] = None,
        max_steps: int = 200,
        attack_prob: float = 0.30,
        seed: Optional[int] = 42,
    ):
        self.engine = engine or MitigationEngine()
        self.max_steps = max_steps
        self.attack_prob = attack_prob
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)

        # Episode state
        self._step_idx = 0
        self._state = NetworkState()
        self._alert_queue: Deque[Optional[Alert]] = deque()
        self._episode_rewards: List[float] = []
        self._tp = self._fp = self._tn = self._fn = 0

    # ── Gym-like interface ────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns initial observation."""
        self._step_idx = 0
        self._state = NetworkState()
        self._alert_queue.clear()
        self._episode_rewards.clear()
        self._tp = self._fp = self._tn = self._fn = 0
        return self._state.to_vector()

    def step(self, action: int) -> StepResult:
        """
        Apply action and advance simulation by one step.
        Returns (next_obs, reward, done, info).
        """
        action = MitigationAction(min(max(action, 0), N_ACTIONS - 1))
        self._step_idx += 1

        # Generate the current alert (or None = normal traffic)
        alert, is_attack = self._sample_event()
        reward = self._compute_reward(action, alert, is_attack)
        self._update_state(action, alert, is_attack)
        self._episode_rewards.append(reward)

        done = self._step_idx >= self.max_steps
        obs = self._state.to_vector()

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": self._step_idx,
                "is_attack": is_attack,
                "action": action.name,
                "alert": alert.attack_type if alert else None,
                "qos": self._state.qos_score,
                "tp": self._tp, "fp": self._fp,
                "tn": self._tn, "fn": self._fn,
            },
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _sample_event(self) -> Tuple[Optional[Alert], bool]:
        """Probabilistically generate normal traffic or an attack alert."""
        is_attack = self.rng.random() < self.attack_prob
        if not is_attack:
            return None, False

        attack_types = [
            AttackType.SCAN, AttackType.REPLAY,
            AttackType.CMD_INJECTION, AttackType.FLOODING, AttackType.MITM,
        ]
        atype = self.random.choice(attack_types)
        severity_val = self.random.randint(1, 4)
        from cps_defender.core.models import Severity
        sev = Severity(severity_val)

        alert = Alert(
            uid=f"sim-{self._step_idx}",
            timestamp=time.time(),
            src_ip=f"10.0.{self.random.randint(0,9)}.{self.random.randint(1,254)}",
            dst_ip="10.0.0.1",
            attack_type=atype,
            severity=sev,
            confidence=float(self.rng.uniform(0.5, 1.0)),
            score=float(self.rng.uniform(0.5, 1.0)),
            detector="sim",
        )
        return alert, True

    def _compute_reward(
        self, action: MitigationAction, alert: Optional[Alert], is_attack: bool
    ) -> float:
        reward = 0.0

        if is_attack and alert:
            if action == MitigationAction.MONITOR:
                reward -= 2.0   # FN: attack not contained
                self._fn += 1
            else:
                # Proportional to severity matched to action strength
                action_strength = int(action)   # 1-4
                sev_val = int(alert.severity)   # 1-4
                if action_strength >= sev_val:
                    reward += 2.0               # TP: correctly mitigated
                    self._tp += 1
                    if action_strength > sev_val + 1:
                        reward -= 0.3           # over-escalation penalty
                else:
                    reward += 0.5               # partial mitigation
                    self._fn += 1
        else:
            # No attack
            if action == MitigationAction.MONITOR:
                reward += 1.0   # TN: correctly did nothing
                self._tn += 1
            else:
                reward -= 1.0   # FP: mitigated benign traffic
                self._fp += 1

        # QoS penalty
        qos_penalty = (1.0 - self._state.qos_score) * 0.5
        reward -= qos_penalty

        return float(reward)

    def _update_state(
        self, action: MitigationAction, alert: Optional[Alert], is_attack: bool
    ) -> None:
        s = self._state
        s.timestamp = time.time()
        s.current_action = int(action)

        # Rolling alert count (decaying)
        if alert:
            s.alert_count_1m = min(s.alert_count_1m + 1, 100)
            s.avg_severity = 0.9 * s.avg_severity + 0.1 * float(int(alert.severity))
            s.avg_confidence = 0.9 * s.avg_confidence + 0.1 * alert.confidence
        else:
            s.alert_count_1m = max(0, s.alert_count_1m - 1)
            s.avg_severity = 0.95 * s.avg_severity
            s.avg_confidence = 0.95 * s.avg_confidence

        # QoS impact of mitigation actions
        if action == MitigationAction.MONITOR:
            s.latency_ms = max(5.0, s.latency_ms - 0.5)
            s.qos_score = min(1.0, s.qos_score + 0.02)
        elif action == MitigationAction.RATE_LIMIT:
            s.latency_ms += 2.0
            s.qos_score -= 0.05
        elif action == MitigationAction.SEGMENT:
            s.latency_ms += 5.0
            s.qos_score -= 0.10
        elif action == MitigationAction.ISOLATE:
            s.latency_ms += 10.0
            s.qos_score -= 0.20
            s.isolated_hosts = min(s.isolated_hosts + 1, 20)

        s.qos_score = float(np.clip(s.qos_score, 0.0, 1.0))
        s.latency_ms = float(np.clip(s.latency_ms, 1.0, 1000.0))

    # ── Info ──────────────────────────────────────────────────────────────────

    def episode_summary(self) -> Dict:
        total = max(self._tp + self._fp + self._tn + self._fn, 1)
        return {
            "total_steps": self._step_idx,
            "total_reward": float(sum(self._episode_rewards)),
            "mean_reward":  float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0,
            "tp": self._tp, "fp": self._fp, "tn": self._tn, "fn": self._fn,
            "precision": self._tp / max(self._tp + self._fp, 1),
            "recall":    self._tp / max(self._tp + self._fn, 1),
            "final_qos": self._state.qos_score,
        }

    @property
    def action_space_n(self) -> int:
        return N_ACTIONS

    @property
    def obs_dim(self) -> int:
        return OBS_DIM
