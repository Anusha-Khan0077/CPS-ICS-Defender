"""
CPS/ICS Traffic Simulator.

Generates realistic DNP3-style flow records without requiring real hardware.
Used for:
  • Generating training datasets (labelled flows).
  • Integration and load testing.
  • RL environment warmup (fills the replay buffer with plausible transitions).

DNP3 network model:
  • 1 Master Station (10.0.0.1)  — polls all outstations
  • N Outstations (10.0.1.x)    — respond to master; send unsolicited reports
  • 1 HMI (10.0.0.10)            — operator display, occasional config changes
  • Optional Engineering WS (10.0.0.20) — maintenance access

Polling model: master polls each outstation at a configurable interval.
Inter-arrival times follow an exponential distribution (memoryless, realistic).
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from cps_defender.core.models import (
    AttackType,
    FlowRecord,
    Protocol,
)

logger = logging.getLogger(__name__)

# ── DNP3 function codes used in simulation ────────────────────────────────────
FC_READ              = 1
FC_WRITE             = 2
FC_DIRECT_OPERATE    = 3
FC_RESPONSE          = 129
FC_UNSOLICITED       = 130
FC_RECORD_TIME       = 19
FC_ENABLE_UNSO       = 20
FC_DISABLE_UNSO      = 21


@dataclass
class Device:
    ip: str
    role: str          # master | outstation | hmi | engineering
    address: int       # DNP3 data link address


class TrafficSimulator:
    """
    Generates labelled CPS/ICS FlowRecords at a configurable rate.

    Usage:
        sim = TrafficSimulator(n_outstations=8, seed=42)
        flows = sim.generate(n_flows=5000)
        normal_flows, attack_flows = sim.split_by_label(flows)
    """

    def __init__(
        self,
        n_outstations: int = 8,
        polling_interval_s: float = 1.0,
        attack_probability: float = 0.15,
        seed: int = 42,
    ):
        self.n_outstations = n_outstations
        self.polling_interval_s = polling_interval_s
        self.attack_probability = attack_probability
        self._rng = np.random.default_rng(seed)
        self._sim_time = 0.0

        # Build topology
        self.master = Device("10.0.0.1",  "master",      1)
        self.hmi    = Device("10.0.0.10", "hmi",         2)
        self.outstations: List[Device] = [
            Device(f"10.0.1.{i+1}", "outstation", 10 + i)
            for i in range(n_outstations)
        ]

    # ── Main generation interface ─────────────────────────────────────────────

    def generate(self, n_flows: int = 5000) -> List[FlowRecord]:
        flows: List[FlowRecord] = []
        t = 1_700_000_000.0  # Unix timestamp baseline

        while len(flows) < n_flows:
            # Decide whether this event is an attack
            is_attack = self._rng.random() < self.attack_probability

            if is_attack:
                flow = self._gen_attack_flow(t)
            else:
                flow = self._gen_normal_flow(t)

            flows.append(flow)
            # Advance time by a random inter-event interval
            t += float(self._rng.exponential(self.polling_interval_s / self.n_outstations))

        logger.info(
            "Generated %d flows: %d normal, %d attack",
            n_flows,
            sum(1 for f in flows if f.label == AttackType.NORMAL),
            sum(1 for f in flows if f.label != AttackType.NORMAL),
        )
        return flows

    def stream(self, n_flows: int = 1000) -> Generator[FlowRecord, None, None]:
        """Yield flows one at a time (for online/streaming scenarios)."""
        t = time.time()
        for _ in range(n_flows):
            is_attack = self._rng.random() < self.attack_probability
            flow = self._gen_attack_flow(t) if is_attack else self._gen_normal_flow(t)
            yield flow
            t += float(self._rng.exponential(self.polling_interval_s / max(self.n_outstations, 1)))

    @staticmethod
    def split_by_label(flows: List[FlowRecord]) -> Tuple[List[FlowRecord], List[FlowRecord]]:
        normal  = [f for f in flows if f.label == AttackType.NORMAL]
        attacks = [f for f in flows if f.label != AttackType.NORMAL]
        return normal, attacks

    # ── Normal traffic generators ─────────────────────────────────────────────

    def _gen_normal_flow(self, t: float) -> FlowRecord:
        """Pick a random legitimate DNP3 exchange pattern."""
        pattern = self._rng.choice(["poll", "unsolicited", "time_sync", "hmi_read"])
        if pattern == "poll":
            return self._poll_flow(t)
        if pattern == "unsolicited":
            return self._unsolicited_flow(t)
        if pattern == "time_sync":
            return self._time_sync_flow(t)
        return self._hmi_read_flow(t)

    def _poll_flow(self, t: float) -> FlowRecord:
        rtu = self._rng.choice(self.outstations)
        dur = float(self._rng.normal(0.05, 0.01))
        pkts = int(self._rng.integers(2, 6))
        return FlowRecord(
            uid=str(uuid.uuid4()),
            timestamp=t,
            src_ip=self.master.ip,
            dst_ip=rtu.ip,
            src_port=int(self._rng.integers(49152, 65535)),
            dst_port=20000,
            protocol=Protocol.DNP3,
            flow_duration=max(0.001, dur),
            pkt_count=pkts,
            byte_count=pkts * int(self._rng.integers(24, 128)),
            function_code=FC_READ,
            unique_fc_count=2,           # REQUEST + RESPONSE
            req_resp_ratio=0.5,
            inter_arrival_mean=float(self._rng.normal(self.polling_interval_s * 1000, 50)),
            inter_arrival_std=float(abs(self._rng.normal(10, 5))),
            is_broadcast=0,
            direction=0,
            burst_count=int(self._rng.integers(1, 5)),
            error_rate=float(self._rng.uniform(0, 0.02)),
            label=AttackType.NORMAL,
            label_id=0,
        )

    def _unsolicited_flow(self, t: float) -> FlowRecord:
        rtu = self._rng.choice(self.outstations)
        pkts = int(self._rng.integers(1, 4))
        return FlowRecord(
            uid=str(uuid.uuid4()),
            timestamp=t,
            src_ip=rtu.ip,
            dst_ip=self.master.ip,
            src_port=20000,
            dst_port=int(self._rng.integers(49152, 65535)),
            protocol=Protocol.DNP3,
            flow_duration=float(self._rng.uniform(0.01, 0.1)),
            pkt_count=pkts,
            byte_count=pkts * int(self._rng.integers(20, 80)),
            function_code=FC_UNSOLICITED,
            unique_fc_count=1,
            req_resp_ratio=0.0,
            inter_arrival_mean=float(self._rng.normal(2000, 200)),
            inter_arrival_std=float(abs(self._rng.normal(30, 10))),
            is_broadcast=0,
            direction=1,
            burst_count=1,
            error_rate=0.0,
            label=AttackType.NORMAL,
            label_id=0,
        )

    def _time_sync_flow(self, t: float) -> FlowRecord:
        rtu = self._rng.choice(self.outstations)
        return FlowRecord(
            uid=str(uuid.uuid4()),
            timestamp=t,
            src_ip=self.master.ip,
            dst_ip=rtu.ip,
            src_port=int(self._rng.integers(49152, 65535)),
            dst_port=20000,
            protocol=Protocol.DNP3,
            flow_duration=float(self._rng.uniform(0.005, 0.02)),
            pkt_count=2,
            byte_count=60,
            function_code=FC_RECORD_TIME,
            unique_fc_count=1,
            req_resp_ratio=0.5,
            inter_arrival_mean=float(self._rng.normal(3600_000, 1000)),  # hourly
            inter_arrival_std=float(abs(self._rng.normal(50, 10))),
            is_broadcast=0,
            direction=0,
            burst_count=1,
            error_rate=0.0,
            label=AttackType.NORMAL,
            label_id=0,
        )

    def _hmi_read_flow(self, t: float) -> FlowRecord:
        rtu = self._rng.choice(self.outstations)
        pkts = int(self._rng.integers(4, 20))
        return FlowRecord(
            uid=str(uuid.uuid4()),
            timestamp=t,
            src_ip=self.hmi.ip,
            dst_ip=rtu.ip,
            src_port=int(self._rng.integers(49152, 65535)),
            dst_port=20000,
            protocol=Protocol.DNP3,
            flow_duration=float(self._rng.uniform(0.1, 2.0)),
            pkt_count=pkts,
            byte_count=pkts * int(self._rng.integers(50, 300)),
            function_code=FC_READ,
            unique_fc_count=2,
            req_resp_ratio=float(self._rng.uniform(0.4, 0.6)),
            inter_arrival_mean=float(self._rng.normal(500, 100)),
            inter_arrival_std=float(abs(self._rng.normal(50, 20))),
            is_broadcast=0,
            direction=0,
            burst_count=int(self._rng.integers(2, 10)),
            error_rate=float(self._rng.uniform(0, 0.01)),
            label=AttackType.NORMAL,
            label_id=0,
        )

    # ── Attack traffic generators ─────────────────────────────────────────────

    def _gen_attack_flow(self, t: float) -> FlowRecord:
        attack_type = self._rng.choice([
            AttackType.SCAN,
            AttackType.REPLAY,
            AttackType.CMD_INJECTION,
            AttackType.FLOODING,
            AttackType.MITM,
        ])
        generators = {
            AttackType.SCAN:          self._scan_flow,
            AttackType.REPLAY:        self._replay_flow,
            AttackType.CMD_INJECTION: self._injection_flow,
            AttackType.FLOODING:      self._flood_flow,
            AttackType.MITM:          self._mitm_flow,
        }
        return generators[attack_type](t)

    def _scan_flow(self, t: float) -> FlowRecord:
        """Port/device scan — many short flows, sequential ports."""
        rtu = self._rng.choice(self.outstations)
        pkts = int(self._rng.integers(20, 100))
        return FlowRecord(
            uid=str(uuid.uuid4()), timestamp=t,
            src_ip=f"192.168.{self._rng.integers(0,255)}.{self._rng.integers(1,254)}",
            dst_ip=rtu.ip, src_port=int(self._rng.integers(1024, 65535)), dst_port=20000,
            protocol=Protocol.DNP3,
            flow_duration=float(self._rng.uniform(0.01, 0.4)),
            pkt_count=pkts, byte_count=pkts * 20,
            function_code=int(self._rng.integers(100, 200)),   # unusual FCs
            unique_fc_count=int(self._rng.integers(5, 20)),
            req_resp_ratio=0.95,
            inter_arrival_mean=float(self._rng.uniform(1, 10)),
            inter_arrival_std=float(self._rng.uniform(0, 5)),
            is_broadcast=0, direction=0,
            burst_count=int(self._rng.integers(50, 200)),
            error_rate=float(self._rng.uniform(0.2, 0.8)),
            label=AttackType.SCAN, label_id=1,
        )

    def _replay_flow(self, t: float) -> FlowRecord:
        """Captured packet replay — bulk unsolicited responses."""
        rtu = self._rng.choice(self.outstations)
        pkts = int(self._rng.integers(10, 50))
        return FlowRecord(
            uid=str(uuid.uuid4()), timestamp=t,
            src_ip=rtu.ip, dst_ip=self.master.ip,
            src_port=20000, dst_port=int(self._rng.integers(49152, 65535)),
            protocol=Protocol.DNP3,
            flow_duration=float(self._rng.uniform(0.5, 5.0)),
            pkt_count=pkts, byte_count=pkts * int(self._rng.integers(60, 200)),
            function_code=FC_UNSOLICITED,
            unique_fc_count=1,
            req_resp_ratio=0.0,
            inter_arrival_mean=float(self._rng.uniform(50, 200)),
            inter_arrival_std=float(self._rng.uniform(100, 500)),  # high variance
            is_broadcast=0, direction=1,
            burst_count=int(self._rng.integers(20, 100)),
            error_rate=float(self._rng.uniform(0, 0.05)),
            label=AttackType.REPLAY, label_id=2,
        )

    def _injection_flow(self, t: float) -> FlowRecord:
        """Unauthorized control command injection."""
        rtu = self._rng.choice(self.outstations)
        return FlowRecord(
            uid=str(uuid.uuid4()), timestamp=t,
            src_ip=f"10.0.{self._rng.integers(2,9)}.{self._rng.integers(1,50)}",
            dst_ip=rtu.ip, src_port=int(self._rng.integers(1024, 65535)), dst_port=20000,
            protocol=Protocol.DNP3,
            flow_duration=float(self._rng.uniform(0.1, 1.0)),
            pkt_count=int(self._rng.integers(2, 10)),
            byte_count=int(self._rng.integers(100, 500)),
            function_code=int(self._rng.choice([FC_DIRECT_OPERATE, FC_WRITE, 14, 15, 22])),
            unique_fc_count=int(self._rng.integers(1, 3)),
            req_resp_ratio=0.8,
            inter_arrival_mean=float(self._rng.normal(100, 30)),
            inter_arrival_std=float(self._rng.uniform(5, 50)),
            is_broadcast=int(self._rng.choice([0, 1], p=[0.3, 0.7])),
            direction=0,
            burst_count=int(self._rng.integers(1, 10)),
            error_rate=float(self._rng.uniform(0, 0.1)),
            label=AttackType.CMD_INJECTION, label_id=3,
        )

    def _flood_flow(self, t: float) -> FlowRecord:
        """DoS/flooding — massive packet counts, high burst."""
        rtu = self._rng.choice(self.outstations)
        pkts = int(self._rng.integers(500, 5000))
        return FlowRecord(
            uid=str(uuid.uuid4()), timestamp=t,
            src_ip=f"172.16.{self._rng.integers(0,255)}.{self._rng.integers(1,254)}",
            dst_ip=rtu.ip, src_port=int(self._rng.integers(1024, 65535)), dst_port=20000,
            protocol=Protocol.DNP3,
            flow_duration=float(self._rng.uniform(0.1, 2.0)),
            pkt_count=pkts, byte_count=pkts * int(self._rng.integers(64, 256)),
            function_code=FC_READ,
            unique_fc_count=1,
            req_resp_ratio=0.99,
            inter_arrival_mean=float(self._rng.uniform(0.1, 2.0)),
            inter_arrival_std=float(self._rng.uniform(0, 0.5)),
            is_broadcast=1, direction=0,
            burst_count=int(self._rng.integers(500, 5000)),
            error_rate=float(self._rng.uniform(0, 0.3)),
            label=AttackType.FLOODING, label_id=4,
        )

    def _mitm_flow(self, t: float) -> FlowRecord:
        """MitM indicator — traffic on unusual ports from internal subnet."""
        rtu = self._rng.choice(self.outstations)
        return FlowRecord(
            uid=str(uuid.uuid4()), timestamp=t,
            src_ip=f"10.0.0.{self._rng.integers(50,100)}",
            dst_ip=rtu.ip,
            src_port=int(self._rng.integers(1, 1023)),    # privileged port — suspicious
            dst_port=20000,
            protocol=Protocol.DNP3,
            flow_duration=float(self._rng.uniform(5, 30)),
            pkt_count=int(self._rng.integers(100, 1000)),
            byte_count=int(self._rng.integers(50000, 500000)),
            function_code=FC_RESPONSE,
            unique_fc_count=int(self._rng.integers(2, 5)),
            req_resp_ratio=float(self._rng.uniform(0.3, 0.7)),
            inter_arrival_mean=float(self._rng.normal(50, 5)),
            inter_arrival_std=float(self._rng.uniform(1, 10)),
            is_broadcast=0, direction=2,
            burst_count=int(self._rng.integers(10, 50)),
            error_rate=float(self._rng.uniform(0.05, 0.2)),
            label=AttackType.MITM, label_id=5,
        )
