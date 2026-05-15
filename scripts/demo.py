#!/usr/bin/env python3
"""
demo.py — End-to-end demonstration of the CPS/ICS Defender pipeline.

Examples
--------
python scripts/demo.py                             # Quickstart
python scripts/demo.py --attack-prob 0.5           # More attacks
python scripts/demo.py --ml-model data/models/ids_pipeline.joblib
python scripts/demo.py --rl-agent  data/models/dqn_agent.npz
python scripts/demo.py --stream --interval 0.05    # Simulated live feed
python scripts/demo.py --zeek-log zeek_logs/dnp3_flows.log
"""

import argparse
import signal
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cps_defender.core.events import reset_bus
from cps_defender.core.logging_setup import setup_logging
from cps_defender.core.models import FlowRecord, Alert
from cps_defender.ids.pipeline import IDSPipeline
from cps_defender.sdn.controller import create_controller
from cps_defender.sdn.mitigation import MitigationEngine, MitigationAction
from cps_defender.testbed.traffic_sim import TrafficSimulator

# ── ANSI helpers ──────────────────────────────────────────────────
RED, YELLOW, GREEN, CYAN = "\033[91m", "\033[93m", "\033[92m", "\033[96m"
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"

SEV_CLR = {"critical": RED+BOLD, "high": RED, "medium": YELLOW, "low": CYAN}
ACT_CLR = {"BLOCK": RED+BOLD, "ISOLATE": RED, "SEGMENT": YELLOW,
           "RATE_LIMIT": CYAN, "MONITOR": GREEN}

BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════╗
║       CPS / ICS  DEFENDER  —  Live Demo              ║
║  Zeek IDS + SDN Mitigation + RL Adaptive Response    ║
╚══════════════════════════════════════════════════════╝{RESET}
"""

SEV_TO_ACTION = {
    "critical": MitigationAction.ISOLATE,
    "high":     MitigationAction.RATE_LIMIT,
    "medium":   MitigationAction.SEGMENT,
    "low":      MitigationAction.MONITOR,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPS/ICS Defender demo",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n-flows",     type=int,   default=300)
    p.add_argument("--attack-prob", type=float, default=0.30)
    p.add_argument("--warmup",      type=int,   default=150,
                   help="Flows used to initialise detectors (not displayed)")
    p.add_argument("--ml-model",    default=None)
    p.add_argument("--rl-agent",    default=None)
    p.add_argument("--zeek-log",    default=None)
    p.add_argument("--stream",      action="store_true")
    p.add_argument("--interval",    type=float, default=0.05)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


class Stats:
    def __init__(self):
        self.total = self.alerts = self.tp = self.fp = self.fn = self.tn = 0
        self.by_sev: dict[str,int] = {}
        self.actions: dict[str,int] = {}

    def record(self, flow: FlowRecord, alert: Optional[Alert]):
        self.total += 1
        is_attack = flow.label != "normal"
        if alert:
            self.alerts += 1
            sev = alert.severity.value
            self.by_sev[sev] = self.by_sev.get(sev, 0) + 1
            if is_attack: self.tp += 1
            else:          self.fp += 1
        else:
            if is_attack: self.fn += 1
            else:          self.tn += 1

    def record_action(self, act: str):
        self.actions[act] = self.actions.get(act, 0) + 1

    def summary(self) -> str:
        tpr  = self.tp / max(1, self.tp + self.fn)
        fpr  = self.fp / max(1, self.fp + self.tn)
        prec = self.tp / max(1, self.tp + self.fp)
        lines = [
            f"\n{BOLD}{'─'*58}{RESET}",
            f"{BOLD}  SUMMARY{RESET}",
            f"{'─'*58}",
            f"  Flows processed  : {self.total}",
            f"  Alerts fired     : {self.alerts}",
            f"  True Positives   : {self.tp}",
            f"  False Positives  : {self.fp}",
            f"  False Negatives  : {self.fn}",
            f"  True Negatives   : {self.tn}",
            f"  TPR (Recall)     : {GREEN}{tpr:.3f}{RESET}",
            f"  Precision        : {GREEN}{prec:.3f}{RESET}",
            f"  FPR              : {RED}{fpr:.3f}{RESET}",
        ]
        if self.by_sev:
            lines.append("\n  Alerts by severity:")
            for sev, cnt in sorted(self.by_sev.items()):
                c = SEV_CLR.get(sev, "")
                lines.append(f"    {c}{sev.upper():10s}{RESET}  {cnt}")
        if self.actions:
            lines.append("\n  Mitigation actions:")
            for act, cnt in sorted(self.actions.items(), key=lambda x: -x[1]):
                c = ACT_CLR.get(act.upper(), "")
                lines.append(f"    {c}{act:12s}{RESET}  {cnt}")
        lines.append(f"{'─'*58}")
        return "\n".join(lines)


def main() -> None:
    args = parse_args()
    setup_logging("DEBUG" if args.verbose else "ERROR")
    reset_bus()
    np.random.seed(args.seed)
    print(BANNER)

    # ── IDS pipeline ───────────────────────────────────────────────
    if args.ml_model and Path(args.ml_model).exists():
        print(f"[+] Loading IDS pipeline: {args.ml_model}")
        pipeline = IDSPipeline.load(args.ml_model)
    else:
        print("[+] Creating fresh IDS pipeline …")
        pipeline = IDSPipeline()

    # ── SDN + mitigation ───────────────────────────────────────────
    print("[+] Initialising Mock SDN controller …")
    ctrl   = create_controller("mock")
    engine = MitigationEngine(ctrl)

    # ── RL agent ───────────────────────────────────────────────────
    rl_agent = None
    if args.rl_agent and Path(args.rl_agent).exists():
        from cps_defender.rl.agent import DQNAgent
        print(f"[+] Loading RL agent: {args.rl_agent}")
        rl_agent = DQNAgent.load(args.rl_agent)
    else:
        print("    RL agent: not loaded (severity → action heuristic)")

    # ── Traffic ────────────────────────────────────────────────────
    if args.zeek_log:
        print(f"[+] Loading Zeek log: {args.zeek_log}")
        from cps_defender.ids.feature_extractor import zeek_logs_to_flows
        flows = zeek_logs_to_flows(args.zeek_log)
        print(f"    {len(flows)} flows loaded")
    else:
        print(f"[+] Generating {args.n_flows} flows (attack_prob={args.attack_prob}) …")
        sim   = TrafficSimulator(seed=args.seed, attack_probability=args.attack_prob)
        flows = sim.generate(n_flows=args.n_flows)

    if not flows:
        print("[!] No flows. Exiting."); return

    # ── Warm-up ────────────────────────────────────────────────────
    warmup_n = min(args.warmup, len(flows) // 2, 200)
    if warmup_n >= 20 and not (args.ml_model and Path(args.ml_model).exists()):
        print(f"[+] Warming up IDS on {warmup_n} flows …")
        pipeline.train(flows[:warmup_n])
        flows = flows[warmup_n:]
        print(f"    Live demo flows: {len(flows)}")

    stats   = Stats()
    running = True
    signal.signal(signal.SIGINT, lambda *_: globals().update(running=False)
                  or print(f"\n{YELLOW}Stopping…{RESET}"))

    print(f"\n{'─'*64}")
    print(f"  {'Time':8s}  {'Src IP':15s}  {'Attack':20s}  {'Sev':8s}  Action")
    print(f"{'─'*64}")

    for flow in flows:
        if not running: break

        alert = pipeline.analyze(flow)
        stats.record(flow, alert)

        if alert:
            # Action selection
            if rl_agent:
                state_vec = np.array([
                    min(stats.alerts / max(1, stats.total), 1.0),
                    stats.tp / max(1, stats.tp + stats.fn),
                    stats.fp / max(1, stats.fp + stats.tn),
                    0.9, 0.0,
                    alert.confidence, alert.score,
                    float(alert.severity.value == "critical"),
                ], dtype=np.float32)
                action = list(MitigationAction)[
                    rl_agent.select_action(state_vec, greedy=True)]
            else:
                action = SEV_TO_ACTION.get(alert.severity.value,
                                           MitigationAction.MONITOR)

            result   = engine.apply(action, alert)
            act_name = action.name if result.success else "BLOCKED(safety)"
            stats.record_action(act_name)

            ts    = time.strftime("%H:%M:%S")
            src   = str(alert.src_ip)[:15]
            atype = str(alert.attack_type.value)[:20]
            sc    = SEV_CLR.get(alert.severity.value, "")
            ac    = ACT_CLR.get(act_name.upper(), "")
            print(f"  {ts}  {src:15s}  {atype:20s}  "
                  f"{sc}{alert.severity.value.upper():8s}{RESET}  {ac}{act_name}{RESET}")

        if args.stream:
            time.sleep(args.interval)

    print(stats.summary())


if __name__ == "__main__":
    main()
