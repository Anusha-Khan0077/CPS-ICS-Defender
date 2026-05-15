#!/usr/bin/env python3
"""
train_rl.py — Train the DQN reinforcement-learning mitigation agent.

Examples
--------
python scripts/train_rl.py
python scripts/train_rl.py --episodes 500 --curriculum
python scripts/train_rl.py --load data/models/dqn_agent.npz --episodes 200
python scripts/train_rl.py --load data/models/dqn_agent.npz --eval-only
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cps_defender.core.logging_setup import setup_logging
from cps_defender.core.events import reset_bus
from cps_defender.rl.agent import DQNAgent, train_agent
from cps_defender.rl.environment import CPSEnvironment
from cps_defender.sdn.controller import create_controller
from cps_defender.sdn.mitigation import MitigationEngine
from cps_defender.testbed.traffic_sim import TrafficSimulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DQN mitigation agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",       type=int,   default=300)
    p.add_argument("--max-steps",      type=int,   default=200)
    p.add_argument("--eval-interval",  type=int,   default=50)
    p.add_argument("--eval-episodes",  type=int,   default=10)
    p.add_argument("--curriculum",     action="store_true")
    p.add_argument("--load",           default=None)
    p.add_argument("--save",           default="data/models/dqn_agent.npz")
    p.add_argument("--eval-only",      action="store_true")
    p.add_argument("--attack-prob",    type=float, default=0.30)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("-v", "--verbose",  action="store_true")
    return p.parse_args()


def build_env(attack_prob: float = 0.30, seed: int = 42) -> CPSEnvironment:
    reset_bus()
    return CPSEnvironment(attack_prob=attack_prob, seed=seed)


def evaluate(agent: DQNAgent, env: CPSEnvironment, n_ep: int = 10) -> dict:
    rewards, tprs, fprs = [], [], []
    for _ in range(n_ep):
        state = env.reset()
        total_r = 0.0
        done    = False
        while not done:
            action     = agent.select_action(state, greedy=True)
            result     = env.step(action)
            state      = result.observation
            total_r   += result.reward
            done       = result.done
        rewards.append(total_r)
        summary = env.episode_summary()
        tp = summary.get("tp", 0); fn = summary.get("fn", 0)
        fp = summary.get("fp", 0); tn = summary.get("tn", 0)
        tprs.append(tp / max(1, tp + fn))
        fprs.append(fp / max(1, fp + tn))
    return {"mean_reward": float(np.mean(rewards)),
            "std_reward":  float(np.std(rewards)),
            "mean_tpr":    float(np.mean(tprs)),
            "mean_fpr":    float(np.mean(fprs))}


CURRICULUM = [
    {"name": "easy",   "eps": 80,  "attack_prob": 0.15},
    {"name": "medium", "eps": 100, "attack_prob": 0.30},
    {"name": "hard",   "eps": 120, "attack_prob": 0.45},
]


def main() -> None:
    args = parse_args()
    setup_logging("INFO" if args.verbose else "WARNING")
    np.random.seed(args.seed)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print("  CPS/ICS Defender — DQN Mitigation Agent Training")
    print("=" * 58)

    env = build_env(args.attack_prob, args.seed)
    agent = DQNAgent(obs_dim=env.obs_dim, n_actions=env.action_space_n,
                     seed=args.seed)

    if args.load and Path(args.load).exists():
        print(f"[+] Loading checkpoint: {args.load}")
        agent = DQNAgent.load(args.load)

    if args.eval_only:
        print(f"\n[+] Evaluating ({args.eval_episodes} episodes) …")
        m = evaluate(agent, env, args.eval_episodes)
        print(f"  Mean reward : {m['mean_reward']:+.2f} ± {m['std_reward']:.2f}")
        print(f"  Mean TPR    : {m['mean_tpr']:.3f}")
        print(f"  Mean FPR    : {m['mean_fpr']:.3f}")
        return

    # ── Training ───────────────────────────────────────────────────
    if args.curriculum:
        print(f"\n[+] Curriculum training ({len(CURRICULUM)} stages) …\n")
        for stage in CURRICULUM:
            print(f"  Stage [{stage['name'].upper():6s}]  "
                  f"{stage['eps']} episodes  attack_prob={stage['attack_prob']}")
            env   = build_env(stage["attack_prob"], args.seed)
            history = train_agent(
                agent, env,
                n_episodes=stage["eps"],
                max_steps=args.max_steps,
                eval_interval=args.eval_interval,
                verbose=args.verbose,
            )
            if history:
                last = history[-1]
                print(f"    → ep_reward={last.get('episode_reward', 0):+.1f}"
                      f"  ε={agent.epsilon:.3f}\n")
    else:
        print(f"\n[+] Training for {args.episodes} episodes …\n")
        t0      = time.perf_counter()
        history = train_agent(
            agent, env,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            verbose=args.verbose,
        )
        print(f"\n  Completed in {time.perf_counter() - t0:.1f}s")

    # ── Final eval ─────────────────────────────────────────────────
    print(f"\n[+] Final evaluation ({args.eval_episodes} episodes) …")
    m = evaluate(agent, env, args.eval_episodes)
    print(f"  Mean reward : {m['mean_reward']:+.2f} ± {m['std_reward']:.2f}")
    print(f"  Mean TPR    : {m['mean_tpr']:.3f}")
    print(f"  Mean FPR    : {m['mean_fpr']:.3f}")

    agent.save(args.save)
    print(f"\n[+] Agent saved → {args.save}\nDone.")


if __name__ == "__main__":
    main()
