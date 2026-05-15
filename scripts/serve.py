#!/usr/bin/env python3
"""
serve.py — Launch the CPS/ICS Defender web dashboard.

Usage
-----
# Quickstart (opens browser automatically)
python scripts/serve.py

# Custom host / port
python scripts/serve.py --host 0.0.0.0 --port 5001

# With a pre-trained ML model and RL agent
python scripts/serve.py \
    --ml-model data/models/ids_pipeline.joblib \
    --rl-agent data/models/dqn_agent.npz

# Faster simulation (more flows per second)
python scripts/serve.py --flow-delay 0.05 --attack-prob 0.4

Architecture
------------
Flask (main thread)  ←HTTP→  Browser
                                │
                      SSE stream (text/event-stream)
                                │
Engine worker thread ──────────► event_queue ──► SSE generator
        │
  IDSPipeline + MitigationEngine + (optional) DQNAgent
        │
  TrafficSimulator (synthetic DNP3 flows)
"""

import argparse
import sys
import os
import threading
import time
import webbrowser
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cps_defender.core.logging_setup import setup_logging
from cps_defender.api.app import create_app
from cps_defender.api.engine import engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPS/ICS Defender web dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",        default="127.0.0.1")
    p.add_argument("--port",        type=int, default=5000)
    p.add_argument("--debug",       action="store_true",
                   help="Enable Flask debug mode (auto-reload)")
    p.add_argument("--no-browser",  action="store_true",
                   help="Do not auto-open a browser tab")
    p.add_argument("--ml-model",    default=None,
                   help="Path to pre-trained IDSPipeline directory")
    p.add_argument("--rl-agent",    default=None,
                   help="Path to pre-trained DQN agent .npz")
    p.add_argument("--attack-prob", type=float, default=0.30,
                   help="Initial attack probability (0–1)")
    p.add_argument("--flow-delay",  type=float, default=0.15,
                   help="Seconds between simulated flows")
    p.add_argument("--warmup",      type=int,   default=200,
                   help="Warm-up flows before live display")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--autostart",   action="store_true",
                   help="Automatically start the engine on launch")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def print_banner(host: str, port: int) -> None:
    url = f"http://{host}:{port}"
    print()
    print("  ╔════════════════════════════════════════════╗")
    print("  ║   CPS/ICS Defender  —  Web Dashboard       ║")
    print("  ╠════════════════════════════════════════════╣")
    print(f"  ║   Dashboard : \033[96m{url}\033[0m")
    print("  ║   API base  : " + url + "/api/")
    print("  ║   Press Ctrl+C to stop                     ║")
    print("  ╚════════════════════════════════════════════╝")
    print()


def main() -> None:
    args = parse_args()
    setup_logging("DEBUG" if args.verbose else "WARNING")

    # Suppress Flask/Werkzeug startup noise unless verbose
    if not args.verbose:
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

    # Configure engine
    engine.configure(
        attack_prob  = args.attack_prob,
        flow_delay   = args.flow_delay,
        warmup_flows = args.warmup,
        seed         = args.seed,
    )

    # Load optional pre-trained models
    if args.ml_model and Path(args.ml_model).exists():
        print(f"[+] Pre-loading IDS pipeline: {args.ml_model}")
        try:
            from cps_defender.ids.pipeline import IDSPipeline
            engine._pipeline = IDSPipeline.load(args.ml_model)
        except Exception as e:
            print(f"    [!] Could not load pipeline ({e}); will train from scratch.")

    if args.rl_agent and Path(args.rl_agent).exists():
        print(f"[+] Pre-loading RL agent: {args.rl_agent}")
        ok = engine.load_rl_agent(args.rl_agent)
        if ok:
            print("    RL agent: active")
        else:
            print("    [!] RL agent load failed; using severity heuristic.")

    # Auto-start if requested
    if args.autostart:
        print("[+] Auto-starting engine…")
        engine.start()

    app = create_app(debug=args.debug)

    # Auto-open browser
    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print_banner(args.host, args.port)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,   # reloader conflicts with background thread
        threaded=True,        # each request in its own thread → SSE works
    )


if __name__ == "__main__":
    main()
