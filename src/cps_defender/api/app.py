"""
app.py — Flask application factory + all routes.

Routes
------
GET  /                   Dashboard SPA
GET  /api/status         Engine status + current config
GET  /api/metrics        Current metrics snapshot (JSON)
POST /api/start          Start simulation
POST /api/stop           Stop simulation
POST /api/config         Update engine config
GET  /api/stream         Server-Sent Events stream (alerts + ticks)
GET  /api/history        Last N alert events as JSON list

Design decisions
----------------
* Single blueprint keeps the codebase flat — no need for multiple blueprints
  at this scale.
* SSE over WebSockets: SSE is standard HTTP, works with any proxy, requires
  zero extra libraries. Flask's response streaming is all we need.
* Generator-based SSE: the /api/stream response is a generator that yields
  "data: …\n\n" lines. Each client gets its own generator draining the
  shared queue via a per-client copy mechanism (requeue unseen items).
* CORS header is set on /api/* so the dashboard can be loaded from any origin
  during development.
"""
from __future__ import annotations

import json
import time
import queue
import logging
from typing import Iterator

from flask import (Flask, render_template, Response,
                   request, jsonify, stream_with_context)

from .engine import engine, DetectionEngine

logger = logging.getLogger(__name__)

# Sliding window: store last 300 alert events for /api/history
_HISTORY: list[dict] = []
_HISTORY_MAX = 300


def create_app(debug: bool = False) -> Flask:
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")
    app.config["DEBUG"] = debug

    # ── Dashboard ──────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    # ── REST endpoints ─────────────────────────────────────────────

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "status":       engine.status,
            "attack_prob":  engine.attack_prob,
            "flow_delay":   engine.flow_delay,
            "use_rl":       engine.use_rl,
            "warmup_flows": engine.warmup_flows,
            "seed":         engine.seed,
        })

    @app.route("/api/metrics")
    def api_metrics():
        return jsonify(engine.metrics.snapshot())

    @app.route("/api/history")
    def api_history():
        n = min(int(request.args.get("n", 50)), _HISTORY_MAX)
        return jsonify(_HISTORY[-n:])

    @app.route("/api/start", methods=["POST"])
    def api_start():
        ok = engine.start()
        return jsonify({"started": ok, "status": engine.status})

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        engine.stop()
        return jsonify({"status": "stopping"})

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        if engine.status == "running":
            return jsonify({"error": "Stop the engine before resetting"}), 400
        engine.metrics.reset()
        _HISTORY.clear()
        return jsonify({"reset": True})

    @app.route("/api/config", methods=["POST"])
    def api_config():
        data = request.get_json(force=True) or {}
        allowed = {"attack_prob", "flow_delay", "use_rl",
                   "warmup_flows", "seed"}
        updates = {k: v for k, v in data.items() if k in allowed}

        # Type coercions
        if "attack_prob"  in updates: updates["attack_prob"]  = float(updates["attack_prob"])
        if "flow_delay"   in updates: updates["flow_delay"]   = float(updates["flow_delay"])
        if "warmup_flows" in updates: updates["warmup_flows"] = int(updates["warmup_flows"])
        if "seed"         in updates: updates["seed"]         = int(updates["seed"])
        if "use_rl"       in updates: updates["use_rl"]       = bool(updates["use_rl"])

        engine.configure(**updates)
        return jsonify({"updated": updates, "status": engine.status})

    # ── Server-Sent Events ─────────────────────────────────────────

    @app.route("/api/stream")
    def api_stream():
        """
        Streams newline-delimited SSE events.

        Each event is one of:
          data: {"type": "alert",  ...alert fields...}
          data: {"type": "tick",   ...metrics snapshot...}
          data: {"type": "status", "message": "..."}

        Ticks are injected every ~1 s even when no alerts fire,
        so the chart always updates.
        """
        def generate() -> Iterator[str]:
            last_tick = time.monotonic()
            while True:
                # Drain alert queue
                try:
                    item = engine.event_queue.get(timeout=0.2)
                    if "__status__" in item:
                        yield _sse({"type": "status",
                                    "message": item["__status__"]})
                    else:
                        # Store in history
                        _HISTORY.append(item)
                        if len(_HISTORY) > _HISTORY_MAX:
                            _HISTORY.pop(0)
                        yield _sse({"type": "alert", **item})
                except queue.Empty:
                    pass

                # Emit periodic tick with full metrics
                now = time.monotonic()
                if now - last_tick >= 1.0:
                    yield _sse({"type": "tick",
                                **engine.metrics.snapshot(),
                                "engine_status": engine.status})
                    last_tick = now

        resp = Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx buffering
                "Access-Control-Allow-Origin": "*",
            },
        )
        return resp

    # ── CORS for API ───────────────────────────────────────────────

    @app.after_request
    def add_cors(response):
        if request.path.startswith("/api/"):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    return app


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"
