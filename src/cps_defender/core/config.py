"""
Configuration management.

Design: YAML file + env var overrides (CPS_SECTION__KEY=val).
No external config libs — stdlib + pyyaml only.
Deep-merge lets you override just what you need without rewriting the whole file.
"""
from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# ── Defaults (all components read from here) ─────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {
        "log_level": "INFO",
        "log_file": "logs/cps_defender.log",
        "seed": 42,
    },
    "ids": {
        "ensemble_weights": {"rule_based": 0.40, "statistical": 0.30, "ml": 0.30},
        "alert_threshold": 0.50,
        "window_size": 100,
        "min_confidence": 0.25,
        "model_path": "data/models/detector.joblib",
    },
    "sdn": {
        "controller_type": "mock",          # mock | ryu
        "controller_url": "http://localhost:8080",
        "safety_critical_ports": [20000],   # DNP3 default; never blocked
        "max_isolation_sec": 300,
        "rate_limit_kbps": 100,
        "action_cooldown_sec": 5,
    },
    "rl": {
        "lr": 1e-3,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
        "batch_size": 32,
        "memory_size": 10_000,
        "target_update_freq": 50,
        "hidden_sizes": [64, 64],
        "max_episodes": 500,
        "max_steps": 200,
        "model_path": "data/models/rl_agent.npz",
    },
    "genai": {
        "augmentation_factor": 3,
        "noise_std": 0.08,
        "perturbation_budget": 0.15,
        "n_neighbors": 5,
        "latent_dim": 16,
        "vae_epochs": 100,
        "vae_lr": 1e-3,
    },
    "testbed": {
        "normal_flow_rate": 10,
        "attack_probability": 0.15,
        "seed": 42,
        "dnp3_polling_interval_s": 1.0,
        "device_count": 8,
        "sim_duration_s": 60,
    },
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    """Hierarchical, immutable-ish config object."""

    def __init__(self, config_path: Optional[str] = None):
        self._data: Dict[str, Any] = _deep_merge({}, DEFAULT_CONFIG)

        if config_path:
            p = Path(config_path)
            if p.exists():
                self._load_yaml(p)
            else:
                logger.warning("Config file not found: %s — using defaults", config_path)

        self._apply_env_overrides()

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_yaml(self, path: Path) -> None:
        with path.open() as fh:
            overrides = yaml.safe_load(fh) or {}
        self._data = _deep_merge(self._data, overrides)
        logger.info("Loaded config: %s", path)

    def _apply_env_overrides(self) -> None:
        """CPS_IDS__ALERT_THRESHOLD=0.6  →  config['ids']['alert_threshold'] = 0.6"""
        for env_key, raw_val in os.environ.items():
            if not env_key.startswith("CPS_"):
                continue
            parts = env_key[4:].lower().split("__")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            try:
                node[parts[-1]] = ast.literal_eval(raw_val)
            except (ValueError, SyntaxError):
                node[parts[-1]] = raw_val

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if node is default:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def as_dict(self) -> Dict:
        import copy
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config(keys={list(self._data.keys())})"


# Module-level singleton — call init_config() once at startup
_config: Optional[Config] = None


def init_config(path: Optional[str] = None) -> Config:
    global _config
    _config = Config(path)
    return _config


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
