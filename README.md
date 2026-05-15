# CPS/ICS Defender

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)](tests/)
[![Dependencies](https://img.shields.io/badge/deps-minimal-orange)](#dependencies)

> A closed-loop, protocol-aware **Intrusion Detection + SDN Mitigation** framework for
> industrial control system (ICS) networks — powered by Zeek, scikit-learn, and a pure-NumPy
> DQN reinforcement-learning agent.

```
 CPS/ICS Traffic
        │
    ┌───▼────────┐   ┌─────────────────┐   ┌──────────────────┐
    │   Zeek     │──►│  Ensemble IDS   │──►│  RL Mitigation   │
    │ DNP3 parser│   │ Rule+Stat+ML    │   │  DQN Agent       │
    └────────────┘   └─────────────────┘   └────────┬─────────┘
                                                      │
                                           ┌──────────▼─────────┐
                                           │  SDN Controller    │
                                           │  (Ryu / Mock)      │
                                           └────────────────────┘
```


## Web Dashboard (GUI)

The dashboard provides a live detection console with real-time charts, a confusion matrix,
alert feed, and simulation controls — all in a single browser tab.

```bash
# Launch the dashboard (auto-opens your browser)
python scripts/serve.py

# Auto-start simulation immediately on load
python scripts/serve.py --autostart --attack-prob 0.40

# Faster simulation (lower delay between flows)
python scripts/serve.py --autostart --flow-delay 0.05

# With pre-trained models
python scripts/serve.py \
    --ml-model data/models/ids_pipeline.joblib \
    --rl-agent data/models/dqn_agent.npz \
    --autostart
```

Then open **http://127.0.0.1:5000** in your browser.

### Dashboard features

| Panel | What it shows |
|---|---|
| **KPI strip** | Flows processed, alerts, TPR, FPR, Precision, F1, raw TP/FP/FN/TN |
| **Alert timeline** | Rolling alert rate + TPR over the last 60 ticks |
| **Attack distribution** | Bar chart: detected attacks by type (scan/replay/injection/…) |
| **Mitigation actions** | Horizontal bar: how often each SDN action was selected |
| **Severity doughnut** | Proportion of alerts by severity (critical/high/medium/low) |
| **Confusion matrix** | Live TP / FP / FN / TN tiles with colour coding |
| **Config sliders** | Tune attack probability, flow delay, warm-up count, seed — live |
| **Alert feed** | Real-time table (newest on top) with src IP, type, severity, action |
| **System log** | Status messages from the engine (training, errors, throughput) |
| **Throughput chart** | Flows/second time series |

### REST API

The dashboard is backed by a plain JSON API — useful for scripting or integration tests:

```bash
GET  /api/status     # Engine state + current config
GET  /api/metrics    # Full metrics snapshot (JSON)
GET  /api/history    # Last 50 alert events
GET  /api/stream     # Server-Sent Events stream
POST /api/start      # Start simulation
POST /api/stop       # Stop simulation
POST /api/reset      # Clear all metrics
POST /api/config     # Update config (JSON body)
```

Example:
```bash
# Change attack probability to 60% while running
curl -X POST http://localhost:5000/api/config \
     -H "Content-Type: application/json" \
     -d '{"attack_prob": 0.6, "flow_delay": 0.05}'

# Snapshot metrics
curl http://localhost:5000/api/metrics | python3 -m json.tool
```

### Architecture: why SSE over WebSockets?

Server-Sent Events (SSE) push data from server → browser over a standard HTTP/1.1 connection.
No extra library is needed on either end — Flask's `Response(stream_with_context(...))` is
sufficient. SSE also auto-reconnects on drop, handles proxy buffering via
`X-Accel-Buffering: no`, and works over HTTPS without configuration changes.
WebSockets would require `flask-socketio` + `eventlet` or `gevent`, adding ~150 MB
of transitive dependencies for no practical benefit at this scale.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Run the demo](#1-run-the-demo)
  - [Generate a dataset](#2-generate-a-dataset)
  - [Train the IDS detector](#3-train-the-ids-detector)
  - [Train the RL agent](#4-train-the-rl-agent)
  - [Use Zeek with real PCAPs](#5-use-zeek-with-real-pcaps)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Layer | What it does |
|---|---|
| **Zeek scripts** | DNP3 protocol parsing, per-flow feature extraction, inline rule alerting |
| **Rule-based IDS** | 9 hand-crafted DNP3 signatures (broadcast write, scan, flood, injection, …) |
| **Statistical IDS** | Welford online z-score + IsolationForest; zero-label anomaly detection |
| **ML IDS** | RandomForest + calibrated probabilities; 6-class attack classification |
| **Ensemble** | Weighted score fusion `[rule=0.40, stat=0.30, ml=0.30]` with confidence escalation |
| **SDN Mitigation** | 5 primitives: MONITOR → RATE_LIMIT → SEGMENT → ISOLATE → BLOCK |
| **Safety constraints** | Critical hosts (master, HMI) can never be fully isolated/blocked |
| **RL Agent** | Pure-NumPy DQN; learns the minimum-impact response policy |
| **GenAI Augmentation** | Gaussian noise, Mixup, VAE, boundary-search; domain-constrained |
| **Testbed/Simulator** | Synthetic DNP3 topology; 5 attack types; Zeek-free operation |

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and module breakdown.

```
src/cps_defender/
├── core/          Config, EventBus, models, logging
├── ids/           Feature extraction + ensemble detector
│   └── detectors/ Rule-based, Statistical, ML
├── sdn/           Controller abstraction + mitigation engine
├── rl/            Gym-like environment + NumPy DQN agent
├── genai/         Data augmentation pipeline
└── testbed/       DNP3 traffic simulator
```

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/anushakhan/cps-ics-defender.git
cd cps-ics-defender
pip install -r requirements.txt

# 2. Run the demo (no pre-trained models needed)
python scripts/demo.py

# 3. Full pipeline: generate → train → demo
make pipeline
```

Expected output:
```
╔══════════════════════════════════════════════════════╗
║       CPS / ICS  DEFENDER  —  Live Demo              ║
╚══════════════════════════════════════════════════════╝

[+] Initialising IDS pipeline …
[+] Generating 200 synthetic flows (mixed) …
──────────────────────────────────────────────────────
  Time      Src              Type                  Sev       Action
──────────────────────────────────────────────────────
  14:23:01  10.0.1.2         flooding              HIGH      RATE_LIMIT
  14:23:01  10.0.1.3         command_injection      CRITICAL  ISOLATE
  14:23:01  10.0.1.1         scan                  MEDIUM    SEGMENT
  ...

  SUMMARY
  ─────────────────────────────────────────
  Flows processed :  200
  TPR (Recall)    :  0.891
  FPR             :  0.043
```

---

## Installation

### Requirements

- Python 3.10 or later
- pip

### Steps

```bash
# Option A: Runtime only
pip install -r requirements.txt

# Option B: Development (includes pytest, coverage)
pip install -r requirements-dev.txt
pip install -e .   # editable install for IDE support
```

### Optional: Zeek (for real PCAP analysis)

Zeek is **not required** to run tests or the demo — the built-in traffic
simulator handles everything. Install Zeek only when analysing real traffic:

```bash
# Ubuntu/Debian
echo 'deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_22.04/ /' \
  | sudo tee /etc/apt/sources.list.d/security:zeek.list
curl -fsSL https://download.opensuse.org/repositories/security:zeek/xUbuntu_22.04/Release.key \
  | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/security_zeek.gpg > /dev/null
sudo apt update && sudo apt install zeek
```

See [`zeek/README.md`](zeek/README.md) for script usage.

---

## Usage

### 1. Run the demo

```bash
# Quickstart — synthetic traffic, no models needed
python scripts/demo.py

# With trained models
python scripts/demo.py \
    --ml-model data/models/ml_detector.joblib \
    --rl-agent data/models/dqn_agent.npz

# Specific attack scenario
python scripts/demo.py --scenario command_injection --n-flows 300

# Streaming mode (simulates live traffic)
python scripts/demo.py --stream --interval 0.05

# From a Zeek log
python scripts/demo.py --zeek-log zeek_logs/dnp3_flows.log
```

### 2. Generate a dataset

```bash
# Default: 5 000 mixed flows, CSV + NPZ
python scripts/generate_dataset.py

# Custom
python scripts/generate_dataset.py \
    --n-flows 10000 \
    --output data/processed/large \
    --augment \
    --seed 123
```

Output files:
- `data/processed/dataset.csv` — human-readable, one row per flow
- `data/processed/dataset.npz` — compressed NumPy arrays for fast loading
- `data/processed/feature_extractor.joblib` — fitted scaler

### 3. Train the IDS detector

```bash
# Auto-generate training data
python scripts/train_detector.py

# From existing dataset
python scripts/train_detector.py \
    --dataset data/processed/dataset.npz \
    --test-split 0.20 \
    --report \
    --n-estimators 300

# Output: data/models/ml_detector.joblib
```

Sample output:
```
[+] Training MLDetector (200 trees) …
    Training time : 3.41s
    Accuracy      : 0.9712
    TPR (Recall)  : 0.9634   FPR: 0.0187

    Top-10 feature importances:
      burst_count                    0.1823  ████████
      pkt_count                      0.1541  ██████
      inter_arrival_mean             0.1302  █████
      ...
```

### 4. Train the RL agent

```bash
# Standard training
python scripts/train_rl.py --episodes 500

# Curriculum learning (recommended for best results)
python scripts/train_rl.py --curriculum --episodes 500

# Resume from checkpoint
python scripts/train_rl.py \
    --load data/models/dqn_agent.npz \
    --episodes 200

# Evaluate only
python scripts/train_rl.py \
    --load data/models/dqn_agent.npz \
    --eval-only --eval-episodes 50
```

### 5. Use Zeek with real PCAPs

```bash
# Analyse a PCAP
zeek -r /path/to/ics.pcap \
     zeek/scripts/dnp3_monitor.zeek \
     zeek/scripts/ics_anomaly.zeek \
     zeek/scripts/log_to_json.zeek

# Feed the Zeek log into the Python pipeline
python scripts/demo.py --zeek-log zeek_logs/dnp3_flows.log
```

### 6. Makefile shortcuts

```bash
make install          # Install dependencies
make test             # Run all tests
make test-cov         # Tests + HTML coverage report
make generate         # Generate 5 000-flow dataset
make train-detector   # Train IDS
make train-rl         # Train RL agent
make demo             # Run end-to-end demo
make pipeline         # generate → train-detector → train-rl → demo
make clean            # Remove build artefacts
```

---

## Configuration

All settings live in `config/config.yaml`. Override any value with an
environment variable using the pattern `CPS_SECTION__KEY`:

```bash
# Example: switch to a real Ryu controller
CPS_SDN__CONTROLLER_TYPE=ryu \
CPS_SDN__RYU__HOST=192.168.1.10 \
python scripts/demo.py
```

Key configuration sections:

```yaml
ids:
  rule_weight: 0.40          # Ensemble weight for rule-based detector
  statistical_weight: 0.30   # Ensemble weight for statistical detector
  ml_weight: 0.30            # Ensemble weight for ML detector
  alert_threshold: 0.45      # Combined score to trigger an alert

sdn:
  controller_type: mock      # "mock" (default) or "ryu"
  mitigation:
    critical_hosts:          # These hosts can never be blocked
      - "10.0.0.1"           # DNP3 master
      - "10.0.0.10"          # HMI
    isolation_duration: 300  # Seconds before auto-expiry of isolation
    cooldown_period: 60      # Min seconds between actions on same host

rl:
  agent:
    learning_rate: 0.001
    gamma: 0.95
    epsilon_start: 1.0
    epsilon_end: 0.05
    hidden_sizes: [128, 64]
```

See [`config/config.yaml`](config/config.yaml) for the full reference and
[`config/scenarios.yaml`](config/scenarios.yaml) for all threat scenario definitions.

---

## Project Structure

```
cps-ics-defender/
│
├── config/
│   ├── config.yaml            Main configuration
│   └── scenarios.yaml         Threat scenario definitions (8 scenarios)
│
├── src/cps_defender/
│   ├── core/
│   │   ├── config.py          YAML + env-var config with deep-merge
│   │   ├── events.py          Thread-safe pub/sub EventBus
│   │   ├── logging_setup.py   Rotating file + console logging
│   │   └── models.py          FlowRecord, Alert, NetworkState dataclasses
│   │
│   ├── ids/
│   │   ├── feature_extractor.py   StandardScaler wrapper, Zeek log parser
│   │   ├── pipeline.py            Ensemble IDS pipeline
│   │   └── detectors/
│   │       ├── rule_based.py      9 DNP3 signature rules
│   │       ├── statistical.py     Welford z-score + IsolationForest
│   │       └── ml_detector.py     RandomForest + CalibratedClassifierCV
│   │
│   ├── sdn/
│   │   ├── controller.py      SDNController ABC, MockController, RyuController
│   │   └── mitigation.py      MitigationEngine with safety constraints
│   │
│   ├── rl/
│   │   ├── environment.py     Gym-like CPSEnvironment (no gym dependency)
│   │   └── agent.py           NumPy MLP + DQN + experience replay
│   │
│   ├── genai/
│   │   └── augmenter.py       Gaussian, Mixup, VAE, Boundary augmenters
│   │
│   └── testbed/
│       └── traffic_sim.py     Synthetic DNP3 topology + 5 attack types
│
├── tests/
│   ├── conftest.py            Shared pytest fixtures
│   ├── test_ids.py            IDS unit tests
│   ├── test_sdn_rl.py         SDN + RL unit tests
│   └── test_integration.py    End-to-end integration tests
│
├── scripts/
│   ├── demo.py                Live detect-and-respond demo
│   ├── generate_dataset.py    Synthetic dataset generation
│   ├── train_detector.py      ML IDS training CLI
│   └── train_rl.py            DQN agent training CLI
│
├── zeek/
│   ├── scripts/
│   │   ├── dnp3_monitor.zeek  DNP3 feature extraction
│   │   ├── ics_anomaly.zeek   Rule engine + NOTICE alerts
│   │   └── log_to_json.zeek   JSON log exporter
│   └── README.md
│
├── docs/
│   └── architecture.md        Full system architecture + design decisions
│
├── data/
│   ├── models/                Trained model artefacts (.gitignored)
│   └── processed/             Generated datasets (.gitignored)
│
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Testing

```bash
# All tests
pytest

# With coverage
pytest --cov=src/cps_defender --cov-report=term-missing

# Individual modules
pytest tests/test_ids.py -v
pytest tests/test_sdn_rl.py -v
pytest tests/test_integration.py -v
```

The test suite covers:

| Module | Tests |
|---|---|
| FeatureExtractor | Fit/transform, shape validation, scaler stats |
| RuleBasedDetector | All 9 rules individually triggered + normal baseline |
| StatisticalDetector | Welford convergence, IsolationForest anomaly |
| MLDetector | Training, prediction, probability calibration, save/load |
| IDSPipeline | Ensemble scoring, threshold, event bus integration |
| MockController | All 5 SDN operations, state introspection |
| MitigationEngine | Safety constraints, cooldown, auto-expiry |
| CPSEnvironment | Step, reset, reward function, episode summary |
| DQNAgent | Act (explore/exploit), observe, batch update, save/load |
| ReplayBuffer | Capacity, sampling, FIFO eviction |
| Integration | Full detect→mitigate cycle, augmentation pipeline, multi-step campaigns |

---

## Design Decisions

### Why pure NumPy for the DQN?
The state space is 8-dimensional and the action space has 5 elements —
network sizes where a full PyTorch or TensorFlow stack (2 GB+) is
completely disproportionate. The NumPy MLP (forward pass + backprop +
gradient clip) is ~80 lines and easy to audit or modify.

### Why no `gym` dependency?
`gymnasium` frequently breaks on version bumps and adds ~150 MB of
transitive dependencies. Our `CPSEnvironment` exposes the same
`reset() / step() → (state, reward, done, info)` interface with zero
overhead.

### Why Welford online statistics?
ICS flows arrive continuously. Welford's algorithm maintains exact
mean and variance in O(1) space with no numerical instability — no
batch accumulation or sliding windows needed.

### Why `CalibratedClassifierCV` around RandomForest?
Raw RF `predict_proba` outputs are over-confident (probability mass
concentrated near 0/1). Isotonic calibration with cross-validation
produces reliable confidence scores, which are essential for the
ensemble weighting to work correctly.

### Why `IsolationForest` for statistical anomaly?
- Scales O(n log n); handles high-dimensional data without distance metrics
- No label requirement: catches zero-day variants
- Integrates cleanly with scikit-learn pipeline

### Why MockController as the default?
Running Mininet + Ryu requires root, specific kernel settings, and
significant setup time. All tests and demos work identically with the
in-memory mock, which records every SDN operation for inspection and
assertion.

### Ensemble weights `[0.40, 0.30, 0.30]`
- Rules have highest weight: DNP3 violation signatures have near-zero
  false-positive rate when correctly matched.
- Statistical and ML share equal weight; statistical is more robust
  early in deployment (before labelled data accumulates), while ML
  dominates once well-trained.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | DQN, VAE, all numerical operations |
| `pandas` | ≥ 2.0 | Dataset I/O (CSV), Zeek log parsing |
| `scikit-learn` | ≥ 1.3 | RandomForest, IsolationForest, StandardScaler, calibration |
| `joblib` | ≥ 1.3 | Model serialisation (used internally by sklearn) |
| `pyyaml` | ≥ 6.0 | Configuration loading |
| `flask` | ≥ 3.0 | Optional status endpoint |

**No PyTorch. No TensorFlow. No gym. No Mininet required.**

---

## Roadmap

- [ ] PPO agent (continuous action parameters, e.g. rate-limit bandwidth)
- [ ] ONOS controller backend
- [ ] Live Ryu integration guide with a Mininet topology script
- [ ] Modbus / EtherNet-IP protocol support
- [ ] STIX/TAXII threat intelligence feed integration
- [ ] Prometheus metrics exporter
- [ ] Docker Compose deployment (Zeek + Python pipeline)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Run tests: `make test`
4. Open a Pull Request

Please ensure all new features include:
- Unit tests (`tests/`)
- Docstrings on public methods
- An entry in `docs/architecture.md` if a new module is added

---

## License

MIT © 2025 Anusha Khan. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this framework in academic work, please cite:

```bibtex
@software{khan2025cpsicsdefender,
  author    = {Anusha Khan},
  title     = {CPS/ICS Defender: Zeek-based IDS + SDN Mitigation for Industrial Control Networks},
  year      = {2025},
  url       = {https://github.com/anushakhan/cps-ics-defender},
}
```
