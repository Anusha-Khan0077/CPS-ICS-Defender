# Architecture — CPS/ICS Defender

## Overview

CPS/ICS Defender is a closed-loop, programmable intrusion detection and
response framework for industrial control system (ICS) networks. It combines
three layers of defence:

1. **Protocol-aware IDS** — Zeek-based feature extraction + ensemble detection
2. **SDN Mitigation Layer** — programmatic network response via controller API
3. **Adaptive RL Policy** — reinforcement-learning agent that selects the
   minimum-impact response given the current network state

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CPS/ICS Network                               │
│  ┌─────────┐  DNP3   ┌────────────┐  DNP3   ┌──────────────────┐    │
│  │  Master │◄───────►│ Outstations│◄───────►│   HMI / SCADA    │    │
│  └────┬────┘         └─────┬──────┘         └────────┬─────────┘    │
│       │                    │                          │              │
│       └────────────────────┼──────────────────────────┘              │
│                            │ mirrored / tapped                        │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Zeek Engine   │  (optional live/PCAP analysis)
                    │ dnp3_monitor    │
                    │ ics_anomaly     │
                    │ log_to_json     │
                    └────────┬────────┘
                             │  JSON flow logs / FlowRecord objects
                    ┌────────▼────────────────────────────────────────┐
                    │               IDS Pipeline                       │
                    │  ┌─────────────┐ ┌──────────────┐ ┌──────────┐ │
                    │  │  Rule-Based │ │  Statistical  │ │    ML    │ │
                    │  │  Detector   │ │  Detector     │ │ Detector │ │
                    │  │  (9 rules)  │ │  Welford+IF   │ │  RF+Cal  │ │
                    │  └──────┬──────┘ └──────┬───────┘ └────┬─────┘ │
                    │         └───────────┬────┘              │        │
                    │               ┌─────▼──────┐            │        │
                    │               │  Ensemble  │◄───────────┘        │
                    │               │ Aggregator │ w=[0.4, 0.3, 0.3]  │
                    │               └─────┬──────┘                     │
                    └─────────────────────┼──────────────────────────-─┘
                                          │ Alert (+ confidence)
                    ┌─────────────────────▼──────────────────────────┐
                    │           Adaptive Response Layer               │
                    │  ┌──────────────────────────────────────────┐  │
                    │  │              DQN RL Agent                 │  │
                    │  │  State: [alert_count, attack_rate, qos,  │  │
                    │  │          isolated_hosts, confidence, …]   │  │
                    │  │  Action: {MONITOR, RATE_LIMIT, SEGMENT,  │  │
                    │  │           ISOLATE, BLOCK}                 │  │
                    │  └──────────────────────────────────────────┘  │
                    │                    │                            │
                    │  ┌─────────────────▼─────────────────────────┐ │
                    │  │            Mitigation Engine               │ │
                    │  │  Safety constraints, cooldown, audit log  │ │
                    │  └─────────────────┬─────────────────────────┘ │
                    └────────────────────┼───────────────────────────┘
                                         │ REST / in-process
                    ┌────────────────────▼───────────────────────────┐
                    │          SDN Controller (Ryu / Mock)            │
                    │  Flow rules, ACLs, rate-limit queues            │
                    └────────────────────────────────────────────────┘
```

---

## Module Breakdown

### `cps_defender.core`

| Module | Role |
|---|---|
| `config.py` | YAML + env-var config, singleton `get_config()` |
| `events.py` | Thread-safe pub/sub `EventBus`, `EventType` enum |
| `logging_setup.py` | Rotating file + coloured console handler |
| `models.py` | `FlowRecord`, `Alert`, `NetworkState` dataclasses; `FEATURE_NAMES` |

### `cps_defender.ids`

#### Feature extraction (`feature_extractor.py`)
Wraps `sklearn.preprocessing.StandardScaler`. The `zeek_logs_to_flows()`
function parses Zeek JSON logs into `FlowRecord` objects. 16 features are
extracted per flow:

```
flow_duration, pkt_count, byte_count, bytes_per_pkt,
src_port, dst_port, protocol_id, function_code,
unique_fc_count, req_resp_ratio,
inter_arrival_mean, inter_arrival_std,
is_broadcast, direction, burst_count, error_rate
```

#### Detectors

| Detector | Algorithm | Why |
|---|---|---|
| `RuleBasedDetector` | 9 hand-crafted DNP3 rules | Zero false-negatives on known signatures; interpretable |
| `StatisticalDetector` | Welford online mean/var + IsolationForest | Catches novel deviations without labels; O(1) memory |
| `MLDetector` | RandomForest + CalibratedClassifierCV | Multi-class (6 labels); reliable probabilities for ensemble |

#### Ensemble (`pipeline.py`)
Scores are linearly combined with weights `[rule=0.4, stat=0.3, ml=0.3]`.
Confidence escalation: `score > 0.45 → alert`, severity mapped from
confidence deciles. Alert events published to the `EventBus`.

### `cps_defender.sdn`

#### `SDNController` (ABC)
- `MockController` — pure Python in-memory; no network required; full
  state introspection for tests.
- `RyuController` — REST client to a live Ryu OFREST API (`GET/POST`
  to `/stats/flow/<dpid>` etc.).

#### `MitigationEngine`
Five primitives in order of increasing impact:

| Action | Impact | Implementation |
|---|---|---|
| `MONITOR` | None | Increase logging verbosity |
| `RATE_LIMIT` | Low | 1 Mbps QoS queue |
| `SEGMENT` | Medium | VLAN micro-segmentation |
| `ISOLATE` | High | Drop all but essential control traffic; auto-expires in 5 min |
| `BLOCK` | Critical | Full deny-all ACL (blocked for critical hosts by safety check) |

Safety constraints prevent isolating or blocking hosts on the
`critical_hosts` list (master, HMI).

### `cps_defender.rl`

#### `CPSEnvironment` (Gym-like, no gym dep)
- **State** (8-dim): alert_count, attack_rate, FP_rate, QoS_score,
  isolated_hosts, active_mitigations, time_since_last_alert, confidence
- **Actions**: 5 (one per mitigation primitive)
- **Reward**:
  - `+2.0` true-positive mitigation
  - `−2.0` missed attack (false-negative)
  - `−1.0` unnecessary action (false-positive)
  - `+1.0` correctly doing nothing
  - `−0.5 × QoS_penalty` for over-aggressive isolation

#### `DQNAgent`
Pure-NumPy implementation. Architecture:

```
Input(8) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(5)
```

- He weight initialisation
- Experience replay buffer (FIFO deque, 10 000 transitions)
- Soft target-network sync every 100 steps
- ε-greedy with exponential decay (1.0 → 0.05)
- Gradient clipping (‖g‖∞ < 1.0)

### `cps_defender.genai`

Four augmentation strategies, all domain-constrained via `_clip_and_round()`:

| Strategy | Method | Good for |
|---|---|---|
| `GaussianAugmenter` | Per-feature Gaussian noise | General oversampling |
| `MixupAugmenter` | Convex combination of same-class pairs | Smoothing decision boundary |
| `VAEAugmenter` | Numpy VAE with ELBO loss | Learning latent attack manifold |
| `BoundaryAugmenter` | Coordinate-wise evasion search | Hardening near-boundary samples |

`AugmentationPipeline` applies all four and up-samples to `target_per_class`
samples using minority-class priority.

### `cps_defender.testbed`

`TrafficSimulator` generates realistic DNP3 topology flows:
- **Normal**: polling, unsolicited responses, time-sync, HMI reads
- **Attacks**: scan, replay, command_injection, flooding, mitm
- Configurable `attack_ratio`; supports per-scenario generation

---

## Data Flow

```
TrafficSimulator.generate()
        │
        ▼  List[FlowRecord]
FeatureExtractor.transform()
        │
        ▼  np.ndarray (N, 16)
IDSPipeline.process_flow(flow)
        │
        ▼  Optional[Alert]
MitigationEngine.apply(alert, action)
        │
        ▼  MitigationResult
SDNController.{rate_limit,isolate,block,…}(host)
```

---

## Design Decisions

### Why pure NumPy for DQN?
State and action spaces are tiny (8-dim state, 5 actions). A full PyTorch
dependency (~2 GB) would be disproportionate. The NumPy MLP forward/backward
pass is < 100 lines and fully transparent for researchers.

### Why Welford online statistics?
ICS flows are processed in real-time. Welford's algorithm maintains exact
mean and variance in O(1) space and is numerically stable — no batch
accumulation needed.

### Why CalibratedClassifierCV?
RandomForest probability outputs are poorly calibrated out of the box
(over-confident). Isotonic calibration with cross-validation produces
reliable confidence scores for the ensemble weighting step.

### Why IsolationForest for statistical anomaly?
It scales O(n log n), handles multi-dimensional anomalies without distance
metrics, and integrates cleanly with sklearn's `fit/predict` API.

### Why MockController as default?
Running Mininet + Ryu requires root, a specific kernel version, and
considerable setup time. All tests and demos work identically with the
in-memory mock, which records every SDN operation for inspection.

### Ensemble weight choice [0.4, 0.3, 0.3]
- Rules get highest weight because DNP3 violation signatures have near-zero
  false-positive rate when correctly matched.
- ML and statistical get equal weight; statistical is more robust early in
  deployment (before labelled data), ML dominates once trained.

---

## Extending the Framework

### Add a new detection rule
Edit `src/cps_defender/ids/detectors/rule_based.py`, add a method
`_rule_myname(self, flow) -> float` and register it in `self._rules`.

### Add a new SDN controller backend
Subclass `SDNController` in `src/cps_defender/sdn/controller.py`, implement
the abstract methods, and add the type string to `create_controller()`.

### Add a new attack class
1. Add the label to `AttackType` in `models.py`
2. Add a generator method to `TrafficSimulator`
3. Update `INV_LABEL_MAP` in `models.py`

### Swap RL algorithm
Replace `DQNAgent` with any class exposing `.act(state)` and
`.observe(s, a, r, s2, done)`. The `CPSEnvironment` is algorithm-agnostic.
