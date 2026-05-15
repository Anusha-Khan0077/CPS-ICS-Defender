# Zeek Scripts — CPS/ICS Defender

This directory contains Zeek scripts for DNP3 protocol monitoring and
inline rule-based anomaly detection. They are **optional**: the Python
IDS pipeline works with any JSON flow log or the built-in traffic
simulator. Zeek scripts are needed only when analysing real PCAPs or
live interfaces.

## Scripts

| Script | Purpose |
|---|---|
| `dnp3_monitor.zeek` | Per-connection DNP3 feature extraction; writes `dnp3_flows.log` |
| `ics_anomaly.zeek` | Rule engine overlay; writes `ics_anomaly.log` and NOTICE alerts |
| `log_to_json.zeek` | Switches all log output to JSON; required for Python ingestion |

## Requirements

- Zeek ≥ 5.0 (tested on 5.2)
- Built-in DNP3 analyzer (`base/protocols/dnp3`)

Install on Ubuntu/Debian:
```bash
echo 'deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_22.04/ /' \
  | sudo tee /etc/apt/sources.list.d/security:zeek.list
curl -fsSL https://download.opensuse.org/repositories/security:zeek/xUbuntu_22.04/Release.key \
  | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/security_zeek.gpg > /dev/null
sudo apt update && sudo apt install zeek
```

## Usage

### Analyse a PCAP file
```bash
zeek -r /path/to/ics_traffic.pcap \
     zeek/scripts/dnp3_monitor.zeek \
     zeek/scripts/ics_anomaly.zeek \
     zeek/scripts/log_to_json.zeek
```

Logs are written to `./zeek_logs/`:
- `dnp3_flows.log` — one JSON object per DNP3 connection
- `ics_anomaly.log` — anomaly events with rule hits and scores
- `notice.log` — high-confidence NOTICE alerts

### Live capture
```bash
sudo zeek -i eth0 \
     zeek/scripts/dnp3_monitor.zeek \
     zeek/scripts/ics_anomaly.zeek \
     zeek/scripts/log_to_json.zeek
```

### Feed logs into the Python pipeline
```bash
python scripts/demo.py --zeek-log zeek_logs/dnp3_flows.log
```

## Log Schema (`dnp3_flows.log`)

| Field | Type | Description |
|---|---|---|
| `ts` | double | Unix epoch timestamp |
| `uid` | string | Unique connection ID |
| `id.orig_h` | addr | Source IP |
| `id.resp_h` | addr | Destination IP |
| `id.orig_p` | port | Source port |
| `id.resp_p` | port | Destination port |
| `duration` | double | Flow duration (seconds) |
| `orig_pkts` | count | Originator packet count |
| `resp_pkts` | count | Responder packet count |
| `orig_bytes` | count | Originator bytes |
| `resp_bytes` | count | Responder bytes |
| `function_codes` | array | DNP3 function codes observed |
| `unique_fc_count` | count | Number of distinct function codes |
| `request_count` | count | DNP3 request messages |
| `response_count` | count | DNP3 response messages |
| `is_broadcast` | bool | Broadcast destination flag |
| `inter_arrival_mean` | double | Mean inter-packet arrival (s) |
| `inter_arrival_std` | double | Std-dev inter-packet arrival (s) |
| `burst_count` | count | Sub-100ms consecutive packet pairs |
| `error_count` | count | DNP3 parse errors in flow |
| `anomaly_score` | double | Rule-based anomaly score 0–1 |
| `anomaly_label` | string | normal / low / medium / high / critical |
