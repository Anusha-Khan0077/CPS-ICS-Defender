##! ICS Anomaly Rule Engine — overlays rule-based anomaly scores
##! on top of DNP3 flow records written by dnp3_monitor.zeek.
##!
##! Load after dnp3_monitor.zeek:
##!   zeek -r <pcap> dnp3_monitor.zeek ics_anomaly.zeek

@load ./dnp3_monitor

module ICS_Anomaly;

export {
    ## Rule hit record appended to the DNP3 flow log
    type RuleHit: record {
        rule_id:     string;
        severity:    string;   # low / medium / high / critical
        description: string;
        score:       double;   # 0-1 contribution
    };

    ## Per-flow rule evaluation result written to ics_anomaly.log
    type AnomalyRecord: record {
        ts:          time    &log;
        uid:         string  &log;
        src:         addr    &log;
        dst:         addr    &log;
        rules_fired: vector of string &log &optional;
        total_score: double  &log &default=0.0;
        severity:    string  &log &default="normal";
    };

    redef enum Log::ID += { ANOMALY_LOG };
}

# Thresholds (tunable via redef)
const HIGH_RATE_PPS     = 50.0  &redef;
const MAX_UNIQUE_FC     = 6     &redef;
const ERROR_RATE_THRESH = 0.10  &redef;
const REQ_RESP_SKEW     = 3.0   &redef;   # request/response ratio threshold

# Function codes considered config-altering
const CONFIG_FC: set[count] = { 0x02, 0x03, 0x04, 0x07, 0x0D, 0x12, 0x13 };

event zeek_init() {
    Log::create_stream(ICS_Anomaly::ANOMALY_LOG,
                       [$columns=AnomalyRecord, $path="ics_anomaly"]);
}

## Evaluate all rules against a completed DNP3 flow info record.
## Returns a list of RuleHit values and cumulative score.
function evaluate_rules(info: CPS_DNP3::Info): vector of RuleHit {
    local hits: vector of RuleHit = vector();

    # R1: Broadcast WRITE
    if (info$is_broadcast) {
        hits += RuleHit($rule_id="broadcast_write",
                        $severity="critical",
                        $description="DNP3 WRITE to broadcast address",
                        $score=0.95);
    }

    # R2: Excessive unique function codes (recon indicator)
    if (info$unique_fc_count > MAX_UNIQUE_FC) {
        hits += RuleHit($rule_id="high_fc_diversity",
                        $severity="medium",
                        $description=fmt("%d unique function codes (max %d)",
                                         info$unique_fc_count, MAX_UNIQUE_FC),
                        $score=0.55);
    }

    # R3: High-rate flood
    if (info?$duration) {
        local dur = interval_to_double(info$duration);
        local total_pkts = info$orig_pkts + info$resp_pkts;
        if (dur > 0.0 && total_pkts / dur > HIGH_RATE_PPS) {
            hits += RuleHit($rule_id="flooding",
                            $severity="high",
                            $description=fmt("%.1f pps exceeds threshold %.1f",
                                              total_pkts/dur, HIGH_RATE_PPS),
                            $score=0.80);
        }
    }

    # R4: High error rate
    local total_pkts2 = info$orig_pkts + info$resp_pkts;
    if (total_pkts2 > 10 && info$error_count > 0) {
        local err_rate = info$error_count * 1.0 / total_pkts2;
        if (err_rate > ERROR_RATE_THRESH) {
            hits += RuleHit($rule_id="error_storm",
                            $severity="medium",
                            $description=fmt("Error rate %.1f%%", err_rate*100),
                            $score=0.60);
        }
    }

    # R5: Request/response imbalance (replay or scan)
    if (info$response_count > 0) {
        local rr_ratio = info$request_count * 1.0 / info$response_count;
        if (rr_ratio > REQ_RESP_SKEW || rr_ratio < 1.0 / REQ_RESP_SKEW) {
            hits += RuleHit($rule_id="req_resp_imbalance",
                            $severity="low",
                            $description=fmt("Request/response ratio %.2f", rr_ratio),
                            $score=0.40);
        }
    }

    # R6: Burst activity
    if (info$burst_count > 10) {
        hits += RuleHit($rule_id="burst_traffic",
                        $severity="medium",
                        $description=fmt("%d burst intervals detected", info$burst_count),
                        $score=0.50);
    }

    # R7: Very low inter-arrival mean (sub-millisecond) = machine-speed flood
    if (info$inter_arrival_mean > 0.0 && info$inter_arrival_mean < 0.001) {
        hits += RuleHit($rule_id="machine_speed_pacing",
                        $severity="high",
                        $description=fmt("Inter-arrival mean %.4fms", info$inter_arrival_mean*1000),
                        $score=0.75);
    }

    return hits;
}

## Map cumulative score to a human-readable severity label
function score_to_severity(score: double): string {
    if (score >= 0.80) return "critical";
    if (score >= 0.60) return "high";
    if (score >= 0.40) return "medium";
    if (score >= 0.20) return "low";
    return "normal";
}

# Hook into the DNP3 log write to append anomaly evaluation
hook Log::log_stream_policy(rec: any, id: Log::ID) {
    if (id != CPS_DNP3::LOG) return;
    local info = (rec as CPS_DNP3::Info);

    local hits = evaluate_rules(info);
    if (|hits| == 0) return;

    local total_score = 0.0;
    local rule_ids: vector of string = vector();
    for (h in hits) {
        total_score += hits[h]$score;
        rule_ids += hits[h]$rule_id;
    }
    # Cap at 1.0
    if (total_score > 1.0) total_score = 1.0;

    local arec = AnomalyRecord(
        $ts=info$ts,
        $uid=info$uid,
        $src=info$id$orig_h,
        $dst=info$id$resp_h,
        $rules_fired=rule_ids,
        $total_score=total_score,
        $severity=score_to_severity(total_score)
    );

    # Back-annotate the DNP3 flow record
    info$anomaly_score = total_score;
    info$anomaly_label = score_to_severity(total_score);

    Log::write(ICS_Anomaly::ANOMALY_LOG, arec);

    if (total_score >= 0.60) {
        NOTICE([$note=CPS_DNP3::DNP3_Anomaly,
                $src=info$id$orig_h,
                $dst=info$id$resp_h,
                $msg=fmt("ICS Anomaly score=%.2f rules=%s", total_score,
                          cat(rule_ids)),
                $identifier=info$uid]);
    }
}
