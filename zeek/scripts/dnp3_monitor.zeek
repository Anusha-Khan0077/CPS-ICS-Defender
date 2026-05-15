##! DNP3 Protocol Monitor for CPS/ICS Defender
##!
##! Monitors DNP3 traffic, extracts features per flow, and emits
##! structured JSON logs consumed by the Python IDS pipeline.
##!
##! Usage: zeek -i <iface> dnp3_monitor.zeek
##!        zeek -r <pcap>  dnp3_monitor.zeek

@load base/protocols/dnp3
@load base/frameworks/notice

module CPS_DNP3;

export {
    ## Log stream identifier
    redef enum Log::ID += { LOG };

    ## Log record written for every DNP3 connection summary
    type Info: record {
        ts:                  time    &log;
        uid:                 string  &log;
        id:                  conn_id &log;

        ## Flow-level statistics
        duration:            interval &log &optional;
        orig_pkts:           count    &log &default=0;
        resp_pkts:           count    &log &default=0;
        orig_bytes:          count    &log &default=0;
        resp_bytes:          count    &log &default=0;

        ## DNP3-specific fields
        function_codes:      vector of count &log &optional;
        unique_fc_count:     count   &log &default=0;
        request_count:       count   &log &default=0;
        response_count:      count   &log &default=0;
        is_broadcast:        bool    &log &default=F;

        ## Timing features
        inter_arrival_mean:  double  &log &default=0.0;
        inter_arrival_std:   double  &log &default=0.0;
        burst_count:         count   &log &default=0;
        error_count:         count   &log &default=0;

        ## Classification hint (populated by ics_anomaly.zeek)
        anomaly_score:       double  &log &default=0.0;
        anomaly_label:       string  &log &default="normal";
    };

    ## Notice type for high-confidence alerts
    redef enum Notice::Type += {
        DNP3_Anomaly,
        DNP3_Flood,
        DNP3_UnknownFunctionCode,
        DNP3_BroadcastWrite,
    };

    ## Broadcast destination address in DNP3
    const BROADCAST_ADDR = 0xFFFF;

    ## Function codes that should never appear in normal ops
    const SUSPICIOUS_FC: set[count] = {
        0x81,  # Authenticate Request (unusual in legacy deployments)
        0x82,  # Authenticate Error
        0x20,  # Enable Unsolicited (recon indicator)
        0x21,  # Disable Unsolicited
    };

    ## High-rate threshold: packets per second to flag as flood
    const FLOOD_PPS_THRESHOLD = 100.0;
}

# Per-connection state accumulator
type ConnState: record {
    start_ts:      time;
    last_pkt_ts:   time;
    pkt_times:     vector of time;
    fc_seen:       table[count] of count;
    request_count: count &default=0;
    response_count:count &default=0;
    error_count:   count &default=0;
    burst_count:   count &default=0;
    last_burst_ts: time &optional;
    is_broadcast:  bool &default=F;
};

global conn_states: table[string] of ConnState;
global log_info:    table[string] of Info;

event zeek_init() {
    Log::create_stream(CPS_DNP3::LOG, [$columns=Info, $path="dnp3_flows"]);
    print "CPS DNP3 Monitor initialized.";
}

# Helper: compute mean and population std-dev from a vector of times
function time_stats(times: vector of time): record { mean: double; std: double; } {
    if (|times| < 2)
        return [$mean=0.0, $std=0.0];

    local intervals: vector of double = vector();
    local i = 0;
    while (i < |times| - 1) {
        intervals[|intervals|] = interval_to_double(times[i+1] - times[i]);
        ++i;
    }

    local sum = 0.0;
    for (v in intervals) sum += intervals[v];
    local mean = sum / |intervals|;

    local sq_sum = 0.0;
    for (v in intervals) sq_sum += (intervals[v] - mean) ^ 2;
    local std = sqrt(sq_sum / |intervals|);

    return [$mean=mean, $std=std];
}

event connection_established(c: connection) {
    local uid = c$uid;
    conn_states[uid] = ConnState(
        $start_ts=network_time(),
        $last_pkt_ts=network_time(),
        $pkt_times=vector(),
        $fc_seen=table()
    );

    local info = Info(
        $ts=network_time(),
        $uid=uid,
        $id=c$id
    );
    log_info[uid] = info;
}

event dnp3_application_request_header(c: connection, is_orig: bool,
                                       application: count, fc: count) {
    if (c$uid !in conn_states) return;
    local st = conn_states[c$uid];
    local now = network_time();

    # Track function codes
    if (fc !in st$fc_seen)
        st$fc_seen[fc] = 0;
    st$fc_seen[fc] += 1;

    if (is_orig)
        st$request_count += 1;
    else
        st$response_count += 1;

    # Detect broadcast destination (DNP3 dest address 0xFFFF)
    if (c$id$resp_p == 20000/tcp) {  # DNP3 port; broadcast detected by addr
        if (c$id$resp_h == 255.255.255.255)
            st$is_broadcast = T;
    }

    # Track arrival times for burst/timing analysis
    st$pkt_times += now;

    # Burst detection: >5 pkts within 100ms window
    if (|st$pkt_times| >= 2) {
        local recent_delta = interval_to_double(now - st$last_pkt_ts);
        if (recent_delta < 0.1) {  # 100 ms
            st$burst_count += 1;
        }
    }
    st$last_pkt_ts = now;

    # Alert on suspicious function codes
    if (fc in SUSPICIOUS_FC) {
        NOTICE([$note=DNP3_UnknownFunctionCode,
                $conn=c,
                $msg=fmt("Suspicious DNP3 function code 0x%02x from %s", fc, c$id$orig_h),
                $identifier=fmt("%s-fc-%d", c$uid, fc)]);
    }

    # Alert on broadcast writes (FC=2 or FC=3 to broadcast)
    if ((fc == 0x02 || fc == 0x03) && st$is_broadcast) {
        NOTICE([$note=DNP3_BroadcastWrite,
                $conn=c,
                $msg=fmt("DNP3 broadcast WRITE from %s", c$id$orig_h),
                $identifier=fmt("%s-bcast", c$uid)]);
    }
}

event dnp3_debug(c: connection, is_orig: bool, msg: string) {
    # Capture internal Zeek DNP3 parse errors as error count
    if (c$uid in conn_states)
        conn_states[c$uid]$error_count += 1;
}

event connection_state_remove(c: connection) {
    local uid = c$uid;
    if (uid !in conn_states || uid !in log_info) return;

    local st  = conn_states[uid];
    local info = log_info[uid];

    # Compute duration
    local dur = network_time() - st$start_ts;
    info$duration = dur;

    # Packet/byte counts from conn record
    if (c?$orig) {
        info$orig_pkts  = c$orig$num_pkts;
        info$orig_bytes = c$orig$num_bytes_ip;
    }
    if (c?$resp) {
        info$resp_pkts  = c$resp$num_pkts;
        info$resp_bytes = c$resp$num_bytes_ip;
    }

    # DNP3-specific fields
    local fc_vec: vector of count = vector();
    for (fc in st$fc_seen) fc_vec += fc;
    info$function_codes    = fc_vec;
    info$unique_fc_count   = |st$fc_seen|;
    info$request_count     = st$request_count;
    info$response_count    = st$response_count;
    info$is_broadcast      = st$is_broadcast;
    info$burst_count       = st$burst_count;
    info$error_count       = st$error_count;

    # Timing statistics
    local stats = time_stats(st$pkt_times);
    info$inter_arrival_mean = stats$mean;
    info$inter_arrival_std  = stats$std;

    # Flood detection
    local total_pkts = info$orig_pkts + info$resp_pkts;
    local dur_secs   = interval_to_double(dur);
    if (dur_secs > 0.0 && total_pkts / dur_secs > FLOOD_PPS_THRESHOLD) {
        NOTICE([$note=DNP3_Flood,
                $conn=c,
                $msg=fmt("DNP3 flood: %.1f pps from %s", total_pkts/dur_secs, c$id$orig_h),
                $identifier=fmt("%s-flood", uid)]);
    }

    Log::write(CPS_DNP3::LOG, info);

    delete conn_states[uid];
    delete log_info[uid];
}
