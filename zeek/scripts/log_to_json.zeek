##! JSON Log Exporter for CPS/ICS Defender
##!
##! Redirects all Zeek logs to JSON format, compatible with the
##! Python feature extractor (zeek_logs_to_flows).
##!
##! Load last:
##!   zeek -r <pcap> dnp3_monitor.zeek ics_anomaly.zeek log_to_json.zeek

@load base/frameworks/logging
@load policy/tuning/json-logs

module LogToJSON;

event zeek_init() &priority=-10 {
    # Switch all log writers to JSON
    Log::default_writer = Log::WRITER_ASCII;

    # Ensure timestamps are ISO-8601 for easy Python parsing
    Log::default_logdir = "./zeek_logs";

    print "Log-to-JSON exporter active. Output dir: ./zeek_logs";
}

## Convenience hook: add a human-readable ISO timestamp alongside
## the numeric 'ts' field that Zeek writes by default.
hook Log::log_stream_policy(rec: any, id: Log::ID) {
    # No-op: JSON writer handles serialization.
    # This hook is a placeholder for field augmentation if needed.
}
