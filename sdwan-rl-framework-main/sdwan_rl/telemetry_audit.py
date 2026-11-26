"""Absolute timing, age and primary QoS summaries; no line-rate claims."""
import argparse
import json
import math
from dataclasses import asdict
from statistics import fmean, stdev

from .telemetry_core import TelemetryConfig, observation_index, observation_age


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid nonnegative measurement")
    def quantile(q):
        position = q * (len(values) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (position - lo) * (values[hi] - values[lo])
    return {"n": len(values), "mean": fmean(values), "p50": quantile(.5),
            "p95": quantile(.95), "max": values[-1]}


def primary_summary(measurements):
    """Use sent/received counts and echo-count-weighted RTT, not pooled tick means."""
    measurements = list(measurements)
    if not measurements:
        raise ValueError("No primary measurement windows")
    streams = [m["streams"]["primary"] for m in measurements]
    sent = sum(s["sent_packets"] for s in streams)
    received = sum(s["received_packets"] for s in streams)
    echoes = sum(s["echo_received_packets"] for s in streams)
    seconds = math.fsum(m["window_end_s"] - m["window_start_s"] for m in measurements)
    if sent <= 0 or not 0 <= echoes <= received <= sent or seconds <= 0:
        raise ValueError("Invalid primary counts or duration")
    rtt = math.fsum(s["rtt_ms"] * s["echo_received_packets"] for s in streams
                    if s["echo_received_packets"] > 0) / echoes if echoes else None
    probe_bytes = sum(m["probe_payload_bytes_both_directions"] for m in measurements)
    return {"windows": len(streams), "primary_sent_packets": sent,
            "primary_received_packets": received, "primary_echo_received_packets": echoes,
            "primary_loss_pct": 100.0 * (sent - received) / sent,
            "primary_rtt_ms": rtt, "primary_rtt_echo_coverage": echoes / sent,
            "primary_delivered_payload_mbps": 8 * sum(s["received_payload_bytes"] for s in streams) / (1e6 * seconds),
            "probe_payload_bytes_both_directions": probe_bytes,
            "probe_payload_mbps_per_measurement_support": 8 * probe_bytes / (1e6 * seconds),
            "probe_ovs_ingress_match_bytes": sum(
                flows["forward_ingress"]["bytes"] + flows["reverse_ingress"]["bytes"]
                for m in measurements for flows in m["flow_evidence"]["probes"].values())}


def read_timed_trace(path):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < 3 or rows[0].get("kind") != "run" or rows[-1].get("kind") != "complete":
        raise ValueError(f"{path}: incomplete or failed trace")
    header, footer = rows[0], rows[-1]
    if header.get("telemetry_schema") != 1:
        raise ValueError("Trace predates revised telemetry")
    cfg = TelemetryConfig(**header["telemetry_config"])
    ticks = header["experiment_config"]["ticks_per_episode"]
    episodes = header["episode_count"]
    records = rows[1:-1]
    if len(records) != episodes * ticks:
        raise ValueError("Trace budget mismatch")
    keys = set(records[0]["phase_timing_s"])
    for i, row in enumerate(records):
        ep, tick = i // ticks + 1, i % ticks
        if row.get("kind") != "decision" or row["episode"] != ep or row["tick"] != tick:
            raise ValueError("Unordered or missing tick")
        source_tick = observation_index(tick, cfg.observation_lag_ticks)
        if source_tick != row.get("decision_source_tick"):
            raise ValueError("Incorrect source tick")
        if source_tick is not None:
            source = records[(ep - 1) * ticks + source_tick]
            if (row["decision_metrics"] != source["measurement_metrics"]
                    or row["decision_age"] != observation_age(source["measurement_windows"], row["decision_time_s"])):
                raise ValueError("Incorrect telemetry source/age")
        elif row["decision_age"] is not None or row["decision_metrics"] is not None:
            raise ValueError("Unexpected cold-start observation")
        if (set(row["phase_timing_s"]) != keys
                or not math.isclose(sum(row["phase_timing_s"].values()), row["tick_work_s"], abs_tol=1e-6)
                or not row["tick_start_s"] <= row["decision_time_s"] <= row["decision_acknowledged_s"]
                <= row["telemetry"]["window_start_s"] < row["telemetry"]["window_end_s"]):
            raise ValueError("Invalid phase timing")
        if row["controller_ack"]["action"] != row["action"] or row["controller_ack"]["barrier_acknowledged"] is not True:
            raise ValueError("Invalid acknowledgement")
        if i and not math.isclose(row["tick_start_s"] - records[i-1]["tick_start_s"],
                                  row["tick_start_interval_s"], abs_tol=1e-6):
            raise ValueError("Invalid inter-tick interval")
    return header, footer, records


def timing_report(path):
    header, footer, rows = read_timed_trace(path)
    phases = {key: distribution(row["phase_timing_s"][key] for row in rows)
              for key in rows[0]["phase_timing_s"]}
    phases["trace_write_s"] = distribution([r["previous_trace_write_s"] for r in rows[1:]]
                                          + [footer["last_trace_write_s"]])
    phases["tick_work_s"] = distribution(r["tick_work_s"] for r in rows)
    phases["inter_tick_start_s"] = distribution(r["tick_start_interval_s"] for r in rows)
    phases["within_episode_inter_tick_start_s"] = distribution(r["tick_start_interval_s"] for r in rows if r["tick"] != 0)
    phases["decision_to_ack_s"] = distribution(r["decision_acknowledged_s"] - r["decision_time_s"] for r in rows)
    phases["episode_setup_s"] = distribution(r["episode_setup"]["elapsed_s"] for r in rows if r["episode_setup"])
    ages = [r["decision_age"] for r in rows if r["decision_age"] is not None]
    return {"path": str(path), "agent": header["agent_type"], "mode": header["mode"],
            "seed": header["run_seed"], "protocol": header["protocol"],
            "telemetry_config": header["telemetry_config"], "timing_s": phases,
            "age_s": {k: distribution(a[k] for a in ages) for k in ages[0]} if ages else {},
            "primary_and_probe": primary_summary(r["telemetry"] for r in rows),
            "checkpoint_elapsed_s": footer["checkpoint_elapsed_s"], "run_elapsed_s": footer["run_elapsed_s"],
            "interpretation": "Periodic control-loop measurements, not production line-rate proof."}


def age_study(paths, seeds, scenarios, lags=(0, 1, 2)):
    """Matched frozen-policy lag experiment, same checkpoint and measurement setup."""
    from .benchmark_audit import audit_trace
    seeds, scenarios, lags = list(seeds), list(scenarios), list(lags)
    if (not seeds or not scenarios or not lags or 0 not in lags
            or any(len(set(x)) != len(x) for x in (seeds, scenarios, lags))
            or any(type(v) is not int or v < 0 for v in lags)):
        raise ValueError("Specify unique seeds, scenarios and nonnegative lags including zero")
    expected = {(s, e, l) for s in seeds for e in scenarios for l in lags}
    startup = max(lags) + 1
    observed, models = {}, {}
    reference = None
    for path in paths:
        audit = audit_trace(path)
        header, _, records = read_timed_trace(path)
        if audit["agent"] != "dqn" or header["mode"] != "frozen":
            raise ValueError("Age study requires formal frozen DQN test traces")
        cfg = TelemetryConfig(**header["telemetry_config"])
        key = (audit["seed"], audit["scenario_id"], cfg.observation_lag_ticks)
        if key not in expected or key in observed:
            raise ValueError("Unexpected or duplicate age study result")
        signature = (audit["manifest_sha256"], audit["rl_config"], cfg.measurement_settings())
        if reference is None:
            reference = signature
        elif reference != signature:
            raise ValueError("Age study measurement/agent configuration differs")
        model = (audit["model_weights_sha256"], audit["training_protocol"])
        if models.setdefault(key[0], model) != model:
            raise ValueError("Frozen checkpoint differs within replicate")
        steady = [r for r in records if r["tick"] >= startup]
        if not steady:
            raise ValueError("No common steady-state ticks after lag warm-up")
        observed[key] = {"seed": key[0], "scenario": key[1], "lag": key[2],
                         "mean_logged_reward": fmean(r["reward"] for r in steady),
                         "oldest_completion_age_s": distribution(r["decision_age"]["oldest_completion_age_s"] for r in steady),
                         **primary_summary(r["telemetry"] for r in steady)}
    if set(observed) != expected:
        raise ValueError("Incomplete declared age study matrix")
    summaries = []
    for scenario in scenarios:
        for lag in lags:
            rows = [observed[(seed, scenario, lag)] for seed in seeds]
            metrics = {}
            for k in ("mean_logged_reward", "primary_loss_pct", "primary_rtt_ms", "primary_delivered_payload_mbps"):
                vals = [r[k] for r in rows if r[k] is not None]
                metrics[k] = {"n": len(vals), "mean": fmean(vals) if vals else None,
                              "sample_sd": stdev(vals) if len(vals) > 1 else None}
            summaries.append({"scenario": scenario, "lag": lag,
                              "mean_actual_oldest_completion_age_s": fmean(r["oldest_completion_age_s"]["mean"] for r in rows),
                              "metrics": metrics})
    return {"common_startup_ticks_excluded": startup, "runs": list(observed.values()),
            "summaries": summaries, "interpretation": "Artificial lag stress test; not a convergence or causal guarantee."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    timing = sub.add_parser("timing")
    timing.add_argument("--traces", nargs="+", required=True)
    age = sub.add_parser("age")
    age.add_argument("--traces", nargs="+", required=True)
    age.add_argument("--seeds", nargs="+", type=int, required=True)
    age.add_argument("--scenarios", nargs="+", required=True)
    age.add_argument("--lags", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    result = ([timing_report(p) for p in args.traces] if args.command == "timing"
              else age_study(args.traces, args.seeds, args.scenarios, args.lags))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
