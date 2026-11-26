"""Fixed-route paired probe-on/off calibration, independent of RL decisions."""
import argparse
import json
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, stdev

from .experiment_protocol import read_manifest, resolve_run, digest
from .policy_audit import append_trace, validate_run_seed
from .telemetry_core import TelemetryConfig
from .telemetry_cli import add_telemetry_arguments, telemetry_kwargs
from .telemetry_audit import primary_summary
from .statistical_core import describe, equivalence, finite


EQUIVALENCE_METRICS = ("primary_rtt_ms", "primary_loss_pct", "primary_delivered_payload_mbps")


def validate_margins(margins):
    if margins is None:
        return None
    if not isinstance(margins, dict) or set(margins) != set(EQUIVALENCE_METRICS) or any(
            not finite(v) or v <= 0 for v in margins.values()):
        raise ValueError("Specify positive margins for RTT (ms), loss (percentage points), delivered payload (Mbps)")
    return dict(margins)


def plan(manifest, scenarios, seeds, actions=(0, 1), order_seed=8801, cfg=None, margins=None):
    cfg = cfg if cfg is not None else TelemetryConfig()
    if cfg.observation_lag_ticks or cfg.carry_observations_across_episodes:
        raise ValueError('Probe-impact runs do not use policy history')
    scenarios, seeds, actions = list(scenarios), list(seeds), list(actions)
    validate_run_seed(order_seed, required=True)
    if any(not values or len(set(values)) != len(values) for values in (scenarios, seeds, actions)):
        raise ValueError("Declare nonempty unique scenarios, seeds, actions")
    for scenario in scenarios:
        p = resolve_run(manifest, scenario, "calibration", "heuristic")
        if "events" not in p["scenario"]:
            raise ValueError("Probe impact requires v2 calibration scenarios")
    for seed in seeds:
        validate_run_seed(seed, required=True)
    if any(type(a) is not int or a not in (0, 1) for a in actions):
        raise ValueError("Actions must be 0 or 1")
    rng = random.Random(order_seed)
    entries = []
    for scenario in scenarios:
        for seed in seeds:
            for action in actions:
                conditions = [True, False]
                rng.shuffle(conditions)
                entries.extend({"scenario_id": scenario, "seed": seed, "action": action, "probes": p}
                               for p in conditions)
    return {"manifest_sha256": digest(manifest), "scenarios": scenarios, "seeds": seeds,
            "actions": actions, "order_seed": order_seed, "entries": entries,
            "telemetry_config": asdict(cfg), "equivalence_margins": validate_margins(margins),
            "status": "planned_not_executed"}


from .network_guard import exclusive_run, prepare_network, start_network, stop_network


@exclusive_run
def run(manifest, scenario_id, seed, action, probes, cfg, trace):
    from .topology import SDWANTopo
    from .controller_api import ControllerAPI
    from .scenario_runtime import ScenarioRuntime
    from .telemetry_runtime import TelemetryCollector
    validate_run_seed(seed, required=True)
    protocol = resolve_run(manifest, scenario_id, "calibration", "heuristic")
    scenario = protocol["scenario"]
    if ("events" not in scenario or cfg.observation_lag_ticks or cfg.carry_observations_across_episodes
            or type(action) is not int or action not in (0, 1)):
        raise ValueError("Use v2 calibration, fixed action and lag zero")
    run_id = uuid.uuid4().hex
    append_trace(trace, {"kind": "probe_impact", "run_id": run_id, "protocol": protocol,
                        "seed": seed, "action": action, "probes": probes, "telemetry_config": asdict(cfg)}, mode="x")
    net = None
    api = ControllerAPI()
    runtime = collector = None
    try:
        net = prepare_network(SDWANTopo())
        start_network(net)
        api.wait_ready()
        runtime = ScenarioRuntime(net, scenario)
        collector = TelemetryCollector(net, cfg, probes=probes)
        collector.install()
        for ep in range(1, scenario["episodes"] + 1):
            conditions = runtime.apply_episode(ep)
            if ep == 1:
                runtime.start_traffic()
            for tick in range(scenario["ticks_per_episode"]):
                ack = api.apply_decision(action, f"{run_id}:{ep}:{tick}")
                measurement = collector.collect(action)
                background = runtime.check_traffic()
                append_trace(trace, {"kind": "window", "episode": ep, "tick": tick,
                                    "conditions": conditions, "controller_ack": ack,
                                    "background": background, "telemetry": measurement})
        append_trace(trace, {"kind": "complete"})
    except Exception as exc:
        append_trace(trace, {"kind": "failed", "error": str(exc)})
        raise
    finally:
        try:
            try:
                if collector is not None:
                    collector.close()
            finally:
                if runtime is not None:
                    runtime.close()
        finally:
            try:
                api.close()
            finally:
                try:
                    stop_network(net)
                except Exception as exc:
                    append_trace(trace, {"kind": "failed", "error": f"Network teardown: {exc}"})
                    raise


def read_run(path):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < 3 or rows[0].get("kind") != "probe_impact" or rows[-1].get("kind") != "complete":
        raise ValueError("Incomplete probe-impact run")
    header, windows = rows[0], rows[1:-1]
    protocol = header["protocol"]
    if resolve_run(protocol["manifest"], protocol["scenario_id"], "calibration", "heuristic") != protocol:
        raise ValueError("Probe-impact provenance differs")
    cfg = TelemetryConfig(**header["telemetry_config"])
    if cfg.observation_lag_ticks or cfg.carry_observations_across_episodes or type(header["probes"]) is not bool:
        raise ValueError("Invalid probe-impact configuration")
    scenario = protocol["scenario"]
    ticks = scenario["ticks_per_episode"]
    if len(windows) != scenario["episodes"] * ticks:
        raise ValueError("Incomplete measurement budget")
    for i, row in enumerate(windows):
        if (row["kind"] != "window" or row["episode"] != i // ticks + 1 or row["tick"] != i % ticks
                or row["controller_ack"]["action"] != header["action"]
                or row["controller_ack"]["barrier_acknowledged"] is not True):
            raise ValueError("Invalid fixed-action window")
        streams = row["telemetry"]["streams"]
        if set(streams) != ({"primary", "lte", "satellite"} if header["probes"] else {"primary"}):
            raise ValueError("Probe condition differs from measured streams")
        if not header["probes"] and row["telemetry"]["probe_payload_bytes_both_directions"] != 0:
            raise ValueError("Probe-off condition contains extra probe bytes")
    return header, windows


def compare(paths, design):
    expected = {(r["scenario_id"], r["seed"], r["action"], r["probes"]) for r in design["entries"]}
    if len(expected) != len(design["entries"]) or not expected:
        raise ValueError("Invalid paired design")
    observed, common = {}, design["telemetry_config"]
    for path in paths:
        header, windows = read_run(path)
        protocol = header["protocol"]
        key = (protocol["scenario_id"], header["seed"], header["action"], header["probes"])
        if key not in expected or key in observed or protocol["manifest_sha256"] != design["manifest_sha256"]:
            raise ValueError("Unexpected or duplicate probe-impact result")
        cfg = header["telemetry_config"]
        if cfg != common:
            raise ValueError("Different measurement configurations")
        observed[key] = primary_summary(r["telemetry"] for r in windows)
    if set(observed) != expected:
        raise ValueError("Incomplete paired probe-impact matrix")
    margins = validate_margins(design.get("equivalence_margins"))
    metrics = EQUIVALENCE_METRICS
    comparisons = []
    for scenario in design["scenarios"]:
        for action in design["actions"]:
            pairs = []
            for seed in design["seeds"]:
                on, off = observed[(scenario, seed, action, True)], observed[(scenario, seed, action, False)]
                pairs.append({"seed": seed, "on": on, "off": off,
                              "on_minus_off": {k: on[k] - off[k] if on[k] is not None and off[k] is not None else None
                                               for k in metrics}})
            summary = {}
            for key in metrics:
                vals = [p["on_minus_off"][key] for p in pairs]
                stats = describe(vals)
                summary[key] = {"n_pairs": stats["n"], "mean_delta": stats["mean"], **stats,
                                "equivalence": equivalence(vals,margins[key]) if margins is not None else None}
            comparisons.append({"scenario": scenario, "action": action, "pairs": pairs, "summary": summary})
    return {"design": design, "telemetry_config": common, "comparisons": comparisons,
            "interpretation": "Probe-on minus probe-off descriptive paired differences. "
                              "Equivalence is exploratory and requires locked margins, appropriate assumptions and coverage; "
                              "no joint equivalence claim is generated."}


def export_calibration(path):
    """Export raw pretraining calibration snapshots and labelled RTT records."""
    from .scenario_profiles import nominal_profiles
    header, windows = read_run(path)
    if not header["probes"]:
        raise ValueError("Calibration telemetry requires probes enabled")
    nominal = nominal_profiles()
    snapshots, labelled = [], []
    for row in windows:
        metrics = row["telemetry"]["metrics"]
        if all(metrics[t]["delay"] is not None for t in ("lte", "satellite")):
            snapshots.append(metrics)
        for tunnel in ("lte", "satellite"):
            rtt = metrics[tunnel]["delay"]
            if rtt is None:
                continue
            profile = row["conditions"]["requested_profiles"][tunnel]
            condition = "degraded" if profile != nominal[tunnel] else "nominal"
            labelled.append({"tunnel": tunnel, "condition": condition,
                             "scenario_id": f'{header["protocol"]["scenario_id"]}:{header["seed"]}:{header["action"]}',
                             "rtt_ms": rtt})
    return {"split": "calibration", "samples": labelled, "snapshots": snapshots,
            "unavailable_paired_snapshots": len(windows) - len(snapshots),
            "source_manifest_sha256": header["protocol"]["manifest_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--manifest", required=True)
    planning.add_argument("--scenarios", nargs="+", required=True)
    planning.add_argument("--seeds", nargs="+", type=int, required=True)
    planning.add_argument("--actions", nargs="+", type=int, default=[0, 1])
    planning.add_argument("--order-seed", type=int, default=8801)
    planning.add_argument("--equivalence-margins", help="JSON file; optional prespecified positive physical margins")
    add_telemetry_arguments(planning)
    running = sub.add_parser("run")
    running.add_argument("--manifest", required=True)
    running.add_argument("--scenario-id", required=True)
    running.add_argument("--seed", type=int, required=True)
    running.add_argument("--action", type=int, choices=[0, 1], required=True)
    running.add_argument("--probes", choices=["on", "off"], required=True)
    running.add_argument("--trace", required=True)
    add_telemetry_arguments(running)
    comparing = sub.add_parser("compare")
    comparing.add_argument("--plan", required=True)
    comparing.add_argument("--traces", nargs="+", required=True)
    exporting = sub.add_parser("export-calibration")
    exporting.add_argument("--trace", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(read_manifest(args.manifest), args.scenario_id, args.seed, args.action,
            args.probes == "on", TelemetryConfig(**telemetry_kwargs(args)), args.trace)
        return
    if args.command == "plan":
        value = plan(read_manifest(args.manifest), args.scenarios, args.seeds, args.actions,
                     args.order_seed, TelemetryConfig(**telemetry_kwargs(args)),
                     json.loads(Path(args.equivalence_margins).read_text()) if args.equivalence_margins else None)
    elif args.command == "compare":
        value = compare(args.traces, json.loads(Path(args.plan).read_text()))
    else:
        value = export_calibration(args.trace)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
