"""Complete run summaries and matched scalability / RR diagnostic comparisons."""
import argparse
import json
import math
from statistics import fmean, stdev
import numpy as np

from .agents.uniform_random import UniformRandomAgent
from .config import RLConfig
from .statistical_core import describe, paired
from .scale_cases import (CASES, validate_scenario, digest, primary_rules, static_rules,
                          path_residual, PRIMARY_PORT)
from .scale_policy import PathEnvironment, policy_signature
from .telemetry_core import TelemetryConfig, throughput_mbps
from .telemetry_audit import distribution
from .policy_audit import validate_run_seed


def stream_summary(measurements, name):
    rows = [m["streams"][name] for m in measurements]
    sent, received, echoes = (sum(r[k] for r in rows) for k in
                              ("sent_packets", "received_packets", "echo_received_packets"))
    seconds = math.fsum(m["window_end_s"] - m["window_start_s"] for m in measurements)
    if sent <= 0 or not 0 <= echoes <= received <= sent or seconds <= 0:
        raise ValueError("Invalid stream totals")
    late = sum(r["late_sequence_unique"] for r in rows)
    if not 0 <= late <= received:
        raise ValueError("Invalid late-arrival totals")
    return {"sent": sent, "received": received, "echoes": echoes,
            "loss_pct": 100*(sent-received)/sent,
            "mean_rtt_ms": math.fsum(r["rtt_ms"]*r["echo_received_packets"] for r in rows
                                    if r["echo_received_packets"]) / echoes if echoes else None,
            "echo_coverage": echoes/sent,
            "delivered_payload_mbps": 8*sum(r["received_payload_bytes"] for r in rows)/(1e6*seconds),
            "late_sequence_unique": late, "late_sequence_fraction": late/received if received else None,
            "max_sequence_lag": max(r["max_sequence_lag"] for r in rows)}


def validate_measurement(scenario, action, secondary, m):
    graph = scenario["graph"]
    rules = static_rules(scenario, secondary) + primary_rules(graph, action)
    expected = {str(r["cookie"]): r for r in rules}
    if set(m["flow_deltas"]) != set(expected):
        raise ValueError("Missing or unexpected route counters")
    for cookie, rule in expected.items():
        row = m["flow_deltas"][cookie]
        if row["site"] != rule["site"] or row["out_port"] != rule["out_port"] or any(
                type(row[k]) is not int or row[k] < 0 for k in ("packets", "bytes")):
            raise ValueError("Invalid route counter")
    edge_rates = {e["id"]: throughput_mbps(m["counters_before"][e["id"]], m["counters_after"][e["id"]])
                  for e in graph["edges"]}
    if edge_rates != m["edge_tx_mbps"]:
        raise ValueError("Inconsistent edge throughput")
    names = {"primary", *(p["id"] for p in graph["paths"])}
    if secondary != "off":
        names.add("secondary")
    if set(m["streams"]) != names:
        raise ValueError("Unexpected measurement stream")
    for p in graph["paths"]:
        sample = m["streams"][p["id"]]
        expect = {"delay": sample["rtt_ms"], "packet_loss": sample["loss_pct"],
                  **path_residual(graph, p, edge_rates)}
        if m["metrics"][p["id"]] != expect:
            raise ValueError("Inconsistent path metrics")
    # Validate receiver counts against the exact primary destination rule.
    final = [r for r in rules if r["site"] == 2 and r["match"].get("udp_dst") == PRIMARY_PORT]
    if len(final) != 1 or m["streams"]["primary"]["received_packets"] > m["flow_deltas"][str(final[0]["cookie"])]["packets"]:
        raise ValueError("Primary delivery exceeds route evidence")
    if secondary == "rr":
        counts = m["rr_dispatched"]
        received = m["rr_path_egress_matches"]
        if (len(counts) != len(graph["paths"]) or len(received) != len(counts)
                or any(type(n) is not int or n < 0 for n in counts + received)
                or sum(counts) <= 0 or max(counts) - min(counts) > 1
                or any(a > b for a, b in zip(received, counts))):
            raise ValueError("Invalid RR dispatch/receive evidence")
    elif m["rr_dispatched"] is not None or m["rr_path_egress_matches"] is not None:
        raise ValueError("Unexpected RR dispatch evidence")


def audit(path):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < 3 or rows[0].get("kind") != "scale_run" or rows[-1].get("kind") != "complete":
        raise ValueError("Incomplete or failed scalability run")
    header, footer, records = rows[0], rows[-1], rows[1:-1]
    scenario = validate_scenario(header["scenario"])
    signature, seed, agent = header["signature"], header["seed"], header["agent"]
    validate_run_seed(seed, required=True)
    cfg = RLConfig(**signature["rl_config"])
    telemetry = TelemetryConfig(**signature["telemetry_config"])
    if signature != policy_signature(scenario, cfg, telemetry) or header["scenario_sha256"] != digest(scenario):
        raise ValueError("Scenario/signature mismatch")
    if scenario["split"] not in ("calibration", "held_out_test"):
        raise ValueError("Report only evaluation runs, not training")
    fixed = scenario["split"] == "calibration"
    if (fixed and agent != "fixed") or (not fixed and agent not in ("dqn", "greedy", "random")):
        raise ValueError("Unexpected evaluation policy")
    if not fixed and header["secondary"] != "off":
        raise ValueError("RR diagnostic is separate from held-out benchmarking")
    if agent == "dqn":
        meta = header["training_metadata"]
        if (not isinstance(meta, dict) or meta["signature"] != signature or meta["training_seed"] != seed
                or meta["weights_sha256"] != header["initial_weights_sha256"]
                or header["initial_weights_sha256"] != footer["weights_sha256"]):
            raise ValueError("Inconsistent frozen checkpoint")
        train = validate_scenario(meta["training_scenario"])
        if train["split"] != "train" or digest(train["graph"]) != signature["graph_sha256"]:
            raise ValueError("Training provenance mismatch")
        state = footer["training_state"]
        if any(state[k] != 0 for k in ("optimizer_updates", "target_syncs", "replay_size", "epsilon")):
            raise ValueError("Unexpected learning during frozen evaluation")
    elif any(v is not None for v in (header["initial_weights_sha256"], footer["weights_sha256"],
                                      header["training_metadata"], header["checkpoint"])):
        raise ValueError("Non-learning policy has checkpoint provenance")
    env = PathEnvironment(scenario["graph"], cfg)
    random_agent = UniformRandomAgent(seed=seed, action_size=env.n) if agent == "random" else None
    ticks = scenario["ticks_per_episode"]
    if len(records) != ticks * scenario["episodes"]:
        raise ValueError("Wrong evaluation budget")
    previous, last_action = None, 0
    rewards, costs, measurements, actions = [], [], [], []
    for i, row in enumerate(records):
        ep, tick = i//ticks+1, i%ticks
        if tick == 0:
            env.reset()
            previous, last_action = None, 0
        if (row["kind"] != "scale_decision" or row["episode"] != ep or row["tick"] != tick
                or row["decision_metrics"] != previous or row["epsilon"] != 0
                or row["exploratory"] is not False or row["dqn_update"] is not None
                or row["state"] != env.state().tolist()):
            raise ValueError("Inconsistent causal observation or evaluation tick")
        action = row["action"]
        if type(action) is not int or not 0 <= action < env.n:
            raise ValueError("Invalid action")
        if agent == "dqn":
            q = np.asarray(row["q_values"])
            if q.shape != (env.n,) or not np.isfinite(q).all() or action != int(np.argmax(q)):
                raise ValueError("Not a frozen greedy Q decision")
        elif agent == "greedy":
            expected, values = env.greedy(last_action)
            if expected != action or values != row["greedy_values"]:
                raise ValueError("Inconsistent utility baseline")
        elif agent == "random":
            if random_agent.decide(previous)[0] != action:
                raise ValueError("Inconsistent seeded random decision")
        elif action != header["fixed_action"]:
            raise ValueError("Fixed route changed")
        ack = row["controller_ack"]
        if (ack["action"] != action or ack["scenario_sha256"] != digest(scenario)
                or ack["decision_id"] != f'{header["run_id"]}:{ep}:{tick}'
                or ack["path_id"] != scenario["graph"]["paths"][action]["id"]
                or ack["datapaths"] != scenario["graph"]["sites"] or ack["barrier_acknowledged"] is not True):
            raise ValueError("Unacknowledged or mismatched action")
        m = row["measurement"]
        validate_measurement(scenario, action, header["secondary"], m)
        env.update(m["metrics"])
        try:
            terms = env.terms(action)
        except ValueError:
            if not fixed:
                raise
            terms = None
        if terms != row["reward_terms"] or env.state().tolist() != row["next_state"]:
            raise ValueError("Inconsistent shared reward or next observation")
        if terms is not None:
            rewards.append(terms["reward"])
        costs.append(scenario["graph"]["paths"][action]["cost_index"])
        measurements.append(m)
        actions.append(action)
        previous, last_action = m["metrics"], action
    return {"path": str(path), "scenario": scenario, "scenario_sha256": digest(scenario),
            "signature": signature, "agent": agent, "seed": seed,
            "training_metadata": header["training_metadata"], "secondary": header["secondary"],
            "fixed_action": header["fixed_action"], "primary": stream_summary(measurements, "primary"),
            "secondary_qos": stream_summary(measurements, "secondary") if header["secondary"] != "off" else None,
            "rr_dispatched_by_path": [sum(m["rr_dispatched"][i] for m in measurements)
                                      for i in range(env.n)] if header["secondary"] == "rr" else None,
            "rr_path_egress_matches": [sum(m["rr_path_egress_matches"][i] for m in measurements)
                                        for i in range(env.n)] if header["secondary"] == "rr" else None,
            "mean_reward": fmean(rewards) if rewards else None,
            "reward_coverage": len(rewards)/len(records),
            "mean_selected_cost_index": fmean(costs),
            "selection_fractions": [actions.count(i)/len(actions) for i in range(env.n)],
            "probe_payload_bytes": sum(m["probe_payload_bytes_both_directions"] for m in measurements),
            "probe_ovs_ingress_match_bytes": sum(m["probe_ovs_ingress_match_bytes"] for m in measurements),
            "timing_s": {k: distribution(r["timing_s"][k] for r in records) for k in records[0]["timing_s"]},
            "elapsed_s": footer["elapsed_s"]}


def stats(values):
    return describe(values)


def benchmark(reports, cases, seeds, agents):
    cases, seeds, agents = list(cases), list(seeds), list(agents)
    if any(not v or len(set(v)) != len(v) for v in (cases, seeds, agents)) or not set(cases) <= set(CASES) or not set(agents) <= {"dqn","greedy","random"}:
        raise ValueError("Specify unique nonempty benchmark dimensions")
    for seed in seeds:
        validate_run_seed(seed, required=True)
    expected = {(case, seed, agent) for case in cases for seed in seeds for agent in agents}
    observed, signatures, train_signatures = {}, {}, {}
    common, training_budget = None, None
    for r in reports:
        scenario = r["scenario"]
        case = scenario["graph"]["case"]
        key = (case, r["seed"], r["agent"])
        if scenario["split"] != "held_out_test" or key not in expected or key in observed:
            raise ValueError("Unexpected or duplicate run")
        sign = (r["scenario_sha256"], r["signature"])
        if signatures.setdefault(case, sign) != sign:
            raise ValueError("Within-case configuration mismatch")
        settings = (r["signature"]["rl_config"], r["signature"]["telemetry_config"],
                    scenario["episodes"], scenario["ticks_per_episode"])
        if common is None:
            common = settings
        elif common != settings:
            raise ValueError("Cross-case configuration/budget differs")
        if r["agent"] == "dqn":
            train = r["training_metadata"]["training_scenario"]
            if train_signatures.setdefault(case, digest(train)) != digest(train):
                raise ValueError("Different training scenarios within a case")
            budget = (train["episodes"], train["ticks_per_episode"])
            if training_budget is None:
                training_budget = budget
            elif training_budget != budget:
                raise ValueError("Different training budgets across cases")
        observed[key] = r
    if set(observed) != expected:
        raise ValueError("Incomplete declared scalability matrix")
    summaries = []
    for case in cases:
        for agent in agents:
            runs = [observed[(case, seed, agent)] for seed in seeds]
            values = {"mean_reward": [r["mean_reward"] for r in runs],
                      "cost_index": [r["mean_selected_cost_index"] for r in runs],
                      "policy_decision_mean_s": [r["timing_s"]["encode_and_decide"]["mean"] for r in runs],
                      "collection_mean_s": [r["timing_s"]["collection"]["mean"] for r in runs],
                      "tick_work_mean_s": [r["timing_s"]["tick_work"]["mean"] for r in runs]}
            for k in ("loss_pct", "mean_rtt_ms", "echo_coverage", "delivered_payload_mbps"):
                values[k] = [r["primary"][k] for r in runs]
            effects = None
            if agent != "dqn" and "dqn" in agents:
                candidates = [observed[(case, seed, "dqn")] for seed in seeds]
                effects = {k: paired([r["primary"][k] for r in candidates],
                                     [r["primary"][k] for r in runs],
                                     relative=k in ("loss_pct","mean_rtt_ms"))
                           for k in ("loss_pct","mean_rtt_ms","echo_coverage","delivered_payload_mbps")}
            summaries.append({"case": case, "agent": agent, "metrics": {k: stats(v) for k,v in values.items()},
                              "dqn_minus_this_baseline": effects})
    return {"cases": cases, "seeds": seeds, "agents": agents, "summaries": summaries,
            "runs": list(observed.values()),
            "interpretation": "Matched within-case comparisons; changing catalog changes difficulty. "
                              "Finite-size scalability evidence only, not transfer or a complexity proof."}


def rr_compare(reports, case, seeds, actions):
    seeds, actions = list(seeds), list(actions)
    if not seeds or not actions or len(set(seeds)) != len(seeds) or len(set(actions)) != len(actions):
        raise ValueError("Specify unique replicate seeds and fixed primary actions")
    expected = {(seed, action, mode) for seed in seeds for action in actions for mode in ("pinned", "rr")}
    observed, common = {}, None
    for r in reports:
        key = (r["seed"], r["fixed_action"], r["secondary"])
        if (r["agent"] != "fixed" or r["scenario"]["graph"]["case"] != case
                or key not in expected or key in observed):
            raise ValueError("Unexpected or duplicate RR comparison run")
        sign = (r["scenario_sha256"], r["signature"])
        if common is None:
            common = sign
        elif common != sign:
            raise ValueError("RR arms have different configurations")
        observed[key] = r
    if set(observed) != expected:
        raise ValueError("Incomplete pinned/RR pairs")
    summaries = []
    for action in actions:
        pairs = []
        for seed in seeds:
            pinned, rr = (observed[(seed, action, mode)] for mode in ("pinned", "rr"))
            delta = {}
            for stream in ("primary", "secondary_qos"):
                for k in ("loss_pct", "mean_rtt_ms", "echo_coverage", "delivered_payload_mbps", "late_sequence_fraction"):
                    a, b = pinned[stream][k], rr[stream][k]
                    delta[stream+"_"+k] = b-a if a is not None and b is not None else None
            pairs.append({"seed": seed, "rr_minus_pinned": delta})
        summaries.append({"action": action, "pairs": pairs,
                          "metrics": {k: stats([p["rr_minus_pinned"][k] for p in pairs])
                                      for k in pairs[0]["rr_minus_pinned"]}})
    return {"case": case, "summaries": summaries, "runs": list(observed.values()),
            "interpretation": "RR versus one fixed secondary path with equal offered load; "
                              "routing distribution and controller processing also differ. "
                              "Not a pure causal isolation of reordering alone."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    single = sub.add_parser("audit")
    single.add_argument("--traces", nargs="+", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--traces", nargs="+", required=True)
    bench.add_argument("--cases", nargs="+", choices=CASES, required=True)
    bench.add_argument("--seeds", nargs="+", type=int, required=True)
    bench.add_argument("--agents", nargs="+", choices=["dqn", "greedy", "random"], default=["dqn", "greedy"])
    rr = sub.add_parser("rr")
    rr.add_argument("--traces", nargs="+", required=True)
    rr.add_argument("--case", choices=CASES, required=True)
    rr.add_argument("--seeds", nargs="+", type=int, required=True)
    rr.add_argument("--actions", nargs="+", type=int, required=True)
    args = parser.parse_args()
    reports = [audit(p) for p in args.traces]
    result = (reports if args.command == "audit" else
              benchmark(reports, args.cases, args.seeds, args.agents) if args.command == "benchmark" else
              rr_compare(reports, args.case, args.seeds, args.actions))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
