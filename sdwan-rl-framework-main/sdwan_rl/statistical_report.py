"""Predeclared seed-level analysis of the main, revised dual-tunnel benchmark."""
import argparse
import json
import random
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from .statistical_core import describe, paired, holm, ALPHA, finite
from .config import RLConfig
from .dqn_settings import CURRENT_DQN
from .experiment_protocol import normalize_manifest, read_manifest, resolve_run, digest
from .telemetry_core import TelemetryConfig
from .policy_audit import validate_run_seed
from .benchmark_audit import BENCHMARK_AGENTS, audit_trace, paired_report
from .learning_curves import read_run
from .telemetry_audit import read_timed_trace, primary_summary
from .cost_audit import cost_qos_summary
from .runtime_environment import runtime_environment

PRIMARY_METRICS = ("primary_loss_pct", "primary_rtt_ms")
RELATIVE_METRICS = set(PRIMARY_METRICS)
METRICS = PRIMARY_METRICS + ("primary_rtt_echo_coverage", "primary_delivered_payload_mbps",
    "mean_logged_reward", "mean_selected_cost_index", "mean_qos_utility_valid_selected",
    "mean_selected_loss_pct", "mean_selected_rtt_ms", "mean_selected_residual_mbps",
    "lte_selection_fraction", "reward_to_cost_ratio", "mean_tick_work_s")


def episode_windows(scenario):
    """Declared inclusive impairment union, never selected from measured outcomes."""
    episodes = set(range(1, scenario["episodes"]+1))
    ranges = ([(e["start_episode"], e["end_episode"]) for e in scenario["events"]]
              if "events" in scenario else scenario["degrade_lte_ranges"])
    impaired = {ep for lo,hi in ranges for ep in range(lo,hi+1)}
    if not impaired <= episodes:
        raise ValueError("Invalid impairment windows")
    windows = {"full": sorted(episodes)}
    if impaired:
        windows["impairment"] = sorted(impaired)
        windows["last_impairment_episode"] = [max(impaired)]
        post = [ep for ep in sorted(episodes) if ep > max(impaired)]
        if post:
            windows["post_impairment"] = post
    return windows


def make_plan(manifest, train_id, primary_scenario, seeds, cfg, telemetry,
              agents=BENCHMARK_AGENTS, ewma_alpha=.3, order_seed=350037):
    manifest = normalize_manifest(manifest)
    seeds, agents = list(seeds), list(agents)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Predeclare at least two unique replicate seeds")
    for seed in seeds:
        validate_run_seed(seed, required=True)
    validate_run_seed(order_seed, required=True)
    if len(set(agents)) != len(agents) or not set(agents) <= set(BENCHMARK_AGENTS) or not {"dqn","ewma","greedy_utility"} <= set(agents):
        raise ValueError("Primary family requires DQN, EWMA and Greedy Utility")
    if (cfg.dqn_version != CURRENT_DQN or cfg.reward_mode != "normalized"
            or telemetry.observation_lag_ticks or telemetry.carry_observations_across_episodes):
        raise ValueError("Main revised benchmark uses current DQN, normalized reward, standard observations")
    if not finite(ewma_alpha) or not 0 < ewma_alpha <= 1:
        raise ValueError("Invalid EWMA alpha")
    training = resolve_run(manifest, train_id, "train", "online")
    tests = [r for r in manifest["scenarios"] if r["split"] == "held_out_test"]
    windows = {r["id"]: episode_windows(r) for r in tests}
    if "impairment" not in windows.get(primary_scenario, {}):
        raise ValueError("Primary test scenario must have a declared impairment window")
    family = [{"scenario_id": primary_scenario, "window": "impairment", "metric": metric,
               "candidate": "dqn", "reference": baseline}
              for metric in PRIMARY_METRICS for baseline in ("ewma","greedy_utility")]
    rng = random.Random(order_seed)
    training_order = list(seeds)
    rng.shuffle(training_order)
    blocks = [(seed,r["id"]) for seed in seeds for r in tests]
    rng.shuffle(blocks)
    evaluation_order = []
    for seed,scenario_id in blocks:
        order = list(agents)
        rng.shuffle(order)
        evaluation_order.extend({"seed": seed, "scenario_id": scenario_id, "agent": a} for a in order)
    return {"schema": 1, "manifest": manifest, "manifest_sha256": digest(manifest),
            "training_scenario_id": training["scenario_id"], "primary_scenario_id": primary_scenario,
            "seeds": seeds, "agents": agents, "rl_config": asdict(cfg),
            "telemetry_config": asdict(telemetry), "ewma_alpha": ewma_alpha,
            "windows": windows, "primary_family": family, "alpha": ALPHA,
            "order_seed": order_seed, "training_order": training_order,
            "evaluation_order": evaluation_order,
            "status": "predeclared_design_not_execution_evidence"}


def validate_plan(plan):
    expected = make_plan(plan["manifest"], plan["training_scenario_id"], plan["primary_scenario_id"],
                         plan["seeds"], RLConfig(**plan["rl_config"]),
                         TelemetryConfig(**plan["telemetry_config"]), plan["agents"],
                         plan["ewma_alpha"], plan["order_seed"])
    if expected != plan:
        raise ValueError("Analysis plan differs from versioned predeclared design")
    return plan


def window_metrics(rows, cfg, telemetry):
    """Within-run aggregation, no inference over ticks or packets."""
    if not rows:
        raise ValueError("Empty measurement window")
    measurements = [r["telemetry"] for r in rows]
    for m in measurements:
        primary = m["streams"]["primary"]
        sent, received, echoes = [primary[k] for k in ("sent_packets","received_packets","echo_received_packets")]
        if (any(type(v) is not int for v in (sent,received,echoes)) or sent != telemetry.primary_packets
                or not 0 <= echoes <= received <= sent
                or primary["received_payload_bytes"] != received*telemetry.primary_payload_bytes):
            raise ValueError("Primary counts differ from the declared workload")
        rtt = primary["rtt_ms"]
        if (echoes == 0 and rtt is not None) or (echoes > 0 and (not finite(rtt) or rtt <= 0)):
            raise ValueError("Invalid RTT/echo coverage")
        duration = m["window_end_s"]-m["window_start_s"]
        if not math.isclose(duration, telemetry.window_s+telemetry.grace_s, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Measurement support differs from workload configuration")
    primary = primary_summary(measurements)
    cost = cost_qos_summary(rows, cfg)
    return {**primary, **cost, "mean_logged_reward": fmean(r["reward"] for r in rows),
            "mean_tick_work_s": fmean(r["tick_work_s"] for r in rows)}


def read_inventory(path, plan, split):
    """Track failed/incomplete attempts; do not silently substitute retries."""
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or rows[0].get("kind") != "run":
        raise ValueError(f"{path}: no identifiable run header")
    h, end = rows[0], rows[-1]
    agent, seed, protocol = h["agent_type"], h["run_seed"], h["protocol"]
    validate_run_seed(seed, required=True)
    mode = "online" if split == "train" else ("frozen" if agent == "dqn" else "heuristic")
    expected = resolve_run(plan["manifest"], protocol["scenario_id"], split, mode)
    if (h["mode"] != mode or protocol != expected or seed not in plan["seeds"]
            or h["rl_config"] != plan["rl_config"] or h.get("telemetry_schema") != 1
            or h.get("telemetry_config") != plan["telemetry_config"]):
        raise ValueError(f"{path}: trace does not match locked plan")
    if split == "train":
        if agent != "dqn" or protocol["scenario_id"] != plan["training_scenario_id"]:
            raise ValueError("Unexpected training run")
    elif agent not in plan["agents"]:
        raise ValueError("Unexpected evaluation agent")
    if agent == "ewma" and h["heuristic_config"] != {"alpha": plan["ewma_alpha"], "reset_each_episode": True}:
        raise ValueError("EWMA alpha differs from the locked plan")
    status = ("complete" if end.get("kind") == "complete" else
              "failed" if end.get("kind") == "failed" else "incomplete")
    return {"path": str(path), "agent": agent, "seed": seed, "scenario_id": protocol["scenario_id"],
            "status": status, "failure": end.get("error") if status == "failed" else None,
            "failure_episode": end.get("episode") if status == "failed" else None,
            "failure_tick": end.get("tick") if status == "failed" else None,
            "recorded_decisions": sum(r.get("kind") == "decision" for r in rows),
            "run_id": h.get("run_id")}


def report(plan, training_paths, evaluation_paths):
    validate_plan(plan)
    cfg, telemetry = RLConfig(**plan["rl_config"]), TelemetryConfig(**plan["telemetry_config"])
    train, evaluations, train_inventory, test_inventory = {}, {}, {}, {}
    run_ids = set()
    def record(row, inventory, key):
        if key in inventory:
            raise ValueError("Duplicate attempt for a declared unit; preserve retries separately")
        identifier = row["run_id"]
        if not isinstance(identifier,str) or not identifier or identifier in run_ids:
            raise ValueError("Missing or repeated run identifier")
        run_ids.add(identifier)
        inventory[key] = row

    for path in training_paths:
        row = read_inventory(path,plan,"train")
        seed = row["seed"]
        record(row,train_inventory,seed)
        if row["status"] == "complete":
            train[seed] = read_run(path,"train")
    if len({r["initial_weights_sha256"] for r in train.values()}) != len(train):
        raise ValueError("Training runs reused initial weights")
    expected = {(scenario,seed,agent) for scenario in plan["windows"] for seed in plan["seeds"] for agent in plan["agents"]}
    audits = []
    for path in evaluation_paths:
        row = read_inventory(path,plan,"held_out_test")
        key = (row["scenario_id"],row["seed"],row["agent"])
        if key not in expected:
            raise ValueError("Unexpected evaluation cell")
        record(row,test_inventory,key)
        if row["status"] != "complete":
            continue
        a = audit_trace(path)
        h,_,records = read_timed_trace(path)
        if row["agent"] == "dqn":
            tr = train.get(row["seed"])
            if tr is None or tr["final_weights_sha256"] != a["model_weights_sha256"] or tr["protocol"] != a["training_protocol"]:
                raise ValueError("Complete frozen DQN trace lacks its matching complete training trace")
        windows = {}
        for name,episodes in plan["windows"][row["scenario_id"]].items():
            subset = set(episodes)
            windows[name] = window_metrics([r for r in records if r["episode"] in subset],cfg,telemetry)
        per_episode = [window_metrics([r for r in records if r["episode"] == ep],cfg,telemetry)
                       for ep in range(1,h["episode_count"]+1)]
        evaluations[key] = {**row,"windows": windows,"episodes": per_episode}
        audits.append(a)
    complete = set(train) == set(plan["seeds"]) and set(evaluations) == expected
    if complete:
        paired_report(audits,plan["seeds"],plan["agents"])
    inventory = {"training": [train_inventory.get(s,{"seed":s,"status":"missing"}) for s in plan["seeds"]],
                 "evaluation": [test_inventory.get((e["scenario_id"],e["seed"],e["agent"]),{**e,"status":"missing"})
                                for e in plan["evaluation_order"]]}
    summaries, curves, contrasts = [], [], []
    def values(scenario,agent,window,metric):
        return [evaluations[(scenario,s,agent)]["windows"][window][metric]
                if (scenario,s,agent) in evaluations else None for s in plan["seeds"]]
    for scenario, windows in plan["windows"].items():
        for agent in plan["agents"]:
            for window in windows:
                summaries.append({"scenario_id":scenario,"agent":agent,"window":window,
                    "episodes":windows[window],"metrics":{m:describe(values(scenario,agent,window,m)) for m in METRICS}})
            curves.append({"scenario_id":scenario,"agent":agent,"points":[
                {"episode": ep, "metrics": {m: describe([
                    evaluations[(scenario,s,agent)]["episodes"][ep-1][m] if (scenario,s,agent) in evaluations else None
                    for s in plan["seeds"]]) for m in PRIMARY_METRICS+("primary_rtt_echo_coverage",)}}
                for ep in windows["full"]]})
        for window in windows:
            for baseline in plan["agents"]:
                if baseline == "dqn":
                    continue
                for metric in METRICS:
                    spec = {"scenario_id":scenario,"window":window,"metric":metric,"candidate":"dqn","reference":baseline}
                    primary = spec in plan["primary_family"]
                    result = paired(values(scenario,"dqn",window,metric),values(scenario,baseline,window,metric),
                                    relative=metric in RELATIVE_METRICS,test=primary and complete)
                    contrasts.append({**spec,"primary":primary,"effect":result})
    family = [next(r for r in contrasts if all(r[k]==v for k,v in spec.items())) for spec in plan["primary_family"]]
    pvals = [r["effect"]["test"]["p_value"] if r["effect"]["test"] is not None else None for r in family]
    ready = complete and all(p is not None for p in pvals)
    for r, correction in zip(family,holm(pvals,plan["alpha"])):
        r["inference"] = {**correction, "reject": correction["reject"] and ready,
                          "family_complete": ready}
    return {"analysis_schema":1,"plan":plan,"plan_sha256":digest(plan),
            "analysis_environment":runtime_environment(),"status":"complete_matrix" if complete else "incomplete_matrix_descriptive_only",
            "primary_family_ready":ready,"inventory":inventory,"seed_order":plan["seeds"],
            "individual_training_runs":list(train.values()),"individual_evaluation_runs":list(evaluations.values()),
            "summaries":summaries,"episode_curves":curves,"contrasts":contrasts,
            "interpretation":"Independent run-level estimates; incomplete arms are descriptive only. "
               "95% t intervals are pointwise/unadjusted; sign-flip assumptions must be justified. "
               "Relative reduction uses matched complete seed means. RTT is conditional on received echoes. "
               "Primary flow metrics differ from selected-path probe metrics."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--manifest",required=True)
    pl.add_argument("--train-id",required=True)
    pl.add_argument("--primary-scenario",required=True)
    pl.add_argument("--seeds",nargs="+",type=int,required=True)
    pl.add_argument("--agents",nargs="+",choices=BENCHMARK_AGENTS,default=list(BENCHMARK_AGENTS))
    pl.add_argument("--rl-config",required=True)
    pl.add_argument("--telemetry-config",required=True)
    pl.add_argument("--ewma-alpha",type=float,default=.3)
    pl.add_argument("--order-seed",type=int,default=350037)
    pl.add_argument("--output",required=True)
    ev = sub.add_parser("report")
    ev.add_argument("--plan",required=True)
    ev.add_argument("--training",nargs="*",default=[])
    ev.add_argument("--evaluation",nargs="*",default=[])
    ev.add_argument("--output",required=True)
    args = parser.parse_args()
    def load(p):
        return json.loads(Path(p).read_text(encoding="utf-8"))
    if args.command == "plan":
        result = make_plan(read_manifest(args.manifest),args.train_id,args.primary_scenario,args.seeds,
                           RLConfig(**load(args.rl_config)),TelemetryConfig(**load(args.telemetry_config)),
                           args.agents,args.ewma_alpha,args.order_seed)
    else:
        result = report(load(args.plan),args.training,args.evaluation)
    with Path(args.output).open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
