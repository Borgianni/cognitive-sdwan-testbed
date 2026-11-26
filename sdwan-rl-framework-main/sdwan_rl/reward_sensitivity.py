"""Calibration-only reward sensitivity and a controlled policy study plan."""
import argparse
import json
from dataclasses import asdict, replace
from statistics import fmean, stdev

from .agents.utility import snapshot_utilities
from .config import RLConfig
from .experiment_protocol import read_manifest
from .policy_audit import validate_run_seed

WEIGHTS = ("w_b", "w_a", "w_c", "w_d")


def candidates(base):
    """One-factor-at-a-time study, plus zero cost and cost-ratio cases."""
    if base.reward_mode != "normalized" or any(getattr(base, k) <= 0 for k in WEIGHTS):
        raise ValueError("Use a normalized reference with positive weights")
    if not base.lte_cost_index < base.sat_cost_index < 5 * base.lte_cost_index:
        raise ValueError("Reference LEO/LTE cost ratio must be between 1 and 5")
    yield "reference", base
    for key in WEIGHTS:
        for label, factor in (("half", .5), ("double", 2.0)):
            yield f"{key}_{label}", replace(base, **{key: getattr(base, key) * factor})
    yield "cost_free", replace(base, w_d=0.0)
    yield "equal_cost", replace(base, sat_cost_index=base.lte_cost_index)
    yield "high_leo_cost", replace(base, sat_cost_index=5 * base.lte_cost_index)


def offline(snapshots, base):
    """Use identical calibration snapshots; this is not a learned-policy test."""
    snapshots = list(snapshots)
    if not snapshots:
        raise ValueError("Need calibration snapshots")
    common_scores = [snapshot_utilities(m, base) for m in snapshots]
    reference_actions = [int(v[1] > v[0]) for v in common_scores]
    rows = []
    for identifier, cfg in candidates(base):
        actions = [int(v[1] > v[0]) for v in (snapshot_utilities(m, cfg) for m in snapshots)]
        rows.append({"id": identifier, "rl_config": asdict(cfg),
                     "lte_fraction": fmean(a == 0 for a in actions),
                     "different_from_reference_fraction": fmean(a != b for a, b in zip(actions, reference_actions)),
                     "mean_common_reference_utility": fmean(v[a] for v, a in zip(common_scores, actions)),
                     "mean_common_reference_gap": fmean(max(v) - v[a] for v, a in zip(common_scores, actions))})
    return {"status": "calibration_snapshot_diagnostic", "samples": len(snapshots),
            "reference_config": asdict(base), "rows": rows,
            "interpretation": "Snapshot greedy preferences only, not DQN performance or causal effects."}


def make_plan(base, manifest, train_id, test_ids, seeds):
    """Predeclare a manageable matched subset; never launch experiments."""
    seeds, test_ids = list(seeds), list(test_ids)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use nonempty unique seeds")
    for seed in seeds:
        validate_run_seed(seed, required=True)
    by_id = {r["id"]: r for r in manifest["scenarios"]}
    if by_id.get(train_id, {}).get("split") != "train":
        raise ValueError("Expected a training scenario")
    if (not test_ids or len(set(test_ids)) != len(test_ids)
            or any(by_id.get(t, {}).get("split") != "held_out_test" for t in test_ids)):
        raise ValueError("Declare a nonempty unique held-out sensitivity subset")
    from .experiment_protocol import digest, normalize_manifest
    manifest = normalize_manifest(manifest)
    configs = [{"id": name, "rl_config": asdict(cfg)} for name, cfg in candidates(base)]
    return {"status": "planned_not_executed", "manifest_sha256": digest(manifest),
            "training_scenario_id": train_id, "test_scenario_ids": test_ids,
            "seeds": seeds, "reference_config": asdict(base), "configs": configs,
            "training_runs": len(configs) * len(seeds),
            "frozen_runs": len(configs) * len(seeds) * len(test_ids),
            "interpretation": "Sensitivity subset, not full benchmarking matrix. Lock before final testing."}


def score_trace(path, base):
    """Validate a frozen DQN trace, then rescore its recorded choices under base."""
    from .benchmark_audit import audit_trace
    from .env import SDWANEnv
    report = audit_trace(path)
    if report["agent"] != "dqn":
        raise ValueError("Reward sensitivity requires trained frozen DQN traces")
    with open(path, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    choices, utilities, costs, delays, losses, residuals = [], [], [], [], [], []
    switches = opportunities = 0
    previous = None
    env = SDWANEnv(base)
    for row in records[1:-1]:
        action = row["action"]
        if row["tick"] == 0:
            previous = None
        if previous is not None:
            opportunities += 1
            switches += action != previous
        previous = action
        env.update_from_metrics(row["measurement_metrics"])
        terms = env.reward_terms(action)  # normalized reference: reject invalid selected metrics
        selected = row["measurement_metrics"]["lte" if action == 0 else "satellite"]
        choices.append(action)
        utilities.append(terms["reward"])
        costs.append(terms["cost_index"])
        delays.append(selected["delay"])
        losses.append(selected["packet_loss"])
        residuals.append(terms["residual_mbps"])
    return {**report, "mean_common_reference_reward": fmean(utilities),
            "mean_common_reference_cost_index": fmean(costs),
            "selected_delay_ms": fmean(delays), "selected_loss_pct": fmean(losses),
            "selected_residual_mbps": fmean(residuals), "lte_fraction": fmean(a == 0 for a in choices),
            "within_episode_switch_fraction": switches / opportunities if opportunities else None}


def policy_report(paths, plan):
    """Require every predeclared config/seed/test case, with model consistency."""
    base = RLConfig(**plan["reference_config"])
    configurations = {row["id"]: row["rl_config"] for row in plan["configs"]}
    if configurations != {name: asdict(cfg) for name, cfg in candidates(base)}:
        raise ValueError("Sensitivity plan configuration differs from candidate design")
    expected = {(name, seed, scenario) for name in configurations
                for seed in plan["seeds"] for scenario in plan["test_scenario_ids"]}
    observed, models = {}, {}
    telemetry_config = None
    have_config = False
    for path in paths:
        row = score_trace(path, base)
        if have_config and row.get("telemetry_config") != telemetry_config:
            raise ValueError("Sensitivity runs use different telemetry configurations")
        telemetry_config = row.get("telemetry_config")
        have_config = True
        matches = [name for name, cfg in configurations.items() if cfg == row["rl_config"]]
        if len(matches) != 1:
            raise ValueError("Unexpected reward configuration")
        key = (matches[0], row["seed"], row["scenario_id"])
        training = row["training_protocol"]
        if (key not in expected or key in observed or row["manifest_sha256"] != plan["manifest_sha256"]
                or training["scenario_id"] != plan["training_scenario_id"]):
            raise ValueError("Unexpected, duplicate or mismatched sensitivity run")
        model_key = key[:2]
        if models.setdefault(model_key, row["model_weights_sha256"]) != row["model_weights_sha256"]:
            raise ValueError("Model changed between test scenarios")
        observed[key] = row
    if set(observed) != expected or not expected:
        raise ValueError("Incomplete sensitivity matrix")
    summaries = []
    metrics = ("mean_common_reference_reward", "mean_common_reference_cost_index",
               "selected_delay_ms", "selected_loss_pct", "selected_residual_mbps", "lte_fraction",
               "within_episode_switch_fraction")
    for name in configurations:
        for scenario in plan["test_scenario_ids"]:
            rows = [observed[(name, seed, scenario)] for seed in plan["seeds"]]
            stats = {}
            for metric in metrics:
                values = [r[metric] for r in rows if r[metric] is not None]
                stats[metric] = {"n": len(values), "mean": fmean(values) if values else None,
                                 "sample_sd": stdev(values) if len(values) > 1 else None}
            summaries.append({"id": name, "scenario_id": scenario, "metrics": stats})
    return {"plan": plan, "summaries": summaries, "runs": list(observed.values()),
            "interpretation": "Common-reference rescore of observed trajectories. "
                              "Not actual economic cost or a counterfactual network outcome."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Reference RLConfig JSON (optional)")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("offline")
    sweep.add_argument("--snapshots", required=True)
    sweep.add_argument("--split", choices=["calibration"], required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--train-id", required=True)
    plan.add_argument("--test-ids", nargs="+", required=True)
    plan.add_argument("--seeds", nargs="+", type=int, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--plan", required=True)
    evaluate.add_argument("--traces", nargs="+", required=True)
    args = parser.parse_args()
    base = RLConfig()
    if args.config:
        with open(args.config, encoding="utf-8") as stream:
            base = RLConfig(**json.load(stream))
    if args.command == "offline":
        with open(args.snapshots, encoding="utf-8") as stream:
            result = offline(json.load(stream), base)
    elif args.command == "plan":
        result = make_plan(base, read_manifest(args.manifest), args.train_id, args.test_ids, args.seeds)
    else:
        if args.config:
            parser.error("Evaluation reference comes from the predeclared plan")
        with open(args.plan, encoding="utf-8") as stream:
            result = policy_report(args.traces, json.load(stream))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
