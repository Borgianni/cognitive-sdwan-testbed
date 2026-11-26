"""Offline exhaustive one-step utility diagnostic; never a live oracle."""
import argparse
import json
import math
from pathlib import Path
from statistics import fmean, stdev

from .agents.utility import GreedyUtilityAgent, finite, snapshot_utilities, valid_tunnels
from .agents.ewma import EWMALatencyAgent
from .agents.uniform_random import UniformRandomAgent
from .config import RLConfig
from .cost_audit import cost_qos_summary
from .experiment_protocol import resolve_run, validate_training_provenance
from .policy_audit import validate_run_seed
from .telemetry_core import TelemetryConfig, observation_index, observation_age


def snapshot_gap(metrics, action, cfg):
    if type(action) is not int or action not in (0, 1):
        raise ValueError("Unknown action")
    if not all(valid_tunnels(metrics)):
        return None
    scores = snapshot_utilities(metrics, cfg)
    best = max(scores)
    return {"utility_by_action": list(scores), "selected_utility": scores[action],
            "oracle_utility": best, "gap": best - scores[action],
            "expected_uniform_utility": fmean(scores),
            "matches_oracle": math.isclose(scores[action], best, rel_tol=0, abs_tol=1e-9)}


def summarize(rows):
    available = [r for r in rows if r is not None]
    return {"valid_paired_snapshots": len(available),
            "unavailable_paired_snapshots": len(rows) - len(available),
            "mean_selected_utility": fmean(r["selected_utility"] for r in available) if available else None,
            "mean_oracle_utility": fmean(r["oracle_utility"] for r in available) if available else None,
            "mean_snapshot_gap": fmean(r["gap"] for r in available) if available else None,
            "mean_expected_uniform_utility": fmean(r["expected_uniform_utility"] for r in available) if available else None,
            "oracle_match_fraction": fmean(r["matches_oracle"] for r in available) if available else None}


def audit_trace(path):
    """Check a complete formal held-out trace and report both timing views."""
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < 3 or rows[0].get("kind") != "run" or rows[-1].get("kind") != "complete":
        raise ValueError("Incomplete trace")
    header, end = rows[0], rows[-1]
    agent = header.get("agent_type")
    mode = "frozen" if agent == "dqn" else "heuristic"
    if agent not in ("dqn", "ewma", "greedy_utility", "random") or header.get("mode") != mode:
        raise ValueError("Unsupported evaluation policy or mode")
    seed = header.get("run_seed")
    validate_run_seed(seed, required=True)
    protocol = header.get("protocol") or {}
    if protocol.get("split") != "held_out_test":
        raise ValueError("Expected a formal held-out test trace")
    if resolve_run(protocol["manifest"], protocol["scenario_id"], "held_out_test", mode) != protocol:
        raise ValueError("Invalid scenario provenance")
    cfg = RLConfig.from_record(header["rl_config"])
    telemetry_cfg = TelemetryConfig(**header["telemetry_config"]) if header.get("telemetry_schema") == 1 else None
    episodes, ticks = (protocol["scenario"][k] for k in ("episodes", "ticks_per_episode"))
    if header.get("episode_count") != episodes or len(rows) != episodes * ticks + 2:
        raise ValueError("Trace does not match declared tick budget")
    initial = header.get("initial_weights_sha256")
    final = end.get("weights_sha256")
    if agent == "dqn":
        validate_training_provenance(header.get("training_protocol"), protocol)
        if (type(header.get("training_seed")) is not int or header["training_seed"] != seed
                or not isinstance(initial, str) or len(initial) != 64 or initial != final):
            raise ValueError("Invalid frozen checkpoint provenance")
    elif (initial is not None or final is not None or header.get("checkpoint") is not None
          or header.get("training_protocol") is not None or header.get("training_seed") is not None):
        raise ValueError("Non-learning policy must not claim model weights or training")
    if end.get("epsilon") != 0 or end.get("replay_size") != 0:
        raise ValueError("Unexpected learning state during evaluation")

    decision, measured, rewards = [], [], []
    replay = None
    heuristic_config = header.get("heuristic_config")
    if agent == "greedy_utility":
        replay = GreedyUtilityAgent(cfg)
        if heuristic_config != {"utility": "shared_env_reward", "reset_each_episode": True}:
            raise ValueError("Invalid greedy configuration")
    elif agent == "ewma":
        if not isinstance(heuristic_config, dict) or set(heuristic_config) != {"alpha", "reset_each_episode"}:
            raise ValueError("Missing EWMA configuration")
        if heuristic_config["reset_each_episode"] is not True:
            raise ValueError("Unexpected EWMA reset semantics")
        replay = EWMALatencyAgent(heuristic_config["alpha"])
    elif agent == "random":
        replay = UniformRandomAgent(seed=seed, action_size=2)
        if heuristic_config != replay.configuration:
            raise ValueError("Invalid uniform random configuration")
    for index, row in enumerate(rows[1:-1]):
        ep, tick = index // ticks + 1, index % ticks
        if (row.get("kind") != "decision" or row.get("episode") != ep or row.get("tick") != tick
                or row.get("epsilon") != 0 or row.get("exploratory") is not False):
            raise ValueError("Invalid evaluation decision record")
        action, reward = row.get("action"), row.get("reward")
        if type(action) is not int or action not in (0, 1) or not finite(reward):
            raise ValueError("Invalid action or logged reward")
        previous = row["decision_metrics"]
        current = row["measurement_metrics"]
        if tick == 0 and previous is not None:
            raise ValueError("Expected reset decision observation")
        source_tick = observation_index(tick, telemetry_cfg.observation_lag_ticks if telemetry_cfg else 0)
        if source_tick is not None:
            source = rows[1 + (ep - 1) * ticks + source_tick]
            if previous != source["measurement_metrics"]:
                raise ValueError("Decision telemetry must match the declared past tick")
        if telemetry_cfg is not None:
            expected_age = None if source_tick is None else observation_age(
                source["measurement_windows"], row["decision_time_s"])
            if (row.get("decision_source_tick") != source_tick or row.get("decision_age") != expected_age):
                raise ValueError("Decision source or telemetry age differs from recorded windows")
            if row.get("controller_ack", {}).get("action") != action or row["controller_ack"].get("barrier_acknowledged") is not True:
                raise ValueError("Missing acknowledged routing action")
            if row["decision_acknowledged_s"] > row["telemetry"]["window_start_s"]:
                raise ValueError("Measurement started before the routing decision was acknowledged")
        if agent == "dqn":
            q = row.get("q_values")
            if (not isinstance(q, list) or len(q) != 2 or not all(finite(v) for v in q)
                    or action != (0 if q[0] >= q[1] else 1)
                    or row.get("greedy_action") != action):
                raise ValueError("Frozen DQN did not choose the logged greedy action")
        elif row.get("q_values") is not None or row.get("greedy_action") is not None:
            raise ValueError("Non-learning comparator must not claim Q-values")
        if replay is not None:
            if tick == 0:
                replay.reset()
            expected_action, details = replay.decide(previous)
            if expected_action != action or details != row.get("heuristic_details"):
                raise ValueError("Comparator action differs from replayed policy decision")
        decision.append(snapshot_gap(previous, action, cfg))
        post = snapshot_gap(current, action, cfg)
        if post is not None and not math.isclose(post["selected_utility"], reward, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Logged reward differs from shared snapshot utility")
        measured.append(post)
        rewards.append(reward)
    return {"path": str(path), "agent": agent, "seed": seed,
            "scenario_id": protocol["scenario_id"], "manifest_sha256": protocol["manifest_sha256"],
            "rl_config": header["rl_config"], "heuristic_config": heuristic_config,
            "telemetry_config": header.get("telemetry_config"),
            "test_scenario_ids": sorted(r["id"] for r in protocol["manifest"]["scenarios"]
                                        if r["split"] == "held_out_test"),
            "model_weights_sha256": initial,
            "training_protocol": header.get("training_protocol"),
            "ticks": len(rewards), "mean_logged_reward_all_ticks": fmean(rewards),
            "cost_qos": cost_qos_summary(rows[1:-1], cfg),
            "cost_qos_by_episode": [{"episode": ep, **cost_qos_summary(
                rows[1 + (ep - 1) * ticks:1 + ep * ticks], cfg)}
                for ep in range(1, episodes + 1)],
            "decision_time_snapshot": summarize(decision),
            "subsequent_measurement_snapshot": summarize(measured),
            "interpretation": "Paired snapshot diagnostics; subsequent measurements are hindsight only. "
                              "No counterfactual network trajectory, return bound or forwarding proof."}


# Compare one DQN configuration at a time; observation parameters may differ
# between the DQN and non-learning comparators. All remaining RL fields
# must match, including any future reward parameters added to RLConfig.
OBSERVATION_FIELDS = {"state_representation", "continuous_delay_scale_ms",
                      "lte_delay_threshold_ms", "sat_delay_threshold_ms",
                      "lte_loss_threshold_pct", "sat_loss_threshold_pct",
                      "bandwidth_state_mode", "lte_residual_threshold",
                      "sat_residual_threshold"}
BENCHMARK_AGENTS = ("dqn", "ewma", "greedy_utility", "random")


def paired_report(reports, seeds, agents=BENCHMARK_AGENTS):
    """Require every declared agent/seed/test scenario; no silent omissions."""
    reports, seeds, agents = list(reports), list(seeds), list(agents)
    if not reports or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Declare nonempty unique replicate seeds")
    for seed in seeds:
        validate_run_seed(seed, required=True)
    if not agents or len(set(agents)) != len(agents) or not set(agents) <= set(BENCHMARK_AGENTS):
        raise ValueError("Declare unique supported agents")
    ref = reports[0]
    scenarios = ref["test_scenario_ids"]
    if not scenarios:
        raise ValueError("Manifest has no test scenarios")
    expected = {(agent, seed, scenario) for agent in agents for seed in seeds for scenario in scenarios}
    observed = {}
    policy_configs, models = {}, {}
    common_cfg = lambda r: {k: v for k, v in r["rl_config"].items() if k not in OBSERVATION_FIELDS}
    for row in reports:
        key = (row["agent"], row["seed"], row["scenario_id"])
        if key not in expected or key in observed:
            raise ValueError("Unexpected or duplicate benchmark result")
        if (row["manifest_sha256"] != ref["manifest_sha256"]
                or row["test_scenario_ids"] != scenarios or common_cfg(row) != common_cfg(ref)
                or row.get("telemetry_config") != ref.get("telemetry_config")):
            raise ValueError("Benchmark runs do not share scenario manifest and common configuration")
        config = (row["rl_config"], row["heuristic_config"])
        if policy_configs.setdefault(row["agent"], config) != config:
            raise ValueError("Policy configuration changes within benchmark")
        if row["agent"] == "dqn":
            model = (row["model_weights_sha256"], row["training_protocol"])
            if models.setdefault(row["seed"], model) != model:
                raise ValueError("DQN checkpoint changes between scenarios for a replicate")
        observed[key] = row
    if set(observed) != expected:
        raise ValueError("Incomplete declared benchmark matrix")
    summaries = []
    for scenario in scenarios:
        for agent in agents:
            runs = [observed[(agent, seed, scenario)] for seed in seeds]
            values = [r["mean_logged_reward_all_ticks"] for r in runs]
            cost_qos_stats = {}
            for metric in ("reward_to_cost_ratio", "mean_selected_cost_index",
                           "mean_qos_utility_valid_selected", "mean_selected_rtt_ms",
                           "mean_selected_loss_pct", "mean_selected_residual_mbps",
                           "lte_selection_fraction"):
                numbers = [r["cost_qos"][metric] for r in runs if r["cost_qos"][metric] is not None]
                cost_qos_stats[metric] = {
                    "n_seeds": len(numbers), "mean": fmean(numbers) if numbers else None,
                    "sample_sd": stdev(numbers) if len(numbers) > 1 else None}
            summaries.append({"scenario_id": scenario, "agent": agent, "n_seeds": len(seeds),
                              "mean_logged_reward": fmean(values),
                              "sample_sd_across_seeds": stdev(values) if len(values) > 1 else None,
                              "cost_qos_across_seeds": cost_qos_stats})
    return {"manifest_sha256": ref["manifest_sha256"], "seeds": seeds,
            "agents": agents, "scenario_ids": scenarios, "summaries": summaries,
            "interpretation": "Descriptive run-level summaries only; no significance claims. "
                              "Matching seeds does not guarantee identical network randomness."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="New report JSON")
    parser.add_argument("--paired-seeds", nargs="+", type=int, help="Predeclared seeds for complete matrix")
    parser.add_argument("--paired-agents", nargs="+", choices=BENCHMARK_AGENTS,
                        default=list(BENCHMARK_AGENTS))
    args = parser.parse_args()
    reports = [audit_trace(path) for path in args.traces]
    result = {"runs": reports}
    if args.paired_seeds is not None:
        result["paired"] = paired_report(reports, args.paired_seeds, args.paired_agents)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
