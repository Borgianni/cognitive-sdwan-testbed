"""Summarize independent online learning runs and separate frozen test runs."""
import argparse
import json
import math
import re
from pathlib import Path
from statistics import fmean, stdev

from .config import RLConfig
from .experiment_protocol import resolve_run, validate_training_provenance
from .policy_audit import validate_run_seed


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def stats(values):
    values = list(values)
    return {"n": len(values), "mean": fmean(values),
            "sample_sd": stdev(values) if len(values) > 1 else None}


def read_run(path, split):
    """Accept only complete, seeded, formal DQN traces with every expected tick."""
    if split not in ("train", "held_out_test"):
        raise ValueError("Use train or held_out_test traces")
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < 3 or rows[0].get("kind") != "run" or rows[-1].get("kind") != "complete":
        raise ValueError(f"{path}: incomplete trace")
    header, end = rows[0], rows[-1]
    mode = "online" if split == "train" else "frozen"
    if header.get("mode") != mode or header.get("agent_type") != "dqn":
        raise ValueError(f"{path}: incorrect mode or agent")
    seed = header.get("run_seed")
    validate_run_seed(seed, required=True)
    protocol = header.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("split") != split:
        raise ValueError(f"{path}: incorrect split")
    if resolve_run(protocol["manifest"], protocol["scenario_id"], split, mode) != protocol:
        raise ValueError(f"{path}: invalid protocol provenance")
    # Validate the recorded RL configuration, including observation semantics.
    cfg = RLConfig.from_record(header["rl_config"])
    scenario = protocol["scenario"]
    episodes, ticks = scenario["episodes"], scenario["ticks_per_episode"]
    if header.get("episode_count") != episodes or len(rows) != episodes * ticks + 2:
        raise ValueError(f"{path}: trace does not match the declared budget")
    initial, final = header.get("initial_weights_sha256"), end.get("weights_sha256")
    if not is_hash(initial) or not is_hash(final):
        raise ValueError(f"{path}: missing weight digest")
    if split == "held_out_test":
        validate_training_provenance(header.get("training_protocol"), protocol)
        if (type(header.get("training_seed")) is not int or header["training_seed"] != seed
                or initial != final or end.get("epsilon") != 0 or end.get("replay_size") != 0):
            raise ValueError(f"{path}: invalid frozen policy provenance")

    summaries = []
    for ep in range(1, episodes + 1):
        batch = rows[1 + (ep - 1) * ticks:1 + ep * ticks]
        rewards, epsilons, exploration = [], [], []
        for tick, row in enumerate(batch):
            if (row.get("kind") != "decision" or row.get("episode") != ep
                    or row.get("tick") != tick):
                raise ValueError(f"{path}: missing, duplicate or unordered decision")
            r, eps, exploratory = row.get("reward"), row.get("epsilon"), row.get("exploratory")
            if (not finite(r) or not finite(eps) or not 0 <= eps <= 1
                    or type(exploratory) is not bool):
                raise ValueError(f"{path}: invalid reward or exploration record")
            action, greedy = row.get("action"), row.get("greedy_action")
            q = row.get("q_values")
            if (type(action) is not int or action not in (0, 1)
                    or type(greedy) is not int or greedy not in (0, 1)
                    or not isinstance(q, list) or len(q) != 2 or not all(finite(v) for v in q)
                    or greedy != (0 if q[0] >= q[1] else 1)
                    or (not exploratory and action != greedy)):
                raise ValueError(f"{path}: inconsistent policy decision")
            if split == "held_out_test" and (eps != 0 or exploratory):
                raise ValueError(f"{path}: exploration during frozen evaluation")
            rewards.append(r)
            epsilons.append(eps)
            exploration.append(exploratory)
        summaries.append({"episode": ep, "total_reward": math.fsum(rewards),
                          "mean_reward": fmean(rewards), "mean_epsilon": fmean(epsilons),
                          "exploration_fraction": fmean(exploration)})
    return {"path": str(path), "seed": seed, "protocol": protocol,
            "rl_config": header["rl_config"], "initial_weights_sha256": initial,
            "final_weights_sha256": final, "training_protocol": header.get("training_protocol"),
            "episodes": summaries, "mean_reward": fmean(e["mean_reward"] for e in summaries)}


def report(training_paths, evaluation_paths=()):
    training = sorted((read_run(p, "train") for p in training_paths), key=lambda r: r["seed"])
    if len(training) < 2 or len({r["seed"] for r in training}) != len(training):
        raise ValueError("At least two distinct training seeds are required")
    ref = training[0]
    if any(r["protocol"] != ref["protocol"] or r["rl_config"] != ref["rl_config"]
           for r in training):
        raise ValueError("Training curves require the same scenario and RL configuration")
    # Distinct seeds are necessary; equal initial digests suggest accidental reuse.
    if len({r["initial_weights_sha256"] for r in training}) != len(training):
        raise ValueError("Training runs share initial weights; check independent initialization")
    curves = []
    for i, episode in enumerate(ref["episodes"]):
        curves.append({"episode": episode["episode"], **{
            key: stats(r["episodes"][i][key] for r in training)
            for key in ("total_reward", "mean_reward", "mean_epsilon", "exploration_fraction")
        }})
    result = {"training": {"seeds": [r["seed"] for r in training],
                           "scenario_id": ref["protocol"]["scenario_id"],
                           "manifest_sha256": ref["protocol"]["manifest_sha256"],
                           "rl_config": ref["rl_config"], "curves": curves,
                           "individual_runs": training},
              "held_out": None,
              "interpretation": "Online learning dynamics, not a convergence proof. "
                                "Sample SD describes variation across runs, not confidence intervals."}
    evaluations = [read_run(p, "held_out_test") for p in evaluation_paths]
    if not evaluations:
        return result
    by_seed = {r["seed"]: r for r in training}
    test_ids = {r["id"] for r in ref["protocol"]["manifest"]["scenarios"]
                if r["split"] == "held_out_test"}
    expected = {(seed, scenario) for seed in by_seed for scenario in test_ids}
    actual = set()
    for r in evaluations:
        key = (r["seed"], r["protocol"]["scenario_id"])
        if key in actual or key not in expected:
            raise ValueError("Duplicate or unexpected frozen evaluation")
        actual.add(key)
        train = by_seed[r["seed"]]
        if (r["training_protocol"] != train["protocol"]
                or r["rl_config"] != train["rl_config"]
                or r["initial_weights_sha256"] != train["final_weights_sha256"]):
            raise ValueError("Frozen evaluation does not match its training trace")
    if actual != expected:
        raise ValueError("Incomplete paired seed/scenario test matrix")
    result["held_out"] = {
        "scenarios": [{"scenario_id": scenario,
                       "mean_tick_reward_across_seeds": stats(r["mean_reward"] for r in evaluations
                            if r["protocol"]["scenario_id"] == scenario)}
                      for scenario in sorted(test_ids)],
        "individual_runs": sorted(evaluations, key=lambda r: (r["protocol"]["scenario_id"], r["seed"])),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", nargs="+", required=True, help="Complete seeded training JSONL traces")
    parser.add_argument("--evaluation", nargs="*", default=[], help="Complete paired held-out test traces")
    parser.add_argument("--output", required=True, help="New JSON report; refuses overwrite")
    args = parser.parse_args()
    result = report(args.training, args.evaluation)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
