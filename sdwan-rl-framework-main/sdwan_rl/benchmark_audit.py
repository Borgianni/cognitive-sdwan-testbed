"""Offline exhaustive one-step utility diagnostic; never a live oracle."""
import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from .agents.utility import GreedyUtilityAgent, finite, snapshot_utilities, valid_tunnels
from .config import RLConfig
from .experiment_protocol import resolve_run, validate_training_provenance
from .policy_audit import validate_run_seed


def snapshot_gap(metrics, action, cfg):
    if type(action) is not int or action not in (0, 1):
        raise ValueError("Unknown action")
    if not all(valid_tunnels(metrics)):
        return None
    scores = snapshot_utilities(metrics, cfg)
    best = max(scores)
    return {"utility_by_action": list(scores), "selected_utility": scores[action],
            "oracle_utility": best, "gap": best - scores[action],
            "matches_oracle": math.isclose(scores[action], best, rel_tol=0, abs_tol=1e-9)}


def summarize(rows):
    available = [r for r in rows if r is not None]
    return {"valid_paired_snapshots": len(available),
            "unavailable_paired_snapshots": len(rows) - len(available),
            "mean_selected_utility": fmean(r["selected_utility"] for r in available) if available else None,
            "mean_oracle_utility": fmean(r["oracle_utility"] for r in available) if available else None,
            "mean_snapshot_gap": fmean(r["gap"] for r in available) if available else None,
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
    if agent not in ("dqn", "ewma", "greedy_utility") or header.get("mode") != mode:
        raise ValueError("Unsupported evaluation policy or mode")
    seed = header.get("run_seed")
    validate_run_seed(seed, required=True)
    protocol = header.get("protocol") or {}
    if protocol.get("split") != "held_out_test":
        raise ValueError("Expected a formal held-out test trace")
    if resolve_run(protocol["manifest"], protocol["scenario_id"], "held_out_test", mode) != protocol:
        raise ValueError("Invalid scenario provenance")
    cfg = RLConfig(**header["rl_config"])
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
    greedy = GreedyUtilityAgent(cfg) if agent == "greedy_utility" else None
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
        if tick != 0 and previous != rows[index]["measurement_metrics"]:
            raise ValueError("Decision telemetry must match the preceding tick")
        if agent == "dqn":
            q = row.get("q_values")
            if (not isinstance(q, list) or len(q) != 2 or not all(finite(v) for v in q)
                    or action != (0 if q[0] >= q[1] else 1)
                    or row.get("greedy_action") != action):
                raise ValueError("Frozen DQN did not choose the logged greedy action")
        elif row.get("q_values") is not None or row.get("greedy_action") is not None:
            raise ValueError("Non-learning comparator must not claim Q-values")
        if greedy is not None:
            if tick == 0:
                greedy.reset()
            expected_action, details = greedy.decide(previous)
            if expected_action != action or details != row.get("heuristic_details"):
                raise ValueError("Greedy action differs from causal utility decision")
        decision.append(snapshot_gap(previous, action, cfg))
        post = snapshot_gap(current, action, cfg)
        if post is not None and not math.isclose(post["selected_utility"], reward, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Logged reward differs from shared snapshot utility")
        measured.append(post)
        rewards.append(reward)
    return {"path": str(path), "agent": agent, "seed": seed,
            "scenario_id": protocol["scenario_id"], "manifest_sha256": protocol["manifest_sha256"],
            "rl_config": header["rl_config"], "heuristic_config": header.get("heuristic_config"),
            "ticks": len(rewards), "mean_logged_reward_all_ticks": fmean(rewards),
            "decision_time_snapshot": summarize(decision),
            "subsequent_measurement_snapshot": summarize(measured),
            "interpretation": "Paired snapshot diagnostics; subsequent measurements are hindsight only. "
                              "No counterfactual network trajectory, return bound or forwarding proof."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="New report JSON")
    args = parser.parse_args()
    reports = [audit_trace(path) for path in args.traces]
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump({"runs": reports}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
