"""Separate bounded scalability study; generate scenarios or run one replicate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

from .agents.uniform_random import UniformRandomAgent
from .config import RLConfig
from .dqn_settings import CURRENT_DQN
from .policy_audit import append_trace, weights_digest, metadata_path, validate_run_seed
from .scale_cases import CASES, SPLITS, make_scenario, validate_scenario, digest
from .scale_policy import PathEnvironment, policy_signature, load_scale, save_scale
from .runtime_environment import runtime_environment
from .telemetry_core import TelemetryConfig
from .telemetry_cli import add_telemetry_arguments, telemetry_kwargs


def read_scenario(path):
    with open(path, encoding="utf-8") as stream:
        return validate_scenario(json.load(stream))


from .network_guard import exclusive_run, start_network, stop_network


@exclusive_run
def run(scenario, agent_type, seed, trace, cfg, telemetry_cfg, checkpoint=None,
        secondary="off", fixed_action=None):
    validate_scenario(scenario)
    validate_run_seed(seed, required=True)
    split, graph = scenario["split"], scenario["graph"]
    if agent_type not in ("dqn", "greedy", "random", "fixed"):
        raise ValueError("Unknown agent")
    if cfg.dqn_version != CURRENT_DQN or telemetry_cfg.observation_lag_ticks or telemetry_cfg.carry_observations_across_episodes:
        raise ValueError("Use current DQN, no lag/history stress in scalability runs")
    if split == "train" and agent_type != "dqn":
        raise ValueError("Only DQN trains")
    if agent_type == "dqn" and (not checkpoint or split == "calibration"):
        raise ValueError("DQN needs a train or frozen held-out checkpoint")
    if agent_type != "dqn" and checkpoint is not None:
        raise ValueError("Non-learning comparators do not use checkpoints")
    if secondary != "off" and (agent_type != "fixed" or split != "calibration"):
        raise ValueError("Secondary packet-order study requires fixed-action calibration")
    if agent_type == "fixed":
        if split != "calibration" or type(fixed_action) is not int or not 0 <= fixed_action < len(graph["paths"]):
            raise ValueError("Fixed policy needs calibration and a valid action")
    elif fixed_action is not None:
        raise ValueError("Fixed action belongs only to fixed policy")
    if split == "calibration" and agent_type != "fixed":
        raise ValueError("Scalability calibration uses fixed paths")
    if split == "train" and (Path(checkpoint).suffix != ".keras" or Path(checkpoint).exists()
                              or metadata_path(checkpoint).exists()):
        raise ValueError("Use a new .keras checkpoint path")

    env = PathEnvironment(graph, cfg)
    agent = None
    training_meta = None
    initial = None
    if agent_type == "dqn":
        from .agents.dqn import DQNAgent
        agent = DQNAgent.from_config(cfg, state_size=3*env.n, action_size=env.n, seed=seed)
        if split == "held_out_test":
            training_meta = load_scale(agent, scenario, cfg, telemetry_cfg, seed, checkpoint)
        initial = weights_digest(agent)
    elif agent_type == "random":
        agent = UniformRandomAgent(seed=seed, action_size=env.n)
    signature = policy_signature(scenario, cfg, telemetry_cfg)
    run_id = uuid.uuid4().hex
    append_trace(trace, {"kind": "scale_run", "run_id": run_id, "scenario": scenario,
                        "scenario_sha256": digest(scenario), "signature": signature,
                        "agent": agent_type, "seed": seed, "secondary": secondary,
                        "fixed_action": fixed_action, "initial_weights_sha256": initial,
                        "training_metadata": training_meta, "checkpoint": checkpoint,
                        "runtime_environment": runtime_environment()}, mode="x")
    from .scale_runtime import create_network, ScaleAPI, ScaleRuntime

    net, runtime = None, None
    api = ScaleAPI(scenario, secondary)
    begin = time.monotonic()
    ep = tick = None
    measurement = None
    try:
        net = create_network(graph)
        start_network(net)
        api.wait_ready()
        runtime = ScaleRuntime(net, scenario, telemetry_cfg, secondary, api)
        for ep in range(1, scenario["episodes"] + 1):
            tick = None
            setup = runtime.apply_episode(ep)
            env.reset()
            previous, last_action = None, 0
            for tick in range(scenario["ticks_per_episode"]):
                measurement = None
                start = time.monotonic()
                state = env.state()
                q, greedy_values, update = None, None, None
                exploratory, epsilon = False, 0.0
                if agent_type == "dqn":
                    epsilon = agent.eps
                    action, q, exploratory = agent.traced_action(state)
                elif agent_type == "greedy":
                    action, greedy_values = env.greedy(last_action)
                elif agent_type == "random":
                    action, _ = agent.decide(previous)
                else:
                    action = fixed_action
                action = int(action)
                inference = time.monotonic() - start
                phase = time.monotonic()
                ack = api.apply(action, f"{run_id}:{ep}:{tick}")
                acknowledgement = time.monotonic() - phase
                phase = time.monotonic()
                if ep == 1 and tick == 0:
                    runtime.start_background()
                background_setup = time.monotonic() - phase
                phase = time.monotonic()
                measurement = runtime.collect(action)
                collection = time.monotonic() - phase
                phase = time.monotonic()
                env.update(measurement["metrics"])
                # Fixed-action diagnostic needs primary QoS, not a scalar reward
                # when a selected probe has no valid RTT.
                try:
                    terms = env.terms(action)
                except ValueError:
                    if agent_type != "fixed":
                        raise
                    terms = None
                next_state = env.state()
                done = tick == scenario["ticks_per_episode"] - 1
                if agent_type == "dqn" and split == "train":
                    agent.remember(state, action, terms["reward"], next_state, done)
                    update = agent.train_step()
                reward_learning = time.monotonic() - phase
                append_trace(trace, {"kind": "scale_decision", "episode": ep, "tick": tick,
                                    "state": state.tolist(), "decision_metrics": previous,
                                    "q_values": q.tolist() if q is not None else None,
                                    "greedy_values": greedy_values, "action": action,
                                    "epsilon": epsilon, "exploratory": bool(exploratory),
                                    "controller_ack": ack, "measurement": measurement,
                                    "reward_terms": terms, "next_state": next_state.tolist(),
                                    "terminal": done, "dqn_update": update,
                                    "episode_setup": setup if tick == 0 else None,
                                    "timing_s": {"encode_and_decide": inference,
                                                 "controller_ack": acknowledgement,
                                                 "background_start": background_setup,
                                                 "collection": collection,
                                                 "reward_and_learning": reward_learning,
                                                 "tick_work": time.monotonic() - start}})
                previous, last_action = measurement["metrics"], action
        if agent_type == "dqn" and split == "train":
            save_scale(agent, scenario, cfg, telemetry_cfg, seed, checkpoint)
        final = weights_digest(agent) if agent_type == "dqn" else None
        if agent_type == "dqn" and split == "held_out_test":
            if initial != final or agent.eps != 0 or len(agent.memory) or agent.updates:
                raise RuntimeError("Frozen model changed")
        append_trace(trace, {"kind": "complete", "weights_sha256": final,
                             "training_state": agent.training_state() if agent_type == "dqn" else None,
                             "elapsed_s": time.monotonic() - begin})
    except Exception as exc:
        append_trace(trace, {"kind": "failed", "episode": ep, "tick": tick,
                             "error": str(exc), "measurement": measurement})
        raise
    finally:
        try:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("scenario")
    gen.add_argument("--case", choices=CASES, required=True)
    gen.add_argument("--split", choices=SPLITS, required=True)
    gen.add_argument("--episodes", type=int, default=12)
    gen.add_argument("--ticks", type=int, default=10)
    gen.add_argument("--output", required=True)
    exe = sub.add_parser("run")
    exe.add_argument("--scenario", required=True)
    exe.add_argument("--agent", choices=["dqn", "greedy", "random", "fixed"], required=True)
    exe.add_argument("--seed", type=int, required=True)
    exe.add_argument("--trace", required=True)
    exe.add_argument("--checkpoint")
    exe.add_argument("--rl-config", help="Explicit common RLConfig JSON; default continuous normalized")
    exe.add_argument("--secondary", choices=["off", "pinned", "rr"], default="off")
    exe.add_argument("--fixed-action", type=int)
    add_telemetry_arguments(exe)
    args = parser.parse_args()
    if args.command == "scenario":
        scenario = make_scenario(args.case, args.split, args.episodes, args.ticks)
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(scenario, stream, indent=2)
            stream.write("\n")
        return
    cfg = RLConfig(state_representation="continuous")
    if args.rl_config:
        with open(args.rl_config, encoding="utf-8") as stream:
            cfg = RLConfig(**json.load(stream))
    run(read_scenario(args.scenario), args.agent, args.seed, args.trace, cfg,
        TelemetryConfig(**telemetry_kwargs(args)), args.checkpoint, args.secondary, args.fixed_action)


if __name__ == "__main__":
    main()
