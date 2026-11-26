from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.cli import CLI
from mininet.clean import cleanup

import time
import argparse
from typing import Dict, Tuple
import random
import uuid
from dataclasses import asdict
from pathlib import Path

from .policy_audit import append_trace, load_frozen, save_checkpoint, weights_digest, metadata_path, validate_run_seed
from .experiment_protocol import read_manifest, resolve_run
from .scenario_runtime import ScenarioRuntime
from .scenario_profiles import nominal_profiles

from .topology import SDWANTopo
from .telemetry_core import TelemetryConfig, observation_age, observation_index
from .telemetry_cli import add_telemetry_arguments, telemetry_kwargs, runtime_scenario
from .telemetry_runtime import TelemetryCollector
from .controller_api import ControllerAPI
from .env import SDWANEnv, A_LTE, A_SAT
from .config import ExperimentConfig, RLConfig
from .reward_cli import add_reward_arguments, reward_kwargs
from .dqn_settings import LEGACY_DQN, add_dqn_arguments, dqn_kwargs
from .runtime_environment import runtime_environment
from .agents.ewma import EWMALatencyAgent
from .agents.utility import GreedyUtilityAgent

from .agents.uniform_random import UniformRandomAgent

HEURISTIC_AGENTS = ("ewma", "greedy_utility", "random")


# ----------------------------------------------------------------------
# Experiment runner with configurable agent
# ----------------------------------------------------------------------
def run_experiment(agent_type='dqn', exp=None, rl_cfg=None, start_cli=False, episodes=None,
                   mode='online', checkpoint=None, trace=None,
                   split='diagnostic', manifest_path=None, scenario_id=None, seed=None,
                   ewma_alpha=0.3, telemetry_cfg=None):
    """
    agent_type: 'dqn', 'random', 'ewma' or 'greedy_utility'
    """
    # Preserve the old diagnostic CLI; formal baselines must use heuristic mode.
    if agent_type == "random" and mode == "online" and split == "diagnostic":
        mode = "heuristic"
    manifest = read_manifest(manifest_path) if manifest_path is not None else None
    protocol = resolve_run(manifest, scenario_id, split, mode)
    scenario = None
    validate_run_seed(seed, required=split != "diagnostic")
    if agent_type in HEURISTIC_AGENTS and mode != "heuristic":
        raise ValueError("Non-learning comparators require --mode heuristic")
    if mode == "heuristic" and agent_type not in HEURISTIC_AGENTS:
        raise ValueError("Heuristic mode requires a non-learning comparator")
    if split != "diagnostic":
        if not trace or (agent_type == "dqn" and not checkpoint) or agent_type not in ("dqn", *HEURISTIC_AGENTS):
            raise ValueError("Formal runs require a trace and DQN checkpoint or non-learning comparator")
        if exp is not None or episodes is not None:
            raise ValueError("Formal runs use the manifest schedule; overrides are forbidden")
        scenario = protocol["scenario"]
        exp = ExperimentConfig(episodes=scenario["episodes"],
                               ticks_per_episode=scenario["ticks_per_episode"],
                               degrade_lte_ranges=tuple(map(tuple, scenario.get("degrade_lte_ranges", []))))
    else:
        exp = exp if exp is not None else ExperimentConfig()
    print(f"[INFO] Protocol split: {split}; scenario: {scenario_id or 'diagnostic'}")
    rl_cfg = rl_cfg if rl_cfg is not None else RLConfig()
    telemetry_cfg = telemetry_cfg if telemetry_cfg is not None else TelemetryConfig()
    if telemetry_cfg.observation_lag_ticks and (agent_type != 'dqn' or mode != 'frozen'):
        raise ValueError('Artificial telemetry lag is a frozen DQN diagnostic only')
    episode_count = exp.episodes if episodes is None else episodes
    nominal = nominal_profiles()
    if (rl_cfg.lte_capacity_mbps != nominal["lte"]["capacity_mbps"]
            or rl_cfg.sat_capacity_mbps != nominal["satellite"]["capacity_mbps"]):
        raise ValueError("RL capacities must match the topology")
    network_scenario = runtime_scenario(scenario, exp, episode_count)
    if mode not in ("online", "frozen", "heuristic"):
        raise ValueError("Unknown experiment mode")
    if mode == "frozen" and (agent_type != "dqn" or not checkpoint or not trace):
        raise ValueError("Frozen evaluation requires DQN, --checkpoint and --trace")
    if checkpoint and agent_type != "dqn":
        raise ValueError("Checkpoints are supported only for DQN")
    if trace and agent_type not in ("dqn", *HEURISTIC_AGENTS):
        raise ValueError("Policy traces require a supported policy")
    if mode == "online" and checkpoint:
        if Path(checkpoint).suffix != ".keras":
            raise ValueError("Use a .keras checkpoint filename")
        if Path(checkpoint).exists() or metadata_path(checkpoint).exists():
            raise FileExistsError(str(checkpoint))
    print(f"[INFO] Packet-loss thresholds: LTE={rl_cfg.lte_loss_threshold_pct}%, "
          f"LEO={rl_cfg.sat_loss_threshold_pct}%")
    print(f"[INFO] RTT thresholds: LTE={rl_cfg.lte_delay_threshold_ms} ms, "
          f"LEO={rl_cfg.sat_delay_threshold_ms} ms")
    print(f"[INFO] Bandwidth state: {rl_cfg.bandwidth_state_mode}; "
          f"residual thresholds LTE={rl_cfg.lte_residual_threshold}, "
          f"LEO={rl_cfg.sat_residual_threshold}")
    print(f"[INFO] State representation: {rl_cfg.state_representation}; "
          f"continuous RTT scale={rl_cfg.continuous_delay_scale_ms} ms")
    print(f"[INFO] Reward: {rl_cfg.reward_mode}; RTT reference={rl_cfg.threshold_delay_ms} ms; "
          f"bandwidth reference={rl_cfg.reward_bandwidth_ref_mbps} Mbps; "
          f"loss reference={rl_cfg.reward_loss_ref_fraction}; "
          f"weights(delay,loss,bw,cost)={(rl_cfg.w_b, rl_cfg.w_a, rl_cfg.w_c, rl_cfg.w_d)}; "
          f"cost indices={(rl_cfg.lte_cost_index, rl_cfg.sat_cost_index)}")
    env = SDWANEnv(cfg=rl_cfg)
    decision_env = SDWANEnv(cfg=rl_cfg)
    run_id = uuid.uuid4().hex

    # DQN seeds before model creation and owns separate exploration/replay RNGs.
    if seed is not None and agent_type != "dqn":
        random.seed(seed)
    print(f"[INFO] Run seed: {seed}")
    if agent_type == "dqn" and rl_cfg.dqn_version == LEGACY_DQN:
        if mode != "frozen" or split != "diagnostic":
            raise ValueError("Legacy DQN is only supported for frozen diagnostic inference")

    # --- Agent instantiation ---
    if agent_type == 'dqn':
        from .agents.dqn import DQNAgent
        agent = DQNAgent.from_config(rl_cfg, state_size=6, action_size=2, seed=seed)
        print(f"[INFO] Using DQN agent {rl_cfg.dqn_version}; hidden={agent.hidden_units}; "
              f"warm-up={rl_cfg.dqn_replay_warmup}; target interval={rl_cfg.dqn_target_update_steps}; "
              f"global clip norm={rl_cfg.dqn_gradient_clipnorm}")
    elif agent_type == 'random':
        agent = UniformRandomAgent(seed=seed, action_size=2)
        print("[INFO] Using Random agent (no learning)")
    elif agent_type == "ewma":
        agent = EWMALatencyAgent(ewma_alpha)
        print(f"[INFO] Using latency-only EWMA (alpha={agent.alpha})")
    elif agent_type == "greedy_utility":
        agent = GreedyUtilityAgent(rl_cfg)
        print("[INFO] Using causal greedy utility comparator")
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    initial_weights = None
    training_meta = None
    if mode == "frozen":
        training_meta = load_frozen(agent, rl_cfg, checkpoint, protocol=protocol, run_seed=seed,
                                    measurement_settings=telemetry_cfg.measurement_settings())
        initial_weights = weights_digest(agent)
    environment = runtime_environment()
    if trace:
        append_trace(trace, {"kind": "run", "mode": mode, "checkpoint": checkpoint,
                            "initial_weights_sha256": weights_digest(agent) if agent_type == "dqn" else None,
                            "rl_config": asdict(rl_cfg), "experiment_config": asdict(exp),
                            "episode_count": episode_count, "protocol": protocol,
                            "agent_type": agent_type, "run_seed": seed,
                            "runtime_environment": environment,
                            "telemetry_schema": 1, "telemetry_config": asdict(telemetry_cfg),
                            "run_id": run_id,
                            "dqn_initial_training_state": agent.training_state() if agent_type == "dqn" else None,
                            "heuristic_config": {"alpha": agent.alpha, "reset_each_episode": True}
                            if agent_type == "ewma" else
                            {"utility": "shared_env_reward", "reset_each_episode": True}
                            if agent_type == "greedy_utility" else
                            agent.configuration if agent_type == "random" else None,
                            "training_protocol": training_meta.get("training_protocol") if training_meta else None,
                            "training_seed": training_meta.get("training_seed") if training_meta else None}, mode="x")
    topo = SDWANTopo()
    net = Mininet(topo=topo, switch=OVSSwitch, controller=None, autoStaticArp=True)
    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
    api = ControllerAPI()
    runtime, collector = None, None
    last_trace_write_s, last_tick_start = 0.0, None
    run_start = time.monotonic()
    current_ep, current_tick = None, None
    latest_measurement = None
    try:
        net.start()
        api.wait_ready()
        runtime = ScenarioRuntime(net, network_scenario)
        collector = TelemetryCollector(net, telemetry_cfg)
        collector.install()
        for ep in range(1, episode_count + 1):
            current_ep, current_tick = ep, None
            setup_start = time.monotonic()
            conditions = runtime.apply_episode(ep)
            impairment_applied_s = time.monotonic()
            if ep == 1:
                runtime.start_traffic()
            env.reset_metrics()
            decision_env.reset_metrics()
            history, episode_reward = [], 0.0
            if agent_type in HEURISTIC_AGENTS:
                agent.reset()
            episode_setup = {"start_s": setup_start, "impairment_applied_s": impairment_applied_s,
                             "elapsed_s": time.monotonic() - setup_start}
            for t in range(exp.ticks_per_episode):
                current_tick = t
                latest_measurement = None
                tick_start = time.monotonic()
                phases = {}
                source_tick = observation_index(t, telemetry_cfg.observation_lag_ticks)
                decision_metrics = None if source_tick is None else history[source_tick]["metrics"]
                decision_windows = None if source_tick is None else history[source_tick]["windows"]
                if decision_metrics is None:
                    decision_env.reset_metrics()
                else:
                    decision_env.update_from_metrics(decision_metrics)
                state = decision_env.make_state().vec
                decision_s = time.monotonic()
                age = observation_age(decision_windows, decision_s)
                phases["observation_encoding_and_age_s"] = time.monotonic() - tick_start
                phase_start = time.monotonic()
                heuristic_details = None
                if agent_type in HEURISTIC_AGENTS:
                    action, heuristic_details = agent.decide(decision_metrics)
                    eps_before, q_values, exploratory, greedy_action = 0.0, None, False, None
                else:
                    eps_before = agent.eps
                    action, q_values, exploratory = agent.traced_action(state)
                    greedy_action = int(q_values.argmax())
                phases["policy_decision_s"] = time.monotonic() - phase_start
                phase_start = time.monotonic()
                decision_id = f"{run_id}:{ep}:{t}"
                ack = api.apply_decision(int(action), decision_id)
                acknowledged_s = time.monotonic()
                phases["rest_openflow_ack_s"] = acknowledged_s - phase_start
                phase_start = time.monotonic()
                latest_measurement = collector.collect(int(action))
                phases["telemetry_collection_and_forwarding_checks_s"] = time.monotonic() - phase_start
                phase_start = time.monotonic()
                traffic_counters = runtime.check_traffic()
                phases["background_traffic_checks_s"] = time.monotonic() - phase_start
                phase_start = time.monotonic()
                metrics = latest_measurement["metrics"]
                metrics["episode"] = {"episode number": ep, "tick": t, "chosen link": int(action),
                                      "decision_id": decision_id}
                env.update_from_metrics(metrics)
                terms = env.reward_terms(action)
                r = terms["reward"]
                episode_reward += r
                metrics["lte"]["cost_index"] = env.cost_index(A_LTE)
                metrics["satellite"]["cost_index"] = env.cost_index(A_SAT)
                metrics["episode"]["reward"] = r
                metrics["episode"]["total reward"] = episode_reward
                history.append({"metrics": metrics, "windows": latest_measurement["measurement_windows"]})
                done = t == exp.ticks_per_episode - 1
                next_source = observation_index(t + 1, telemetry_cfg.observation_lag_ticks)
                decision_env.update_from_metrics(history[next_source]["metrics"])
                next_state = decision_env.make_state().vec
                phases["reward_and_next_state_s"] = time.monotonic() - phase_start
                phase_start = time.monotonic()
                chosen = api.send_metrics(metrics)
                if chosen != ack["link"]:
                    raise RuntimeError("Controller log acknowledgement differs from applied decision")
                phases["rest_metrics_log_s"] = time.monotonic() - phase_start
                phase_start = time.monotonic()
                update = None
                if agent_type == "dqn" and mode == "online":
                    agent.remember(state, action, r, next_state, done)
                    update = agent.train_step()
                phases["learning_s"] = time.monotonic() - phase_start
                elapsed = time.monotonic() - tick_start
                phases["other_tick_work_s"] = max(0.0, elapsed - sum(phases.values()))
                if trace:
                    record = {
                        "kind": "decision", "episode": ep, "tick": t,
                        "decision_metrics": decision_metrics, "decision_source_tick": source_tick,
                        "effective_observation_lag_ticks": None if source_tick is None else t - 1 - source_tick,
                        "decision_time_s": decision_s, "decision_age": age,
                        "state": state.tolist(), "q_values": q_values.tolist() if q_values is not None else None,
                        "greedy_action": greedy_action, "action": int(action), "heuristic_details": heuristic_details,
                        "exploratory": bool(exploratory), "epsilon": eps_before,
                        "measurement_metrics": metrics, "reward": r, "reward_terms": terms,
                        "measurement_windows": latest_measurement["measurement_windows"],
                        "telemetry": latest_measurement, "controller_ack": ack,
                        "decision_acknowledged_s": acknowledged_s,
                        "dqn_update": update, "dqn_training_state": agent.training_state() if agent_type == "dqn" else None,
                        "terminal_transition": done, "next_state": next_state.tolist(),
                        "controller_reported_choice": chosen, "scenario_conditions": conditions,
                        "background_traffic_counters": traffic_counters,
                        "phase_timing_s": phases, "tick_start_s": tick_start, "tick_work_s": elapsed,
                        "previous_trace_write_s": last_trace_write_s,
                        "tick_start_interval_s": None if last_tick_start is None else tick_start - last_tick_start,
                        "episode_setup": episode_setup if t == 0 else None,
                    }
                    write_start = time.monotonic()
                    append_trace(trace, record)
                    last_trace_write_s = time.monotonic() - write_start
                last_tick_start = tick_start
        if mode == "frozen":
            if weights_digest(agent) != initial_weights or agent.eps != 0 or len(agent.memory):
                raise RuntimeError("Frozen evaluation changed weights, epsilon or replay")
        checkpoint_elapsed = None
        if mode == "online" and checkpoint:
            phase_start = time.monotonic()
            save_checkpoint(agent, rl_cfg, checkpoint, protocol=protocol, run_seed=seed,
                            measurement_settings=telemetry_cfg.measurement_settings())
            checkpoint_elapsed = time.monotonic() - phase_start
        if trace:
            append_trace(trace, {"kind": "complete",
                                 "weights_sha256": weights_digest(agent) if agent_type == "dqn" else None,
                                 "epsilon": agent.eps if agent_type == "dqn" else 0.0,
                                 "replay_size": len(agent.memory) if agent_type == "dqn" else 0,
                                 "dqn_final_training_state": agent.training_state() if agent_type == "dqn" else None,
                                 "last_trace_write_s": last_trace_write_s,
                                 "checkpoint_elapsed_s": checkpoint_elapsed,
                                 "run_elapsed_s": time.monotonic() - run_start})
        if start_cli:
            CLI(net)
    except Exception as exc:
        if trace:
            append_trace(trace, {"kind": "failed", "episode": current_ep, "tick": current_tick,
                                 "error": str(exc), "telemetry": latest_measurement,
                                 "run_elapsed_s": time.monotonic() - run_start})
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
            api.close()
            net.stop()
            cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SD-WAN RL emulation")
    parser.add_argument("--agent-type", type=str, default="dqn", choices=["dqn", "random", "ewma", "greedy_utility"],
                        help="Type of agent to use (dqn or random)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes (uses ExperimentConfig otherwise)")
    parser.add_argument("--cli", action="store_true", help="Start Mininet CLI after experiment")
    parser.add_argument("--lte-loss-threshold-pct", type=float, default=10.0,
                        help="LTE packet-loss observation threshold in percent")
    parser.add_argument("--sat-loss-threshold-pct", type=float, default=10.0,
                        help="LEO packet-loss observation threshold in percent")
    parser.add_argument("--lte-delay-threshold-ms", type=float,
                        default=RLConfig.lte_delay_threshold_ms,
                        help="LTE RTT observation threshold in milliseconds")
    parser.add_argument("--sat-delay-threshold-ms", type=float,
                        default=RLConfig.sat_delay_threshold_ms,
                        help="LEO RTT observation threshold in milliseconds")
    parser.add_argument("--bandwidth-state-mode",
                        choices=["residual_threshold", "legacy_throughput", "positive_residual"],
                        default=RLConfig.bandwidth_state_mode)
    parser.add_argument("--lte-residual-threshold", type=float,
                        default=RLConfig.lte_residual_threshold,
                        help="Minimum LTE residual capacity fraction (0, 1]")
    parser.add_argument("--sat-residual-threshold", type=float,
                        default=RLConfig.sat_residual_threshold,
                        help="Minimum LEO residual capacity fraction (0, 1]")
    parser.add_argument("--mode", choices=["online", "frozen", "heuristic"], default="online")
    parser.add_argument("--checkpoint", help="New .keras file in online mode; existing file in frozen mode")
    parser.add_argument("--trace", help="New JSONL policy trace file")
    parser.add_argument("--state-representation", choices=["binary", "continuous"],
                        default=RLConfig.state_representation)
    parser.add_argument("--continuous-delay-scale-ms", type=float,
                        default=RLConfig.continuous_delay_scale_ms,
                        help="Common RTT normalization scale in ms (continuous only)")
    parser.add_argument("--split", choices=["diagnostic", "train", "calibration", "held_out_test"],
                        default="diagnostic")
    parser.add_argument("--manifest", help="Formal scenario manifest JSON")
    parser.add_argument("--scenario-id", help="Scenario identifier in the manifest")
    parser.add_argument("--seed", type=int, help="Run/agent seed; required for formal splits")
    parser.add_argument("--ewma-alpha", type=float, default=0.3,
                        help="Weight on the newest RTT sample for EWMA")
    add_reward_arguments(parser)
    add_dqn_arguments(parser)
    add_telemetry_arguments(parser)
    args = parser.parse_args()

    # Optional: override episode count from command line
    exp_cfg = None if args.manifest else ExperimentConfig()

    rl_cfg = RLConfig(lte_loss_threshold_pct=args.lte_loss_threshold_pct,
                      sat_loss_threshold_pct=args.sat_loss_threshold_pct,
                      lte_delay_threshold_ms=args.lte_delay_threshold_ms,
                      sat_delay_threshold_ms=args.sat_delay_threshold_ms,
                      bandwidth_state_mode=args.bandwidth_state_mode,
                      lte_residual_threshold=args.lte_residual_threshold,
                      sat_residual_threshold=args.sat_residual_threshold,
                      state_representation=args.state_representation,
                      continuous_delay_scale_ms=args.continuous_delay_scale_ms,
                      **reward_kwargs(args), **dqn_kwargs(args))
    run_experiment(agent_type=args.agent_type, exp=exp_cfg, rl_cfg=rl_cfg, start_cli=args.cli,
                   episodes=args.episodes, mode=args.mode, checkpoint=args.checkpoint, trace=args.trace,
                   split=args.split, manifest_path=args.manifest, scenario_id=args.scenario_id,
                   seed=args.seed, ewma_alpha=args.ewma_alpha,
                   telemetry_cfg=TelemetryConfig(**telemetry_kwargs(args)))