from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.cli import CLI
from mininet.clean import cleanup

import time
import argparse
from typing import Dict, Tuple
import random
from dataclasses import asdict
from pathlib import Path

from .policy_audit import append_trace, load_frozen, save_checkpoint, weights_digest, metadata_path, validate_run_seed
from .experiment_protocol import read_manifest, resolve_run
from .scenario_runtime import ScenarioRuntime
from .scenario_profiles import nominal_profiles

from .topology import SDWANTopo
from .metrics import measure_link_tuple
from .controller_api import ControllerAPI
from .env import SDWANEnv, A_LTE, A_SAT
from .config import ExperimentConfig, RLConfig
from .agents.ewma import EWMALatencyAgent


# ----------------------------------------------------------------------
# Simple Random Agent (definizione inline per semplicità)
# ----------------------------------------------------------------------
class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.name = "Random"

    def act(self, state):
        # state non viene usato, sceglie uniformemente a caso
        return random.randrange(self.action_size)

    def remember(self, *args, **kwargs):
        pass

    def train_step(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


# ----------------------------------------------------------------------
# Degradation handling
# ----------------------------------------------------------------------
def set_degradation(net, episode: int, ranges: Tuple[Tuple[int, int], ...]):
    s1, s3 = net.get('s1', 's3')
    degraded = any(lo <= episode <= hi for lo, hi in ranges)
    if degraded:
        s1.cmd("tc qdisc change dev s1-eth1 root netem delay 150ms loss 20%")
        s3.cmd("tc qdisc change dev s3-eth1 root netem delay 150ms loss 20%")
    else:
        s1.cmd("tc qdisc change dev s1-eth1 root netem delay 25ms loss 2%")
        s3.cmd("tc qdisc change dev s3-eth1 root netem delay 25ms loss 2%")


# ----------------------------------------------------------------------
# Experiment runner with configurable agent
# ----------------------------------------------------------------------
def run_experiment(agent_type='dqn', exp=None, rl_cfg=None, start_cli=False, episodes=None,
                   mode='online', checkpoint=None, trace=None,
                   split='diagnostic', manifest_path=None, scenario_id=None, seed=None,
                   ewma_alpha=0.3):
    """
    agent_type: 'dqn', 'random' or 'ewma'
    """
    manifest = read_manifest(manifest_path) if manifest_path is not None else None
    protocol = resolve_run(manifest, scenario_id, split, mode)
    scenario = None
    validate_run_seed(seed, required=split != "diagnostic")
    if agent_type == "ewma" and mode != "heuristic":
        raise ValueError("EWMA requires --mode heuristic")
    if mode == "heuristic" and agent_type != "ewma":
        raise ValueError("Heuristic mode currently supports EWMA")
    if split != "diagnostic":
        if not trace or (agent_type == "dqn" and not checkpoint) or agent_type not in ("dqn", "ewma"):
            raise ValueError("Formal runs require a trace and DQN checkpoint or non-learning EWMA")
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
    episode_count = exp.episodes if episodes is None else episodes
    use_v2 = scenario is not None and "events" in scenario
    if use_v2:
        nominal = nominal_profiles()
        if (rl_cfg.lte_capacity_mbps != nominal["lte"]["capacity_mbps"]
                or rl_cfg.sat_capacity_mbps != nominal["satellite"]["capacity_mbps"]):
            raise ValueError("RL capacities must match the v2 topology")
    if mode not in ("online", "frozen", "heuristic"):
        raise ValueError("Unknown experiment mode")
    if mode == "frozen" and (agent_type != "dqn" or not checkpoint or not trace):
        raise ValueError("Frozen evaluation requires DQN, --checkpoint and --trace")
    if checkpoint and agent_type != "dqn":
        raise ValueError("Checkpoints are supported only for DQN")
    if trace and agent_type not in ("dqn", "ewma"):
        raise ValueError("Policy traces are supported for DQN and EWMA")
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
    env = SDWANEnv(cfg=rl_cfg)

    # Seed before constructing a fresh agent. Scenario-generation RNG is separate.
    if seed is not None:
        if agent_type == "dqn":
            import tensorflow as tf
            tf.keras.utils.set_random_seed(seed)
        else:
            random.seed(seed)
    print(f"[INFO] Run seed: {seed}")

    # --- Agent instantiation ---
    if agent_type == 'dqn':
        from .agents.dqn import DQNAgent
        agent = DQNAgent(state_size=6, action_size=2,
                         lr=rl_cfg.lr, gamma=rl_cfg.gamma,
                         replay_size=rl_cfg.replay_size, batch_size=rl_cfg.batch_size,
                         eps_start=rl_cfg.eps_start, eps_end=rl_cfg.eps_end, eps_decay=rl_cfg.eps_decay)
        print("[INFO] Using DQN agent")
    elif agent_type == 'random':
        agent = RandomAgent(action_size=2)
        print("[INFO] Using Random agent (no learning)")
    elif agent_type == "ewma":
        agent = EWMALatencyAgent(ewma_alpha)
        print(f"[INFO] Using latency-only EWMA (alpha={agent.alpha})")
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    initial_weights = None
    training_meta = None
    if mode == "frozen":
        training_meta = load_frozen(agent, rl_cfg, checkpoint, protocol=protocol, run_seed=seed)
        initial_weights = weights_digest(agent)
    if trace:
        append_trace(trace, {"kind": "run", "mode": mode, "checkpoint": checkpoint,
                            "initial_weights_sha256": weights_digest(agent) if agent_type == "dqn" else None,
                            "rl_config": asdict(rl_cfg), "experiment_config": asdict(exp),
                            "episode_count": episode_count, "protocol": protocol,
                            "agent_type": agent_type, "run_seed": seed,
                            "heuristic_config": {"alpha": agent.alpha, "reset_each_episode": True}
                            if agent_type == "ewma" else None,
                            "training_protocol": training_meta.get("training_protocol") if training_meta else None,
                            "training_seed": training_meta.get("training_seed") if training_meta else None}, mode="x")
    topo = SDWANTopo()
    net = Mininet(topo=topo, switch=OVSSwitch, controller=None, autoStaticArp=True)
    api = ControllerAPI()

    runtime = None
    try:
        net.start()
        if use_v2:
            runtime = ScenarioRuntime(net, scenario)
        for ep in range(1, episode_count + 1):
            conditions = None
            if runtime is not None:
                conditions = runtime.apply_episode(ep)
                if ep == 1:
                    runtime.start_traffic()
            else:
                set_degradation(net, ep, exp.degrade_lte_ranges)
            env.reset_metrics()
            state = env.make_state().vec
            decision_metrics = None
            if agent_type == "ewma":
                agent.reset()

            for t in range(exp.ticks_per_episode):
                # 1) Agent selects action
                heuristic_details = None
                if agent_type == "ewma":
                    action, heuristic_details = agent.decide(decision_metrics)
                    eps_before, q_values, exploratory = 0.0, None, False
                    greedy_action = None  # no Q-function
                elif trace:
                    eps_before = agent.eps
                    action, q_values, exploratory = agent.traced_action(state)
                    greedy_action = int(q_values.argmax())
                else:
                    action = agent.act(state)  # 0 = LTE, 1 = Satellite

                # 2) Measure both links
                lte_d, lte_thr, lte_loss = measure_link_tuple(net, "lte")
                sat_d, sat_thr, sat_loss = measure_link_tuple(net, "satellite")
                traffic_counters = runtime.check_traffic() if runtime is not None else None

                metrics = {
                    "lte": {"delay": lte_d, "throughput": lte_thr, "packet_loss": lte_loss},
                    "satellite": {"delay": sat_d, "throughput": sat_thr, "packet_loss": sat_loss},
                    "episode": {"episode number": ep, "tick": t, "chosen link": int(action)}
                }

                # 3) Send to controller (optional, for compatibility)
                api.send_metrics(metrics)
                chosen = api.get_link_decision()  # not actually used for forwarding here

                # 4) Update environment and compute reward
                env.update_from_metrics(metrics)
                r = env.reward(action)
                done = (t == exp.ticks_per_episode - 1)

                # 5) Next state
                next_state = env.make_state().vec

                # 6) Experience replay & training (only for DQN)
                if trace:
                    append_trace(trace, {
                        "kind": "decision", "episode": ep, "tick": t,
                        "decision_metrics": decision_metrics,
                        "state": state.tolist(),
                        "q_values": q_values.tolist() if q_values is not None else None,
                        "greedy_action": greedy_action, "action": int(action),
                        "heuristic_details": heuristic_details,
                        "exploratory": bool(exploratory), "epsilon": eps_before,
                        "measurement_metrics": metrics, "reward": r,
                        "next_state": next_state.tolist(),
                        "controller_reported_choice": chosen,
                        "scenario_conditions": conditions,
                        "background_traffic_counters": traffic_counters,
                    })
                if agent_type == 'dqn' and mode == "online":
                    agent.remember(state, action, r, next_state, done)
                    agent.train_step()

                decision_metrics = metrics
                state = next_state

            # end ticks
            # Optional: print episode summary
            if agent_type == 'random':
                print(f"[Episode {ep}] Random agent done (no learning)")
        # end episodes

        if mode == "frozen":
            if weights_digest(agent) != initial_weights or agent.eps != 0 or len(agent.memory):
                raise RuntimeError("Frozen evaluation changed weights, epsilon or replay")
        if mode == "online" and checkpoint:
            save_checkpoint(agent, rl_cfg, checkpoint, protocol=protocol, run_seed=seed)
        if trace:
            append_trace(trace, {"kind": "complete",
                                 "weights_sha256": weights_digest(agent) if agent_type == "dqn" else None,
                                 "epsilon": agent.eps if agent_type == "dqn" else 0.0,
                                 "replay_size": len(agent.memory) if agent_type == "dqn" else 0})
        if start_cli:
            print("Entering Mininet CLI (type 'exit' to quit)...")
            CLI(net)

    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            net.stop()
            cleanup()


# ----------------------------------------------------------------------
# Command‑line entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SD-WAN RL emulation")
    parser.add_argument("--agent-type", type=str, default="dqn", choices=["dqn", "random", "ewma"],
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
                      continuous_delay_scale_ms=args.continuous_delay_scale_ms)
    run_experiment(agent_type=args.agent_type, exp=exp_cfg, rl_cfg=rl_cfg, start_cli=args.cli,
                   episodes=args.episodes, mode=args.mode, checkpoint=args.checkpoint, trace=args.trace,
                   split=args.split, manifest_path=args.manifest, scenario_id=args.scenario_id,
                   seed=args.seed, ewma_alpha=args.ewma_alpha)