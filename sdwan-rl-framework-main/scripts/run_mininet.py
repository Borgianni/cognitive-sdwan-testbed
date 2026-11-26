import argparse
from sdwan_rl.runners import run_experiment
from sdwan_rl.config import RLConfig
from sdwan_rl.reward_cli import add_reward_arguments, reward_kwargs
from sdwan_rl.dqn_settings import add_dqn_arguments, dqn_kwargs
from sdwan_rl.telemetry_core import TelemetryConfig
from sdwan_rl.telemetry_cli import add_telemetry_arguments, telemetry_kwargs
from sdwan_rl.config_files import add_config_files, resolve_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-type", type=str, default="dqn", choices=["dqn", "expected_sarsa", "random", "ewma", "greedy_utility"],
                        help="Random is a uniform stochastic reference, not a performance bound")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--cli", action="store_true")
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
    parser.add_argument("--sarsa-alpha", type=float, default=.01)
    add_reward_arguments(parser)
    add_dqn_arguments(parser)
    add_telemetry_arguments(parser)
    add_config_files(parser)
    args = parser.parse_args()
    
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
    rl_cfg, telemetry_cfg = resolve_configs(parser,args,rl_cfg,TelemetryConfig(**telemetry_kwargs(args)))
    run_experiment(
        agent_type=args.agent_type,
        episodes=args.episodes,
        rl_cfg=rl_cfg,
        mode=args.mode, checkpoint=args.checkpoint, trace=args.trace,
        split=args.split, manifest_path=args.manifest, scenario_id=args.scenario_id,
        seed=args.seed, ewma_alpha=args.ewma_alpha,
        telemetry_cfg=telemetry_cfg,
        start_cli=args.cli, sarsa_alpha=args.sarsa_alpha
    )