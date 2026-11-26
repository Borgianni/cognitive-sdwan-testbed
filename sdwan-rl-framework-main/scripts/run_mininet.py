import argparse
from sdwan_rl.runners import run_experiment
from sdwan_rl.config import RLConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-type", type=str, default="dqn", choices=["dqn", "random"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--lte-loss-threshold-pct", type=float, default=10.0,
                        help="LTE packet-loss observation threshold in percent")
    parser.add_argument("--sat-loss-threshold-pct", type=float, default=10.0,
                        help="LEO packet-loss observation threshold in percent")
    args = parser.parse_args()
    
    run_experiment(
        agent_type=args.agent_type,
        episodes=args.episodes,
        rl_cfg=RLConfig(lte_loss_threshold_pct=args.lte_loss_threshold_pct,
                        sat_loss_threshold_pct=args.sat_loss_threshold_pct),
        start_cli=args.cli
    )