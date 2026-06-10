import argparse
from sdwan_rl.runners import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-type", type=str, default="dqn", choices=["dqn", "random"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--cli", action="store_true")
    args = parser.parse_args()
    
    run_experiment(
        agent_type=args.agent_type,
        episodes=args.episodes,
        start_cli=args.cli
    )