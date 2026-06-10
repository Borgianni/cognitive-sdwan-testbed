from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.cli import CLI
from mininet.clean import cleanup

import time
import argparse
from typing import Dict, Tuple
import random

from .topology import SDWANTopo
from .metrics import measure_link_tuple
from .controller_api import ControllerAPI
from .env import SDWANEnv, A_LTE, A_SAT
from .config import ExperimentConfig, RLConfig
from .agents.dqn import DQNAgent


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
def run_experiment(agent_type='dqn', exp=None, rl_cfg=None, start_cli=False, episodes=None):
    """
    agent_type: 'dqn' or 'random'
    """
    topo = SDWANTopo()
    net = Mininet(topo=topo, switch=OVSSwitch, controller=None, autoStaticArp=True)
    net.start()
    api = ControllerAPI()
    env = SDWANEnv(cfg=rl_cfg)

    # --- Agent instantiation ---
    if agent_type == 'dqn':
        agent = DQNAgent(state_size=6, action_size=2,
                         lr=rl_cfg.lr, gamma=rl_cfg.gamma,
                         replay_size=rl_cfg.replay_size, batch_size=rl_cfg.batch_size,
                         eps_start=rl_cfg.eps_start, eps_end=rl_cfg.eps_end, eps_decay=rl_cfg.eps_decay)
        print("[INFO] Using DQN agent")
    elif agent_type == 'random':
        agent = RandomAgent(action_size=2)
        print("[INFO] Using Random agent (no learning)")
    else:
        raise ValueError(f"Unknown agent_type: {agent_type} (use 'dqn' or 'random')")

    try:
        for ep in range(1, exp.episodes + 1):
            set_degradation(net, ep, exp.degrade_lte_ranges)
            env.reset_metrics()
            state = env.make_state().vec

            for t in range(exp.ticks_per_episode):
                # 1) Agent selects action
                action = agent.act(state)  # 0 = LTE, 1 = Satellite

                # 2) Measure both links
                lte_d, lte_thr, lte_loss = measure_link_tuple(net, "lte")
                sat_d, sat_thr, sat_loss = measure_link_tuple(net, "satellite")

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
                if agent_type == 'dqn':
                    agent.remember(state, action, r, next_state, done)
                    agent.train_step()
                else:
                    # For Random agent, optionally log reward
                    pass

                state = next_state

            # end ticks
            # Optional: print episode summary
            if agent_type == 'random':
                print(f"[Episode {ep}] Random agent done (no learning)")
        # end episodes

        if start_cli:
            print("Entering Mininet CLI (type 'exit' to quit)...")
            CLI(net)

    finally:
        net.stop()
        cleanup()


# ----------------------------------------------------------------------
# Command‑line entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SD-WAN RL emulation")
    parser.add_argument("--agent-type", type=str, default="dqn", choices=["dqn", "random"],
                        help="Type of agent to use (dqn or random)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes (uses ExperimentConfig otherwise)")
    parser.add_argument("--cli", action="store_true", help="Start Mininet CLI after experiment")
    args = parser.parse_args()

    # Optional: override episode count from command line
    exp_cfg = ExperimentConfig()
    if args.episodes is not None:
        exp_cfg.episodes = args.episodes

    run_experiment(agent_type=args.agent_type, exp=exp_cfg, start_cli=args.cli)