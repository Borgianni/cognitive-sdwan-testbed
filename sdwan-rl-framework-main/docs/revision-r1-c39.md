# R1-C39: second learner through the same runner

Expected SARSA with affine linear Q(s,a) is an alternative to DQN:
epsilon-greedy, on-policy expected TD backup, one immediate
semi-gradient update per transition, no replay or target network.
It accepts the same binary or normalized continuous observation,
discrete actions, reward and REST/data-plane measurement loop.
Small seeded uniform initial weights, alpha=.01 (configurable),
gamma and epsilon from RLConfig; epsilon decays after each update
(starting with first transition), not DQN's replay warm-up.
This is an engineering reference, not a tuned state-of-the-art
competitor or a proposed novel RL method. Different capacity and
sample-use efficiency are inherent algorithm differences.
Use equal decision budgets, same locked config and seeds, and fresh
train/frozen checkpoints. Keep comparison exploratory and limited
to a predeclared subset of held-out scenarios. Do not change the
C35 four-test family. NPZ checkpoint stores parameters only, bound
to algorithm/settings/seed/protocol/telemetry config with digests.
DQN continues to use Keras checkpoint and prior compatibility rules.
No claim of arbitrary agent or continuous-action support.
