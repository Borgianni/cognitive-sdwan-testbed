# TNSM revision: Reviewer 1, Comments 21-22

## Reconcile manuscript and implementation, not claim historical reproduction

Submitted p.5 describes 6-64-32-2 (ReLU, ReLU, linear); p.6
Algorithm 1 describes online/target networks and minibatch MSE.
The inspected original public implementation actually used 24/24
hidden layers, one online network for bootstrapping, and separate
fit calls for each sampled transition. We cannot establish which
implementation produced the submitted plots. New runs must be
reported as revised experiments, with commit and configuration.

Keep vanilla DQN, not Double/Dueling DQN or prioritized replay.
The revised target_batch_v1 implementation matches the stated
architecture (2,594 trainable online parameters). Explicit settings:
- Dense kernels Glorot uniform, biases zero; target initially copied
  from the online network, never trained by gradient descent.
- gamma=.95, Adam learning rate 1e-4, batch 32, replay capacity 10000.
- Warm-up 32 transitions, including the first update at size 32.
- Uniform sampling without replacement; one optimizer update per
  eligible decision tick. Selected-action TD MSE, not full-output MSE.
- Hard target copy every 100 successful optimizer updates (not
  episodes, and not wall-clock seconds). Initialization excluded.
- Global gradient norm clipping at 10, applied by Adam. Log raw norm
  and TD MSE. These new choices are engineering defaults, not tuned
  optimal values or historical experimental facts.
- Epsilon 1 to .01, multiplied by .995 after each successful
  optimizer update, not during warm-up. Specify this in Algorithm 1.
- FIFO replay is retained across episodes; runner marks the final
  tick terminal, so its TD target does not bootstrap across reset.
  This is an artificial episode boundary, not a physical network outage.
- Seed before model creation; tf.keras.utils.set_random_seed seeds
  Python, NumPy, TensorFlow; independent private RNG streams are
  derived for exploration and uniform replay. Scenario seeds remain
  separate. Optional --dqn-deterministic-ops requires a seed; it
  requests deterministic TF ops, not deterministic network emulation.
  Default off until compatibility/timing validated; record the setting.
- Trace includes update counters, replay size, epsilon, terminal flag,
  loss, raw norm, target synchronization, actual installed package
  versions/Python/platform and TF device list. Checkpoints retain
  training counters/settings and versions; they are for frozen
  inference, not resumable training (no replay/RNG/target state saved).
- Old records lacking DQN fields are labelled legacy_online_v0.
  Old 24/24 checkpoints can be inspected in diagnostic frozen mode
  with matching reward/observation configuration, not relabelled as
  revised models. For exact historical training use the old commit.
  Loaded network architecture is checked against the declared version.

Versions are collected at runtime, not fabricated from requirements.
The original requirements are broad. After validating a real setup,
archive pip freeze --all, Python version, Linux kernel, OVS, tc,
iperf and Mininet/Ryu versions with the run artifacts. Do not claim
a pinned tested environment until this is done.

## C22: discounting is not DQN convergence

For a stationary fully observed discounted MDP with bounded rewards,
the Bellman optimality operator is a gamma-contraction in the
sup norm for gamma<1. This does not establish convergence of
nonlinear neural approximation with off-policy replay and bootstrapping.
Our network observations may be partially observable and the environment
is time-varying; do not imply the contraction proof applies directly.
Target network and clipping are stabilization mechanisms, not proofs.
C12 learning curves across seeds and held-out frozen evaluation are
the evidence, with update diagnostics as supporting information.
No convergence-based stopping criterion or guarantee is introduced.
Keep the online adaptation loop.

## Validation / manuscript later

python -m unittest discover -s tests -p test_dqn_core.py -v
python -m unittest discover -s tests -p test_dqn_training.py -v
python -m unittest discover -s tests -p test_frozen_dqn.py -v

The latter two need real TensorFlow. No Mininet runs in this change.
Page 5 network architecture description; p.6 Algorithm 1 (initial
telemetry/reset, warm-up, selected-action minibatch loss, hard update
cadence, epsilon schedule, terminal boundary), paragraph ending
"ensuring convergence"; p.8 hyperparameter setup and Fig.4 discussion.
Revise in blue after integration and measurements. Preserve the
framework contribution; do not claim a new DQN algorithm.
