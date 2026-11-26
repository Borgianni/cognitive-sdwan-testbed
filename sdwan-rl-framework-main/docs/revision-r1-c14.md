# TNSM revision: Reviewer 1, Comment 14

Retain latency-only EWMA (C13) and add one strong non-learning
multi-metric comparator: greedy utility maximization. Avoid redundant
heuristics with independent weights/thresholds. The scientific question
is whether learned long-horizon decisions help over the same instantaneous
objective, not whether DQN always wins. If actions have little effect
on future network state, greedy may be equally good or better.

The causal greedy controller evaluates both actions on the same previous
tick telemetry passed to DQN (before encoding), using SDWANEnv.reward
directly. It selects max utility, ties favor LTE. It has no separate
reward parameters or learning. C16-C20 reward changes automatically
affect it and the shared snapshot audit. It receives full-resolution
telemetry, not just the DQN's binary features. Continuous DQN is the
closer information-matched comparator; even that comparison is subject
to normalization/clipping and fixed cost inputs. At reset no telemetry
is available: choose LTE; if only one tunnel has valid telemetry select
it without claiming a paired utility, and if neither is valid hold the
last action. The existing upstream zero fallbacks for loss/throughput
are not detectable from numeric samples alone.

The exhaustive snapshot oracle is OFFLINE only. It scores both actions
on identical observed metrics. Two diagnostics are reported: decision-time
utility gap and subsequent-measurement hindsight gap. Neither oracle
uses future observations in the live policy. A zero decision-time gap
is expected for greedy on valid paired snapshots; its hindsight gap can
be positive after conditions change. Report valid/unavailable counts;
do not impute missing comparisons as zero. The oracle is an upper
reference on the same snapshot, not on trajectories, discounted returns,
causal forwarding effects or independently collected runs. Different
policies may alter the telemetry they subsequently encounter.

The benchmark audit checks complete held-out traces, schedule and seed
provenance, tick order, previous-tick observation linkage, frozen greedy
Q decisions, shared utility/reward agreement on valid paired samples,
and the causal greedy action. It retains all-tick logged reward
separately from valid-only oracle statistics. Data-plane correctness,
probe validity and paired experimental aggregation remain to be validated.

Commands for later use:
```bash
sudo python -m sdwan_rl.runners --agent-type greedy_utility --mode heuristic --split held_out_test --manifest randomized_manifest.json --scenario-id test_joint_r001 --seed 1 --trace greedy_test_1.jsonl
python -m sdwan_rl.benchmark_audit --traces greedy_test_1.jsonl --output greedy_audit.json
python -m unittest discover -s tests -p test_utility_baseline.py -v
```

No experiments launched. Page 7 Table II and strategy description,
and pp. 8-9 benchmarking, will be revised after validation. DQN, reward,
and state encoders remain unchanged. Claims of advantage over greedy
must follow evidence, not be assumed.
