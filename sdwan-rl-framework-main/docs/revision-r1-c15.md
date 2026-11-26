# TNSM revision: Reviewer 1, Comment 15

Uniform random action selection is a non-adaptive stochastic reference,
not a mathematical worst-case or lower performance bound. For fixed
snapshot utilities u_1,...,u_N its expected utility is their arithmetic
mean, which can exceed the utility of a deterministic poor choice.
This is not a prediction of random policy performance on another
network trajectory.

Keep Random in the benchmark: it quantifies performance without
telemetry-driven selection. Replace the original inline runner agent
with UniformRandomAgent, using a private seeded Python RNG, uniformly
sampling each decision including cold starts, never reseeding at episode
boundaries. It ignores telemetry and is not epsilon-greedy exploration
(the trace records explicit action probabilities and the uniform policy;
epsilon=0 and exploratory=false indicate absence of a DQN exploration
mechanism, not deterministic behavior).

Formal calibration/test runs accept Random in heuristic mode with the
same manifest, seed, measurements, reward and trace conventions as
EWMA and Greedy Utility. No training/checkpoint. The original diagnostic
--agent-type random (default online mode) is retained as a compatibility
alias for heuristic. The old separate random_agents.py class remains
unused by the runner.

The shared C14 audit now replays EWMA, Greedy Utility and seeded Random
decisions on logged decision-time telemetry, validates configuration and
rejects inconsistent traces. The expected uniform utility on each
valid paired snapshot is reported separately from realized logged reward.
It is an analytical snapshot reference only, not a lower bound.

Optional paired benchmark summary requires a predeclared seed list and
agents, every test scenario in the same manifest, constant per-agent
configuration and shared RL configuration except observation parameters.
DQN must use the same model for all test scenarios of each seed. Compare
one DQN representation at a time; use separate matched reports for
binary and continuous variants. The C12 report additionally links each
frozen trace to its training trace. Report mean and sample SD across
runs by scenario, not pooled ticks as independent replications.
The summary is descriptive; confidence intervals and paired inference
belong to C35-C37. It does not verify physical forwarding or eliminate
stochastic network differences.

Examples for later validation:
```bash
sudo python -m sdwan_rl.runners --agent-type random --mode heuristic --split held_out_test --manifest randomized_manifest.json --scenario-id test_joint_r001 --seed 1 --trace random_test_1.jsonl
python -m sdwan_rl.benchmark_audit --traces random_test_1.jsonl --output random_audit.json
python -m unittest discover -s tests -p test_uniform_random.py -v
# For a complete matrix, pass all expected trace paths and:
# --paired-seeds 1 2 3 4 5 --paired-agents dqn ewma greedy_utility random
```

Manuscript (deferred): p. 6 Section III-D text "random agent serving
as a lower performance bound"; p. 9 impairment-window results phrase
"representing the worst-case performance bound" (including hyphenated
line break). Replace with "uniform random reference policy" and describe
relative empirical outcomes only. Audit other lower/worst bound claims
when rewriting the full benchmarking block. No need to run an
intentionally poor policy to prove the reviewer's point. No experiments
or manuscript edits performed.
