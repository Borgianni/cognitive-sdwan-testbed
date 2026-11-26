# TNSM revision: Reviewer 1, Comment 4

A fixed deterministic greedy policy cannot select different actions for
identical observations. During online learning, weights and exploration
can change the selected action even if the state is aliased. A switch in
the original training trajectory is not evidence of state-aware routing.

## Diagnostic protocol

Train a separate checkpoint for each state encoding. Use the same training
conditions and seeds, and do not tune on held-out observations. Evaluate
with zero epsilon, no replay insertion, no optimizer updates, and verify
the model-weight SHA-256 before/after. Compare paired nominal and impaired
observations using each fixed checkpoint. Report state equality, Q-values,
LTE-minus-LEO Q-margin, greedy action, and desired switch fraction across
pairs/seeds; report failures as well. A different observation is necessary
but not sufficient for a state-conditioned switch. A synthetic pair is a
sanity check, not evidence from Mininet.

From the framework directory (controller and network dependencies required):

```bash
python -m unittest discover -s tests -v
sudo python -m sdwan_rl.runners --mode online --checkpoint revised.keras --trace train.jsonl
sudo python -m sdwan_rl.runners --mode frozen --checkpoint revised.keras --trace eval.jsonl
python -m sdwan_rl.policy_audit --checkpoint revised.keras --snapshots docs/c04_synthetic_snapshots.json
```

For a legacy observation comparator train a NEW checkpoint with
`--lte-delay-threshold-ms 700 --sat-delay-threshold-ms 1200 --lte-loss-threshold-pct 0 --sat-loss-threshold-pct 0 --bandwidth-state-mode legacy_throughput`
and supply the same flags when evaluating it. The saved configuration is
checked on load. This approximates the public original observation on
valid non-boundary measurements; the original commit remains the exact reference.

The online run writes a new .keras file and metadata sidecar; frozen mode
requires both and refuses mismatched configs. Traces log the metrics used
to build the decision state (previous tick), pre-update Q-values, action,
greedy action, exploration flag, post-decision measurement, reward and
next state. Tick zero has no decision telemetry after reset and should be
excluded from paired analysis. The completion record verifies frozen weights;
a missing completion record indicates incomplete evaluation. Use actual
decision_metrics dictionaries from held-out traces in the paired JSON
file instead of synthetic values for empirical evidence.

The commands above exercise the frozen mode on the existing schedule only;
they are NOT a held-out protocol. Randomized held-out scenarios and seeds
are planned for C09-C12. The runner's actual forwarding, probe steering,
timing and reward attribution still require validation; a policy decision
does not establish that user traffic followed that tunnel. No network
performance, convergence, or proactive switching claim follows from the
diagnostic alone. The policy probe unit tests use a deliberately constructed
fake model; the separate TensorFlow tests exercise freeze and checkpoint
roundtrip when TensorFlow is installed.
