# TNSM revision: Reviewer 1, Comment 6

## Observation model

Distinguish latent network condition x_t, measured telemetry o_t, and
the agent input phi(o_t). The binary encoder is many-to-one. Even current
continuous telemetry does not establish the Markov property: queue states,
traffic dynamics, observation delays and other variables remain unobserved.
The feed-forward DQN is an observation-conditioned policy, not a solution
to a fully specified POMDP or a belief-state controller.

The new continuous mode uses six float32 inputs in the original order:
LTE/LEO RTT quality D_ref/(D_ref+D), LTE/LEO loss quality 1-L/100,
LTE/LEO estimated residual fraction clip((C-T)/C,0,1).
D_ref=100 ms is a common provisional scaling reference, not an SLA or
reward parameter; the binary thresholds do not affect the continuous
features. Reward, architecture and action space are unchanged. Invalid
numerical features and reset inputs map to zero conservatively; upstream
zero fallbacks can still mask failed measurements. Clipping and finite
precision may also alias values. A history/trend encoder is deferred until
timestamped and validated telemetry is available; the reviewer requests it
only as a possible extension. No claim of full observability is made.

## Controlled comparison (commands for later validation)

From the framework directory, train a separate DQN for each encoder
using matched seeds, training scenarios and budget. Evaluate each with
the same encoding and held-out conditions. The full checkpoint config is
checked, including representation and scale, and older C04/C05 metadata
is explicitly treated as binary.

```bash
# Original public-code observation approximation
sudo python -m sdwan_rl.runners --mode online --checkpoint legacy.keras --trace legacy_train.jsonl --state-representation binary --lte-delay-threshold-ms 700 --sat-delay-threshold-ms 1200 --lte-loss-threshold-pct 0 --sat-loss-threshold-pct 0 --bandwidth-state-mode legacy_throughput

# Revised threshold binary
sudo python -m sdwan_rl.runners --mode online --checkpoint binary.keras --trace binary_train.jsonl --state-representation binary

# Continuous
sudo python -m sdwan_rl.runners --mode online --checkpoint continuous.keras --trace continuous_train.jsonl --state-representation continuous --continuous-delay-scale-ms 100
```

The original code commit is the exact historical reference. The
legacy_paper bandwidth predicate is also available. The default stays
binary to preserve C01-C05 behavior. These commands are not yet a
held-out benchmark; C09-C12 and forwarding/telemetry fixes are pending.

The C05 aliasing audit now compares four encodings on identical snapshots.
Exact float32 observation grouping gives an empirical hindsight gap; in
continuous mode many singleton groups can mechanically yield zero. Report
singleton fraction and do not equate a zero gap with better policy
performance, generalization, or full observability. Use actual held-out
policy action/utility and network outcomes, not the gap alone, to judge the
encoders. The C05 counterexample is a synthetic sanity check only.

```bash
python -m unittest discover -s tests -v
python -m sdwan_rl.aliasing_audit --snapshots docs/c05_synthetic_snapshots.json
```

## Manuscript (deferred)

Revise the state definition and MDP interpretation in the state-model
discussion, Fig. 2 and experimental methodology. Explain the observation
map, partial observability, normalization, comparison protocol, and
limitations. Only claim empirical gains after evaluation.
