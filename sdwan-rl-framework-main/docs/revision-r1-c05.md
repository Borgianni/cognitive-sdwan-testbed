# TNSM revision: Reviewer 1, Comment 5

The continuous-valued reward does not recover information discarded by a
many-to-one observation map. For fixed weights, identical observations
induce identical greedy actions. Online updates can change action values,
but do not make the current aliased observation more informative.

This comment does not require a change to the reward, agent, or encoder.
Instead, the diagnostic measures decision-relevant aliasing using the
existing instantaneous utility on identical metric snapshots.

For a finite sample x_i, let u_i(a) = reward(x_i,a), and s_i = phi(x_i).
For each observed state group G_s, compute the best action with full
snapshot access and the best single fixed action with group-only access.
The sample-weighted aliasing gap is

    mean_i max_a u_i(a) - sum_s (|G_s|/n) max_a mean_{i in G_s} u_i(a).

It is nonnegative and measures the minimum average immediate-utility
regret of any deterministic observation-only policy on that sample
(relative to the snapshot oracle). A randomized observation-only policy
cannot improve the empirical expected utility over the best fixed action.
It is not a bound on discounted return or a measure of Shannon information.
The group-best action uses hindsight on the same sample: it is an
optimistic empirical comparator, not a trained or deployable policy.
Snapshot utility assumes telemetry is held fixed for both actions; it does
not estimate causal effects of a real forwarding change.

The report includes state groups, strict conflicting action preferences
(tolerance 1e-9 in reward units), group sizes, utility means, fraction of
samples in conflicting groups and the mean aliasing gap. Encodings are
compared on exactly the same snapshots and reward parameters:
legacy_code, legacy_paper and revised_binary. Legacy modes approximate the
old predicates on valid non-boundary observations (inclusive delay bounds
in the revised implementation). The original Git commit remains the
exact historical reference.

## Run

From the framework directory, without TensorFlow or Mininet:

```bash
python -m unittest discover -s tests -p test_aliasing_audit.py -v
python -m sdwan_rl.aliasing_audit --snapshots docs/c05_synthetic_snapshots.json
python -m sdwan_rl.aliasing_audit --trace eval.jsonl
```

The synthetic pair deliberately remains inside all revised thresholds,
yet changes the preferred instantaneous reward action. It is a mathematical
counterexample, not emulation data. For the default reward, the pair gives
an aliasing gap of approximately 1.3517 reward units. Use held-out measured
decision-time snapshots for empirical comparisons; report results per seed
and scenario before aggregating. The trace loader requires a completed
frozen trace with matching weight hashes, excludes reset ticks and rejects
invalid non-reset telemetry. A zero fallback from a failed probe cannot
always be detected from stored numerical telemetry alone. Frozen does not
imply held-out; scenario separation is still required.

## Manuscript work (deferred until the state-representation block is complete)

Original PDF, page 5, Fig. 2 caption ends in "optimize routing decisions
without information loss." Remove that claim and the unsupported
"guarantee stability and prevent rule oscillation" claim. Revise the
discussion of binarization and continuous rewards on pages 4-5.
Distinguish reward resolution from policy observability, and report
empirical results only after validation. The continuous encoding and
the partially observable interpretation belong to C06.
