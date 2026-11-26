# TNSM revision: Reviewer 1, Comment 11

Generate concrete v2 scenarios before any training or testing. For each
template event, draw an integer duration from a declared inclusive
range and an integer onset from the feasible interval given that
duration. The end is inclusive (onset + duration - 1). Reject layouts
that violate guard intervals, chronological same-tunnel ordering,
minimum nominal gaps, template cross-tunnel overlap relationships, or
duplicate overall impairment timing. The accepted joint distribution
is conditional on these constraints; it is not independent uniform
onsets and durations after rejection. Bound the attempts and fail
explicitly if the design cannot be satisfied.

The generator keeps the severity, affected tunnel, traffic vector,
episode budget and ticks per episode unchanged. It uses separate
scenario seeds for train, calibration and held-out test, and per-template
replica RNG streams derived using SHA-256. These are schedule seeds,
not seeds for DQN initialization, replay sampling, Linux packet loss,
or traffic timing. Those require separate reproducibility controls.
The training and calibration schedules are not affected by the test
seed. All algorithms must receive the same generated manifest and
predeclared scenario assignments; do not regenerate per algorithm
or after observing test performance.

Temporal collision checks ignore scenario names, severity, load and
tunnel, comparing the union of impaired episode intervals at the same
episode horizon. This excludes exact timing reuse even between different
impairment profiles. It does not ensure that each individual onset or
duration is disjoint or guarantee statistical independence. Different
horizons are separate schedules. No-event controls must be specified
separately. The original fixed example schedules are also excluded.

Provisional settings: training duration 5-15 episodes; calibration
and held-out duration 3-9 episodes; at least two nominal episodes at
both boundaries; at least one nominal episode between events on the
same tunnel. Default replica counts are 1 training, 1 calibration and
3 per test template (17 scenarios total for the C10 example). These are
software design defaults, not a final run budget or validated times.
The full-episode time varies with actual measurement/processing duration.
The current runner trains one model on one training scenario per run;
pairing training replicates and evaluation cases is deferred to C12.

Example commands FOR LATER, from the framework directory:

```bash
python -m sdwan_rl.scenario_generation --template docs/c10_example_manifest.json --output randomized_manifest.json --receipt randomized_receipt.json
python -m sdwan_rl.experiment_protocol --manifest randomized_manifest.json
python -m unittest discover -s tests -p test_scenario_generation.py -v
```

The receipt records the generator version, template and generated
manifest hashes, split seeds, duration bounds, guards, attempts and
individual scenario hashes. The manifest is compatible with C09/C10;
its actual scenario schedule is recorded in traces and checkpoints.
Preserve the concrete manifest and receipt for reproducibility.
No Mininet execution occurs in the generator. C04 frozen inference
must be used for held-out evaluation and actual forwarding/telemetry
must be validated before measuring detection delay or switching.

For eventual analysis, align traces to actual impairment onset and
recovery (verified application time), report Q-margin/action changes,
response latency in ticks/seconds, false switches in nominal periods,
and missed detections. Keep scenario-level outcomes and matched seeds.
The schedule alone does not establish proactive routing or generalization.
Manuscript changes and final experiment execution remain deferred.
