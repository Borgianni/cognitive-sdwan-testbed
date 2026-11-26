# TNSM revision: Reviewer 1, Comment 10

Version-2 manifests extend C09 with per-tunnel impairment events:
inclusive start/end episodes, affected tunnel, one-direction NetEm
delay in ms and packet loss percent. Multiple events can overlap across
tunnels, but conflicting events on one tunnel are rejected. Identical
adjacent/overlapping events are merged for canonical provenance. Outside
an event the nominal delay/loss is restored. Nominal bandwidth, jitter
and queue limits are preserved. A background traffic vector specifies
constant offered UDP rates per tunnel for the scenario; rates are
bounded by nominal tunnel capacities and the shared access budget.
These are offered rates, not measured goodput or guaranteed loads.

The v2 runtime changes existing child NetEm qdiscs under HTB instead of
replacing the root, checks exit status, preserves queue/jitter, and
logs raw kernel readback. It refuses unexpected qdisc hierarchies.
Readback verifies hierarchy, not portable numerical parsing of all
options; check exact applied parameters against tc output in integration
tests. Background iperf2 flows use dedicated UDP ports 5101/5102 with
high-priority ingress/egress OVS matches pinned to the corresponding
tunnel, separate from the existing default-port loss probe. The runtime
checks process liveness and ingress/egress flow counter deltas, stops
background processes and removes its rules before network cleanup.
Ingress counters verify offered packets entering a flow rule, not actual
wire throughput. Zero egress matches can indicate loss or broken routing.
Record, investigate and validate before performance evaluation.

The example v2 manifest includes train, calibration and held-out
intermediate, severe, LEO, background load and combined impairment
cases. Values are provisional, not measured results. The protocol CLI
reports unseen delay/loss pairs, affected tunnels and load vectors both
relative to training and relative to train+calibration. An unseen pair
does not necessarily lie outside the training range. LEO may be seen
in calibration while unseen during learning; do not call it entirely
unseen development data. Distinct scenario hashes prevent exact
schedule reuse, not statistical independence.

From the framework directory (no network needed for schema checks):
```bash
python -m sdwan_rl.experiment_protocol --manifest docs/c10_example_manifest.json
python -m unittest discover -s tests -p test_scenario_profiles.py -v
```

Do not launch the final campaign yet. Existing v1 manifests and
diagnostic commands remain supported. V2 uses the same frozen checkpoint
and manifest provenance checks introduced in C09. C11 will randomize
onsets and durations with reproducible scenario seeds; C12 and C21 will
address run seeds/training curves. Actual probe isolation, policy-driven
forwarding, OVS/STP behavior, throughput parsing, timing, and real
NetEm/iperf integration remain unvalidated. These are mandatory
before interpreting network performance. The current runner still
uses one scenario per training run; the example has multiple LTE
impairment events inside that scenario. No claim of robustness or
generalization follows from implementing the scenarios.
