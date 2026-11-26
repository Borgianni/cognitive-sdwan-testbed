# Reviewer 1 C24-C28: timing, age and probe perturbation

Report measured stage and inter-tick absolute durations (mean, p50,
p95, maximum), separate episode setup and checkpoint times, and
measurement age/support/skew distributions. Use separate online and
frozen reports. These are periodic overlay-control timings, not
line-rate or production guarantees. Synthetic timing illustrations
are not experiments.

C24: run fixed trained weights with observation lag 0,1,2 on a
predeclared matched subset. The extra lag changes policy input only;
same measurement settings, model and scenario/seed IDs. Compare
actual measured age and primary QoS; exclude max(lag)+1 startup
ticks equally in each arm. No silent partial matrices. Actual
network randomness may differ; lag is an artificial stress factor.

C27: TX bytes are measured at source tunnel interfaces with actual
counter timestamps and decimal units. Estimated residual interface
capacity = max(nominal capacity - total TX rate, 0), includes probe
and workload load, not measured application-available bandwidth.
Retain the term as a cheap operational proxy, not a bandwidth estimator
with a guarantee.

C28: the old nominal iperf configuration offered 5 Mbit/s for 10 s
per tunnel, approximately 50 Mbit payload per tunnel per tick (not
measured bytes). Default new additional probe budget is 100 x 128 B
forward per tunnel; at most 51,200 B UDP payload total including
successful echo attempts on two tunnels. Report actual payload
bytes and OVS match bytes; not exact wire traffic. Primary workload
is separate and remains present when additional tunnel probes are off.

The probe_impact module runs fixed-action calibration scenarios
with identical primary/background settings, extra probes on/off.
Separate from RL benchmarking, because disabling observations
would otherwise change the policy. The plan randomizes order
within each replicate pair; execute the listed order in separate
emulations, preserving failed runs. Seed is a replicate identifier,
not a guarantee of identical NetEm loss realizations. Compare
primary one-way loss, successful-echo RTT and delivered payload
rate over measurement supports. Aggregate paired differences by
scenario and action; report RTT coverage. Declare practical
equivalence margins before collecting evidence and use confidence
intervals (C35-C37) before asserting no material perturbation.
Non-significance alone is not equivalence. No results yet.
The probe-on fixed-action calibration traces can also export
unselected/selected RTT labels and snapshots for C07/C08 without
training a policy first. Labels come from requested profiles;
check actual qdisc readback and exclude unavailable RTT explicitly.

Later commands (from framework directory, Ryu running):
```bash
python -m sdwan_rl.probe_impact plan --manifest randomized_manifest.json --scenarios cal_reference_r001 --seeds 1 2 3 4 5 > probe_plan.json
sudo python -m sdwan_rl.probe_impact run --manifest randomized_manifest.json --scenario-id cal_reference_r001 --seed 1 --action 0 --probes on --trace probe_on.jsonl
python -m sdwan_rl.probe_impact export-calibration --trace probe_on.jsonl > calibration_export.json
python -m sdwan_rl.probe_impact compare --plan probe_plan.json --traces <all_expected_probe_traces>
python -m sdwan_rl.telemetry_audit timing --traces <complete_timed_runs>
python -m sdwan_rl.telemetry_audit age --traces <all_expected_frozen_lag_runs> --seeds 1 2 3 4 5 --scenarios test_severe_r001 --lags 0 1 2
```

Export calibration_export.json samples directly for delay_calibration,
extract the snapshots array for threshold/reward sensitivity tools.
Do not run final experiments yet. Unit/loopback checks do not validate
Mininet, Ryu, OVS, real timing, traffic isolation or perturbation.
