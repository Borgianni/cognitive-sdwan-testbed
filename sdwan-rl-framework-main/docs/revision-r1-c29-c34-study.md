# C29-C34: running the separate bounded scalability study (later)

These commands are for a validated Linux testbed, NOT executed here.
From sdwan-rl-framework-main, use a fresh Ryu process per scenario
replicate; ensure PYTHONPATH includes the framework. Do not run both
controller apps on the same ports.

Create and lock scenario files before experiments. Example:
```bash
python -m sdwan_rl.scale_runner scenario --case parallel3 --split train --episodes 60 --ticks 20 --output scale_train.json
python -m sdwan_rl.scale_runner scenario --case parallel3 --split held_out_test --episodes 12 --ticks 20 --output scale_test.json
# terminal 1 (in the Ryu environment):
PYTHONPATH=. SDWAN_SCALE_SCENARIO="$PWD/scale_train.json" ryu-manager --ofp-tcp-listen-port 6653 --wsapi-port 8080 ryu_apps/scale_controller.py
# terminal 2 (root, framework Python environment):
sudo -E env PYTHONPATH=. python -m sdwan_rl.scale_runner run --scenario scale_train.json --agent dqn --seed 1 --checkpoint scale_3_s1.keras --trace scale_3_train_s1.jsonl
```

Stop/restart Ryu with the test scenario file. Run frozen DQN with
the same seed/checkpoint and --scenario scale_test.json; run greedy
with the same file/seed but --agent greedy and no checkpoint.
Random is optional. Repeat parallel2, parallel3, parallel4, triangle3.
One common continuous encoder/reward/telemetry configuration for
all cases. Train a fresh network per case and seed. Training and
test budgets above are illustrative: lock actual budget after
timing/calibration, and preserve all failures. Same seed labels
do not guarantee identical Linux/NetEm stochastic realizations.
Separate matched comparisons within each case; cross-case timing
and resource growth are descriptive. Do not pool ticks as independent
replicates. The scalability reference (parallel2) uses the new path
catalog runtime; do not merge its numbers with the main dual-tunnel
benchmark. The limited suite does not establish arbitrary N, arbitrary
graph discovery, multi-agent control, or cross-topology transfer.

Report absolute encode+decision, controller acknowledgement,
collection and tick-work distributions, actual probe bytes, selected
path fractions, primary QoS and synthetic cost index. Main benchmark
audit C14-C20 is unchanged; scale_audit is a dedicated report.
```bash
python -m sdwan_rl.scale_audit benchmark --traces <all_test_traces> --cases parallel2 parallel3 parallel4 triangle3 --seeds 1 2 3 4 5 --agents dqn greedy
```

C34: create a calibration scenario; launch controller with
SDWAN_SCALE_SECONDARY=pinned or rr, use the same --secondary option
on runner. Select --agent fixed --fixed-action 0 (or 1 etc.) and
no checkpoint. Keep all other settings and seed matched, predeclare
and counterbalance run order. Compare primary and secondary QoS,
late-arrival fraction, receive coverage, per-path RR dispatch.
```bash
python -m sdwan_rl.scale_runner scenario --case parallel2 --split calibration --output rr_cal.json
PYTHONPATH=. SDWAN_SCALE_SCENARIO="$PWD/rr_cal.json" SDWAN_SCALE_SECONDARY=rr ryu-manager --ofp-tcp-listen-port 6653 --wsapi-port 8080 ryu_apps/scale_controller.py
sudo -E env PYTHONPATH=. python -m sdwan_rl.scale_runner run --scenario rr_cal.json --agent fixed --fixed-action 0 --secondary rr --seed 1 --trace rr_s1.jsonl
# restart controller with pinned and repeat, then compare:
python -m sdwan_rl.scale_audit rr --traces <all_paired_traces> --case parallel2 --seeds 1 2 3 4 5 --actions 0
```

Both arms have concurrent primary, extra secondary, probes and pinned
cross traffic. RR routes successive source packet-in events over
paths and can alter load distribution and controller overhead as well
as order; this comparison does not isolate reordering as the sole
cause of any difference. RTT is conditional on successful echo and
reverse echo is always pinned to first path in this diagnostic.
Primary reward in normal scalability runs still comes from selected
path probes, not the independent primary workload; compare both.

No final claim until actual smoke tests (route readback including
3-hop switch path, qdisc configuration, received counts, background
traffic, checkpoint reload, timing, reordering) and replicated runs.
