# Smoke test v1 -- diagnostics only

First run the non-privileged inventory:
  python3 scripts/preflight.py --imports > preflight.json
Send the report, including errors, before installing/changing versions.
If Ryu and TensorFlow use separate Python environments, run the
inventory with each interpreter. No credentials are requested; inspect
and redact personal paths in errors if needed.

After reviewing inventory and unit tests, use an ISOLATED Linux host.
Never use Mininet cleanup or root network mutation on a production
network. The following are staged instructions, not a request to run
all stages now. Start Ryu in its tested environment:
  ryu-manager --ofp-tcp-listen-port 6653 --wsapi-port 8080 ryu_apps/switch_controller.py

Run these only after environment and controller readiness have been
checked. From the framework directory, use the exact Python executable
for the Mininet/ML environment with root rights. The placeholders
PYTHON below must be replaced with its absolute path. Create a fresh
output directory, preserving logs for every attempted run.

  mkdir -p smoke_v1_output
  sudo env PYTHONPATH=. PYTHON -m sdwan_rl.probe_impact run --manifest docs/reference_smoke_v1/manifest.json --scenario-id smoke_cal --seed 1 --action 0 --probes on --trace smoke_v1_output/fixed_lte.jsonl

Check every tick's controller ack, flow evidence, primary/probe counts,
TX timestamps, qdisc readback, timing and no failed footer. Repeat with
action 1 and fresh trace, only after first run passes. Check nominal,
impairment and recovery. These fixed-route runs do not require TF.

Train a small smoke DQN with full JSON configs:
  sudo env PYTHONPATH=. PYTHON -m sdwan_rl.runners --agent-type dqn --mode online --split train --manifest docs/reference_smoke_v1/manifest.json --scenario-id smoke_train --seed 1 --rl-config docs/reference_smoke_v1/rl.json --telemetry-config docs/reference_smoke_v1/telemetry.json --checkpoint smoke_v1_output/dqn.keras --trace smoke_v1_output/dqn_train.jsonl

Then frozen test on smoke_test with --mode frozen --split held_out_test
and matching config/checkpoint/seed, fresh trace. Verify frozen digest,
epsilon 0, no learning. Test Expected SARSA analogously using
--agent-type expected_sarsa --checkpoint smoke_v1_output/sarsa.npz.
Use fresh controller process per replicate/diagnostic if practical and
capture its stderr/log. Stop at first unexpected error and preserve
trace, controller logs and exact command. No final run until diagnostics
pass and actual calibrated configuration/budget is locked.
