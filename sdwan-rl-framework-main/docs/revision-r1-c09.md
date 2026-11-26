# TNSM revision: Reviewer 1, Comment 9

The original 60-episode online trajectory is a training trajectory, not
an independent evaluation. C04 introduced frozen weights; C09 adds
explicit, disjoint scenario roles and checkpoint provenance.

## Protocol and controls

A version-1 JSON manifest lists named scenarios assigned to train,
calibration or held_out_test. The currently implemented scenario fields
are episode count, ticks per episode and inclusive LTE degradation
ranges. Ranges are normalized (sorted and merged), and the effective
schedule fingerprint ignores the scenario name and split. Duplicate
schedules and identifiers are rejected, including across splits.
The manifest is canonically hashed, with scenario ordering irrelevant.

Formal training requires online DQN, a new checkpoint and trace.
Formal calibration/test require a frozen DQN, a training checkpoint
from the exact same manifest, a distinct scenario in the requested
split, and a new trace. Schedule overrides are prohibited in formal
runs. The checkpoint metadata and trace header include the protocol.
C04 weight-hash verification and C06 observation configuration matching
remain in force. The threshold sensitivity tool accepts a trace only
if its recorded split is calibration; raw snapshot files still require
independent provenance verification.

The default diagnostic split preserves existing CLI examples and
legacy checkpoints. Diagnostic runs must not be called held-out results,
and their checkpoints cannot be used for formal evaluation. The audit
tools can still inspect diagnostic traces as diagnostics.

## Example only (not the experimental campaign)

docs/c09_example_manifest.json is a smoke-test schema example, NOT a
final experimental design. It uses the same LTE impairment severity
and traffic settings for all scenarios and changes only the temporal
schedule. C10 and C11 will extend the scenario model to held-out
severity, tunnel, traffic and randomized onset/duration. C12 and later
work will establish paired seeds, training budget and learning curves.
A different fingerprint is not proof of statistical independence.
The operator must predeclare splits, avoid tuning on test outcomes,
and validate the actual emulation and telemetry. Metadata prevents
accidental reuse, not deliberate mislabelling or data leakage.

The intended workflow is to gather separate calibration telemetry,
lock thresholds/hyperparameters using training/calibration data,
train a fresh model, then evaluate on untouched test scenarios. The
present runner supports one schedule per run and frozen calibration
of an already trained checkpoint; an independent pre-training
telemetry collector and richer scenario suite are still pending.

Commands for later validation, from the framework directory:

```bash
# Validate schema without starting Mininet:
python -m sdwan_rl.experiment_protocol --manifest docs/c09_example_manifest.json
python -m unittest discover -s tests -p test_experiment_protocol.py -v

# Illustrative only; wait for the finalized experiment protocol:
sudo python -m sdwan_rl.runners --split train --manifest docs/c09_example_manifest.json --scenario-id train_example --mode online --checkpoint train_example.keras --trace train_example.jsonl
sudo python -m sdwan_rl.runners --split calibration --manifest docs/c09_example_manifest.json --scenario-id cal_example --mode frozen --checkpoint train_example.keras --trace cal_example.jsonl
sudo python -m sdwan_rl.runners --split held_out_test --manifest docs/c09_example_manifest.json --scenario-id test_example --mode frozen --checkpoint train_example.keras --trace test_example.jsonl
```

Do not yet interpret policy decisions as verified forwarding outcomes.
The underlying impairment application, traffic and measurement pipeline
must be corrected and validated. No Mininet experiments or performance
claims are made by C09. Manuscript changes remain deferred.
