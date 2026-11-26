# C39: exploratory learner-comparison protocol

Create the main C35-C37 statistical plan after calibration, then
predeclare a smaller learner comparison before held-out evaluation.
Reuse the corresponding DQN training/evaluation traces; train Expected
SARSA from scratch for the same seed and training scenario using the
same RLConfig (including encoder), telemetry, workload and decision
budget. Algorithm-specific hyperparameters are distinct. Choose alpha
on calibration/development data or keep the documented .01 default.
Do not tune on held-out results. One training checkpoint per seed,
not one per test scenario. Keep ordinary episode resets and lag zero.

Example (actual names/configuration to be chosen after pilot):
python -m sdwan_rl.learner_comparison plan --main-plan stats_plan.json --seeds 1 2 3 4 5 --scenarios test_severe_r001 --output learner_plan.json

Run the same sdwan_rl.runners commands as DQN, replacing agent-type
with expected_sarsa, checkpoint extension with .npz and specifying
--sarsa-alpha .01 for both training and frozen evaluation.
Do not include Expected SARSA in the primary four-test family.
Use learner_comparison report with both algorithms' complete train
and held-out traces. It checks common settings, scenario and checkpoint
provenance and reconstructs SARSA updates/decisions from its trace;
DQN remains validated through its existing trace and weight hashes.
Run-level paired differences and pointwise CIs are exploratory. A
failure is not a zero score: keep an external ledger, do not silently
replace failed runs, and withhold complete-matrix comparisons when
required cells are missing. No empirical results yet.
