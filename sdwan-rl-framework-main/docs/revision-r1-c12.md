# TNSM revision: Reviewer 1, Comment 12

## Scientific decision

Retain the online cognitive routing/learning loop. Non-stationarity is
part of the problem, not a defect to remove merely to obtain a smooth
curve. A rolling average of rewards under changing conditions and
decaying exploration is not evidence of convergence or generalization.
Report online learning dynamics separately from fixed-policy held-out
performance. Do not add a convergence claim, stopping criterion, or
new RL architecture merely to satisfy this comment. C21-C22 will
address DQN implementation consistency and convergence wording.

## Independent runs and frozen checkpoints

Both entry points now accept --seed. Formal train/calibration/test
runs require an explicit integer seed in [0, 2^32-1]. The DQN seed is
set with tf.keras.utils.set_random_seed before creating the model,
which seeds Python, NumPy and TensorFlow. Scenario-generation seeds
(C11) are separate. This is not full determinism: kernel NetEm random
loss, process scheduling, traffic, GPU operations and software versions
still require control/documentation. No change to the reward, network
architecture, replay algorithm, or online epsilon schedule is made.

Checkpoint metadata records the training seed; formal frozen runs
must use the matching replicate seed, RL configuration and manifest.
The trace header includes the seed, agent type and training provenance.
Old diagnostic examples remain usable without a seed; formal examples
in C09/C10 require --seed now. Old unseeded formal checkpoints cannot
be used for the new formal evaluation. The same seed identifies a
replicate, not identical physical network randomness.

## Learning-curve report

The new module reads only complete seeded formal DQN traces, checks
the declared schedule budget, every ordered tick, finite rewards,
exploration fields and consistency of greedy actions. It compares
independently initialized runs on the same training scenario and RL
configuration; if training scenarios differ, summarize separately.
Report raw per-run episode totals and tick means, plus episode-level
mean and sample SD across seeds, mean epsilon and exploration fraction.
No smoothing is necessary. If included for readability, use a declared
window and show the unsmoothed curve; do not call a plateau convergence.

Optional frozen evaluation reports require one result per declared
test scenario per training seed, with matching training protocol,
configuration and final weight hash, zero exploration and unchanged
weights. The report summarizes test mean tick reward by scenario
across seeds, separately from training. It rejects incomplete matrices
rather than silently dropping failed cases. Failed runs and reruns
must still be documented externally; these checks are not proof of
statistical independence or of effective data-plane forwarding.
Uncertainty intervals and paired inference are deferred to C35-C37.

Example commands FOR LATER (one configuration, same manifest for every seed):
```bash
sudo python -m sdwan_rl.runners --split train --manifest randomized_manifest.json --scenario-id train_reference_r001 --mode online --seed 1 --checkpoint train_1.keras --trace train_1.jsonl
sudo python -m sdwan_rl.runners --split held_out_test --manifest randomized_manifest.json --scenario-id test_joint_r001 --mode frozen --seed 1 --checkpoint train_1.keras --trace eval_joint_1.jsonl
python -m sdwan_rl.learning_curves --training train_1.jsonl train_2.jsonl train_3.jsonl train_4.jsonl train_5.jsonl --output learning_report.json
python -m unittest discover -s tests -p test_learning_curves.py -v
```

For a complete held-out report append --evaluation with all expected
seed/scenario files. Use a predeclared feasible number of seeds and
test cases after measuring runtime and fixing telemetry/forwarding;
do not reduce a matrix after seeing results. The example seed count
is provisional, not a power analysis. Existing C10/C11 example
manifests are software examples, not final experiment plans.

## Manuscript (deferred)

Page 8, section D, replace the "Total Reward Convergence and
Generalization" heading and the sentence claiming the seven-episode
rolling average confirms successful convergence. Retain the study of
online learning, revise Fig. 4 to show multiple seeded runs with
dispersion, and add a separate frozen-policy held-out evaluation.
Only claim adaptation, robustness or performance improvements if
supported by actual results. No emulation was executed in this change.
