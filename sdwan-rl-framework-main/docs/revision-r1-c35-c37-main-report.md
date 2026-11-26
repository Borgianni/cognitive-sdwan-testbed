# C35-C37: locked main benchmark analysis

The statistical_report module is a separate formal analysis layer, not a
new experiment runner. It locks the full normalized scenario manifest,
one training scenario, one primary held-out scenario, RL/telemetry
configuration, seeds, agents, EWMA alpha, planned run order and windows.
Primary family: DQN vs EWMA and DQN vs Greedy Utility, each for primary
flow loss and RTT over the union of declared impairment episodes in the
one primary held-out scenario (four tests, Holm alpha .05).
Choose this scenario based on research question/development data,
not test outcomes. Random is a secondary comparator. All other
scenarios/windows/endpoints are exploratory; no tests at every episode.
The plan records a reproducible shuffled training order and randomized
policy order within each seed/scenario block. Execute and log the order
with fresh networks/processes. A plan hash does not prove compliance or
prevent retrospective edits: archive/commit it before held-out runs.

Both training and evaluation traces must be supplied. Complete DQN
evaluations must match the final training digest and protocol for that
seed; initial digests across completed training runs must be distinct.
The existing benchmark and timing audits validate complete traces.
A missing, failed or incomplete cell is retained in an inventory.
Duplicate attempts for a cell are rejected. Preserve all attempts in
a separate failure ledger; do not silently replace a failed cell.
Report partial completed-run summaries as descriptive only and no
primary tests if the planned matrix is incomplete. If a complete
matrix has undefined RTT in a primary arm, retain missing values and
suppress confirmatory rejection across the primary family. Do not
treat undefined RTT as zero. Report echo coverage. A trace with no
identifiable header is missing/invalid, not an evaluable run. External
logs are needed for failures before the header. No automatic retries.

For each run and inclusive window, primary loss = 100*(sent-received)/
sent, primary RTT = echo-count-weighted mean of successful echoes,
delivered payload = received payload / summed measurement support.
These are distinct from selected-path probe metrics and from the reward.
Across runs every seed has equal weight. Loss differences are percentage
points, RTT differences ms. Relative reductions use the ratio of the
two matched run-level means; no relative estimate for missing pairs or
zero denominator. Per-episode pointwise curves retain every seed value;
they are descriptive and do not establish simultaneous coverage.
Absolute paired differences and unadjusted 95% CI are the principal
effect estimates. Report Holm-adjusted primary p-values separately.
The full report keeps individual training and evaluation summaries.
Do not pool scenarios sharing a trained checkpoint as independent runs.

Original manuscript C36: (11.3-4.2)/11.3 = 62.8319%, using rounded
full-window values, not evidence about episode 40. (102-78)/102 =
23.5294% similarly. Original p.8 text says episode 40; p.9 aggregate
figures show window means. Historical provenance and unrounded
seed-level data are not established. Do not retroactively assign a
statistic to the old plots. Revised paper will regenerate numbers from
the locked window, with explicit source metrics, comparisons, n and CI.
No result is asserted before new measurements.

Analysis dependencies: install requirements-analysis.txt in an analysis
environment. No extra Mininet or Ryu dependency. Later examples:
```bash
python -m sdwan_rl.statistical_report plan --manifest manifest.json --train-id train_reference_r001 --primary-scenario test_severe_r001 --seeds 1 2 3 4 5 6 7 8 9 10 --rl-config locked_rl.json --telemetry-config locked_telemetry.json --output stats_plan.json
python -m sdwan_rl.statistical_report report --plan stats_plan.json --training <training_traces> --evaluation <all_evaluation_attempt_traces> --output stats_report.json
```
Use actual manifest IDs and calibrated configs. No experiments here.
