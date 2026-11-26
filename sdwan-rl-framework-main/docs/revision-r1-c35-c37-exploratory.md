# C35-C37: common exploratory reporting

The main predeclared statistical_report is the only formal four-test
family. Older benchmark and learning-curve reports keep their existing
keys and add seed-level values, SD, quantiles and pointwise t intervals.
Reward sensitivity uses candidate minus reference under a fixed common
utility; age sensitivity compares with lag zero; scalability compares
DQN and each baseline within a case; RR retains paired deltas; probe
impact reports on-minus-off differences and optional prespecified
equivalence margins. None runs per-tick significance tests.
Calibration-only threshold/aliasing snapshot sweeps remain diagnostics:
the samples are dependent, so no independent-tick CIs are added.
For learned encoder/threshold comparisons use matched formal runs per
configuration, with separate locked analysis plans, and compare run
summaries as exploratory paired effects. A dedicated multi-encoder
report/launcher remains a later consolidation task.

For probe equivalence use --equivalence-margins margins.json on plan,
with positive values for primary_rtt_ms (ms), primary_loss_pct
(percentage points), primary_delivered_payload_mbps (Mbps). Do not
choose margins after viewing paired differences. If no margins were
locked, report differences/CI only. The per-endpoint exploratory TOST
indicator does not establish joint equivalence or account for all
other impacts; zero variance or missing pairs are not evidence.
Use a separate predeclared inferential design if a joint equivalence
claim is needed.

The main report marks a truncated final JSONL line as incomplete,
not a completed replicate. Malformed intermediate records are errors.
Missing run headers must be tracked using external execution logs.
SciPy versions are captured in the analysis runtime metadata.
