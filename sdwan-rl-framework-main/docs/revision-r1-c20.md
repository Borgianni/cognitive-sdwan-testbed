# TNSM revision: Reviewer 1, Comment 20

The original Fig.9 (p.10) is a DQN-only reward/cost ratio without an
explicit definition; p.9 interprets its recovery as optimal economic
efficiency. Neither the formula used historically nor an empirical
economic cost is documented. Do not attempt to reproduce the old
0.287 value from inconsistent logs (LTE cost 5 vs 7).

Define a secondary signed metric for any fixed run/episode:
  eta_RC = sum_t R_t / sum_t c_{a_t}
not the mean of per-tick ratios. c is a positive synthetic preference
index per decision, not per packet, byte or elapsed second. Ratios
are comparable only with the same reward, cost definition and
experiment conditions. Negative values are possible. This metric
penalizes cost both in reward and denominator, and is sensitive to
the additive reward offset (including a common T_D change).
Do not claim optimality, monetary/energy savings or robust economic
efficiency from eta_RC.

Prefer a QoS/cost tradeoff analysis: separate mean selected index
and mean cost-excluded QoS utility (weighted delay, loss, bandwidth
terms), with physical selected-tunnel delay, loss, residual capacity
and LTE selection fraction. These metrics are computed on measured
tunnel snapshots, not yet verified primary-flow QoS. Mean QoS
utility still depends on the reference scale/weights; compare under
one fixed configuration. Actual network efficiency would require
measured service cost, delivered traffic, time and verified routing.

The benchmark audit now emits cost/QoS metrics for each complete
frozen DQN, EWMA, Greedy Utility and Random held-out trace, both per
run and per episode. It verifies available cost/reward fields against
the shared environment, counts missing selected measurements, and
adds run-level mean/sample SD per scenario and policy for complete
paired matrices. The ratio is calculated per run then summarized
across seeds, not as a ratio of pooled episodes. Invalid selected
legacy telemetry is counted as unavailable for QoS, not imputed;
normalized traces must have valid selected telemetry. The old
upstream zero fallback problem remains pending. In weight/cost
sensitivity use the fixed reference rescore (C17) instead of
comparing own-policy reward ratios across objectives.

Future paper: p.9 "Link Selection Dynamics and Cost Efficiency"
and p.10 Fig.9. Replace economic/optimal balance language with
synthetic QoS-cost tradeoff analysis. Retain eta_RC only if
useful as a secondary explicitly defined statistic for all baselines;
otherwise replace the main figure and explain why. Results pending.

No emulation or manuscript edits performed.
