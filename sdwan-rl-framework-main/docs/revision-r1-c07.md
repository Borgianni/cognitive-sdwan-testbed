# TNSM revision: Reviewer 1, Comment 7

Two-stage sensitivity analysis: (1) a factorial observation-only sweep
on common calibration telemetry; (2) retrain separate policies for a
predeclared one-factor-at-a-time design, plus any justified interaction
settings selected on calibration data before final held-out evaluation.

The six provisional axes are LTE RTT [100,175,350] ms, LEO RTT
[150,240,480] ms, LTE and LEO loss [5,10,20] percent, and LTE and LEO
residual fractions [0.1,0.2,0.4]. These are candidates, not SLA values.
Validate measured ranges, then revise the grid and baseline if necessary.
The inclusive loss boundary at 20 percent intentionally tests a loose
threshold that may not detect the nominal 20 percent impairment.

The offline factorial design has 729 configurations and requires no
DQN or Mininet training. The report includes feature activation fractions,
observation count, conflicting preferences, sample-weighted immediate
utility aliasing gap, and snapshot oracle reward. A constant bit (mean
0 or 1) is a diagnostic, not automatically an invalid feature. Report
results by impairment scenario and tunnel (including congestion and
recovery) rather than relying only on pooled samples; call the module
separately on each scenario/seed subset. The snapshot oracle uses fixed
telemetry, not causal forwarding effects. Exact continuous singleton
groups can mechanically yield zero empirical hindsight gap.

The policy plan contains 13 OFAT configurations (central baseline plus
two variations per axis), and optionally three reference encoders:
legacy_code, legacy_paper, continuous. With five paired seeds this is
65 training runs plus 65 frozen evaluations for OFAT alone, or 80+80
including all references. This is a provisional budget, not an executed
campaign. Interaction effects on actual policy performance cannot be
inferred from the OFAT design alone. Add selected joint configurations
only under a predeclared calibration-based rule if warranted.

The plan is deliberately non-executable: the current runner lacks
complete seeded, disjoint scenario support and validated forwarding.
It records full RL configurations and intended train/test roles, not
actual scenario definitions. Finalize train/calibration/test splits,
seed implementation, and budgets in C09-C12 and reproducibility work.
Do not choose thresholds on the final held-out data. Frozen alone does
not imply held-out. The offline split declaration is not a provenance
check; the operator must supply genuine calibration data.

Commands for later use, from the framework directory:

```bash
python -m unittest discover -s tests -p test_threshold_sensitivity.py -v
python -m sdwan_rl.threshold_sensitivity offline --split calibration --snapshots calibration.json --design factorial > sensitivity.json
python -m sdwan_rl.threshold_sensitivity plan --seeds 1 2 3 4 5 > sensitivity_plan.json
# Optional global arguments before the subcommand: --config cfg.json --grid grid.json
```

Use existing normalized continuous and binary comparators with the same
reward and training budget. Report held-out action/utility, QoS, switching
frequency and uncertainty across seeds, not just the offline aliasing gap.
No experimental outcomes are claimed in this revision. Manuscript changes
remain deferred until validation.
