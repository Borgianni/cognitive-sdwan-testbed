# TNSM revision: Reviewer 1, Comment 17

Retain the original weights as reference, not as an optimum. An
OFAT design varies each positive weight by factors .5 and 2 (nine
configs including the reference). Add cost-free reward (w_d=0),
equal tunnel indices and LEO/LTE cost ratio 5 (12 unique candidates
at the default). This also addresses sensitivity of the synthetic cost
assumptions (C19). It does not estimate interactions between weights.
The reference bandwidth and loss scales preserve the old numerical
tradeoff; alternative scales can be predeclared as separate studies.
Weights and references jointly determine effective tradeoffs.

Run the offline diagnostic on common calibration snapshots to assess
changes in greedy choices. It is not a DQN sensitivity result. For
learned-policy sensitivity, train a separate fresh DQN per candidate
and seed with the same observation representation, training scenario
and budget; use matched frozen tests. Predeclare a tractable subset
of held-out scenarios and seeds, not the full cross-product of all
encoders/thresholds/weights/algorithms. The plan records all configs
and a manifest hash. It launches nothing. Select the reference
configuration and any subset using development data only.

Never compare each agent's reward under its own different weights as
if the values had a common scale. The evaluate subcommand validates
frozen traces using the benchmark audit, matches the planned
config/seed/scenario matrix, checks model identity per replicate and
rescores each chosen action on its recorded measurements under one
fixed reference config. Report reference utility, selected-tunnel
RTT/loss/residual capacity, LTE share and within-episode switches,
with mean/sample SD across seeds by scenario. These remain measured
snapshot proxies rather than verified primary-flow QoS, causal effects,
or actual monetary costs. A reference cost rescore is hypothetical
when a policy was trained under a different cost vector. Report own
training costs/weights separately. A plan cannot prove no data leakage;
keep it immutable and do not select using final test outcomes.

Commands for later validation:
```bash
python -m sdwan_rl.reward_sensitivity offline --split calibration --snapshots calibration.json
python -m sdwan_rl.reward_sensitivity plan --manifest randomized_manifest.json --train-id train_reference_r001 --test-ids test_severe_r001 test_joint_r001 --seeds 1 2 3 4 5 > reward_plan.json
python -m sdwan_rl.reward_sensitivity evaluate --plan reward_plan.json --traces <all_expected_trace_paths>
python -m unittest discover -s tests -p test_reward_sensitivity.py -v
```

The plan is provisional; budget and final subset depend on actual
measurement timing. No experiments or manuscript edits made.
