# TNSM revision: Reviewer 1, Comment 13

## Scientific position

Retain the latency-only EWMA described in Section IV-A / Table II
of the submitted manuscript (p. 7). It is a legitimate, inexpensive,
measurement-driven single-metric reference. The limitation is the
scope of conclusions: outperforming it does not establish that
learning itself outperforms non-learning multi-metric selection.
C14 will add a competitive multi-metric comparator on the same
decision-time telemetry and utility. No need to discard or retrofit
the original EWMA into a different algorithm.

The publicly inspected runner previously exposed only DQN and random;
the EWMA class added here implements the manuscript's described
lowest-smoothed-RTT rule. The exact historical EWMA implementation
and smoothing coefficient have not been recovered. Do not claim
numerical reproduction of the submitted EWMA results.

## Implementation and fairness

For a valid positive finite RTT sample D_k(t), initialize the estimate
to the first sample; subsequently m_k(t) = alpha D_k(t) +
(1-alpha) m_k(t-1). Select the tunnel with the smallest smoothed RTT.
Alpha=0.3 is a provisional default, not claimed to be the original
value. Consider a small predeclared calibration-only grid
(e.g. 0.1, 0.3, 0.7, 1.0), select by an agreed objective, freeze alpha
before held-out evaluation and report the selection rule. Alpha=1 is
unsmoothed latest-RTT selection.

Both EWMA and DQN consume the previous tick's available telemetry.
No scenario labels, current/future probe results or reward are used
to select the EWMA action. EWMA intentionally uses only RTT from the
same telemetry collection process; this is a single-metric reference,
not an information-matched comparator. The first tick of each episode
has no telemetry: default LTE, as for a tied greedy DQN, though
a trained DQN can choose a different action on its zero state. Reset
the smoother at the same episode boundaries as the DQN observation
reset. For invalid/missing RTT, exclude that tunnel from the current
comparison (retain its historical estimate for later valid updates);
if both invalid, hold the last choice. Ties favor LTE. This policy
uses mutable filter state but no learned parameters or exploration.

The runner accepts --agent-type ewma --mode heuristic in calibration
and held-out test splits; no checkpoint is required or allowed.
Formal runs still require manifest, scenario id, seed and trace.
The same measurement, reward, controller and scenario paths are used.
Trace records the alpha, smoothed RTT estimates, validity and decision
reason. Q-values/greedy-Q action are null, not fabricated. The C12
learning curve report remains DQN-specific; a common benchmark report
will be introduced with C14-C15 / statistical validation.

## Commands for later validation only

```bash
python -m unittest discover -s tests -p test_ewma.py -v
python -m unittest discover -s tests -p test_heuristic_protocol.py -v
sudo python -m sdwan_rl.runners --agent-type ewma --mode heuristic --split calibration --manifest randomized_manifest.json --scenario-id cal_reference_r001 --seed 1 --ewma-alpha 0.3 --trace ewma_cal_1.jsonl
sudo python -m sdwan_rl.runners --agent-type ewma --mode heuristic --split held_out_test --manifest randomized_manifest.json --scenario-id test_joint_r001 --seed 1 --ewma-alpha 0.3 --trace ewma_test_1.jsonl
```

The same scenario/seed identifiers permit matched comparisons but do
not guarantee identical stochastic network realizations. Parameter
selection is not enforced across separate calibration/test invocations
yet; record and lock the selected configuration before test execution.
Underlying probe routing, actual policy forwarding and telemetry
validity remain unvalidated. Do not launch the final campaign or
claim new performance results at this stage.

## Manuscript (deferred)

Page 7, Section IV-A, Table II and text describing latency-aware EWMA:
retain EWMA but state its single-metric role, formula, alpha,
initialization and comparison limits. Page 8-9 Results: distinguish
advantages over latency-only selection from evidence of learning
benefits versus multi-metric heuristics. Revise claims only after
actual matched experiments. No manuscript edits yet.
