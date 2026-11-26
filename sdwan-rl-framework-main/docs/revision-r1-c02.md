# TNSM revision: Reviewer 1, Comment 2

The original delay observation used hard-coded 700 ms (LTE) and 1200 ms
(LEO) thresholds. The new per-tunnel thresholds are configurable and
operate on positive measured end-to-end ping RTT in milliseconds. The
inclusive comparison matches the manuscript's mathematical convention;
the original implementation used strict less-than. Zero (the current
ping parser's failure sentinel), negative values and non-finite values
are not classified as acceptable. The reward calculation is unchanged;
its delay reference is independent of the observation thresholds.

The provisional defaults are 175 ms LTE and 240 ms LEO. These are based
on an illustrative symmetric tunnel-only RTT of 50/120 ms nominal and
300 ms impaired LTE (twice the configured one-direction NetEm delay).
175 ms is the midpoint of the LTE values; 240 ms is twice nominal LEO.
These are not measured RTTs or SLA limits. The topology specifies 50 ms
access-link delays, but does not explicitly use TCLink on access links;
verify the actual qdisc configuration and the return route. If both
access links impose 50 ms per direction, the corresponding idealized
RTTs would instead be 250/320/500 ms. Calibrate thresholds on measured
training/validation data before evaluating the policy. Do not tune on
held-out test outcomes.

From the framework directory:

```bash
python -m unittest discover -s tests -v

# Original numerical thresholds (inclusive boundary in revised code)
sudo python -m sdwan_rl.runners --agent-type dqn --lte-delay-threshold-ms 700 --sat-delay-threshold-ms 1200 --lte-loss-threshold-pct 0 --sat-loss-threshold-pct 0

# Provisional revised binary state
sudo python -m sdwan_rl.runners --agent-type dqn --lte-delay-threshold-ms 175 --sat-delay-threshold-ms 240 --lte-loss-threshold-pct 10 --sat-loss-threshold-pct 10
```

The original commit is the exact reference for the previous strict
comparison and invalid-input semantics. Unit tests use synthetic RTTs,
hold loss fixed for the delay-only ablation and do not verify Mininet
or policy performance. Probe routing, missing telemetry in reward,
degradation application and synchronization require separate validation.
