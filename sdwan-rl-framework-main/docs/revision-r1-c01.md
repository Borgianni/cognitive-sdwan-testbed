# TNSM revision: Reviewer 1, Comment 1

The original observation marks packet loss as acceptable only when the measured
loss is zero. The revised observation compares measured loss in percent against
a configurable per-tunnel threshold, inclusively. The default 10% is a
provisional experimental separator for the nominal 2%/3% and impaired 20%
configured loss values, not a validated service-level objective.

The feature ordering, state dimension, reward and agent are unchanged. Setting
both thresholds to 0 reproduces the original encoding for valid measurements.

From the framework directory:

```bash
python -m unittest discover -s tests -v
sudo python -m sdwan_rl.runners --agent-type dqn --lte-loss-threshold-pct 10 --sat-loss-threshold-pct 10
sudo python -m sdwan_rl.runners --agent-type dqn --lte-loss-threshold-pct 0 --sat-loss-threshold-pct 0
```

The unit tests use controlled metric inputs; they do not establish policy
performance or validate the Mininet measurement pipeline. Configured netem loss
and measured end-to-end loss can differ. Calibrate thresholds on measured
training/validation observations, report overlap and threshold sensitivity, and
evaluate a newly trained policy for each encoding on held-out scenarios. The
existing probe parser can return zero when parsing fails; missing telemetry
handling is not resolved by this change. Other known state and forwarding
limitations are outside the scope of this comment.
