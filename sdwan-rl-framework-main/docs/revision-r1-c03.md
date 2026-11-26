# TNSM revision: Reviewer 1, Comment 3

The published paper describes B_k > 0 (positive available bandwidth),
whereas the original public code actually checks measured throughput > 0.
These are distinct predicates. The revised default compares the estimated
residual capacity fraction max(0, min(1, (C_k - T_k)/C_k)) against an
independent per-tunnel threshold beta_k in (0, 1]. C is configured capacity
in Mbps and T is the measured interface transmission rate in Mbps.
The provisional beta defaults to 0.20 (20% residual), not an SLA.

The code exposes three bandwidth predicates: residual_threshold (revised),
legacy_throughput (original code), positive_residual (paper predicate).
The latter two allow controlled ablations, not claims of exact replication of
the full originally submitted implementation. The reward and six-feature
ordering are unchanged. Invalid numerical rates are not classified as
acceptable; reset observations remain zero until metrics are received.

Run from the framework directory:

```bash
python -m unittest discover -s tests -v
sudo python -m sdwan_rl.runners --agent-type dqn --bandwidth-state-mode residual_threshold --lte-residual-threshold 0.2 --sat-residual-threshold 0.2
sudo python -m sdwan_rl.runners --agent-type dqn --bandwidth-state-mode legacy_throughput --lte-delay-threshold-ms 700 --sat-delay-threshold-ms 1200 --lte-loss-threshold-pct 0 --sat-loss-threshold-pct 0
```

The original commit is the exact code reference. Revised comparisons retain
inclusive delay boundaries and reject invalid inputs. Earlier C01/C02 command
examples predate the bandwidth-state flag; use legacy_throughput to hold the
old bandwidth predicate fixed.

The residual fraction is an interface-capacity proxy, not measured end-to-end
available bandwidth. The existing bwm-ng parser, rate units, interface mapping,
missing-rate fallback to zero, configured vs actual shaping capacity, and
measurement windows require verification before interpreting experiment data.
A missing sample currently can still appear as zero throughput; the reset
guard cannot detect that upstream fallback. No changes to the measurement
pipeline or reward are made here to isolate the observation change.

Use independent bandwidth-load scenarios (e.g. verified offered cross-traffic
sweeps at 20%, 80%, and 90% utilization), while controlling delay and loss
configuration and measuring actual rates. The existing LTE impairment changes
delay and loss, not nominal configured capacity, so a bandwidth-bit transition
during that scenario is not guaranteed. Calibrate thresholds on validation
observations, freeze them before held-out testing, and compare with normalized
continuous observations in C06. Unit tests use synthetic inputs only.
