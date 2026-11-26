# C23-C24-C27-C28: concurrent measurement primitive

The old per-tunnel sequential ping/bwm-ng/iperf sequence is not the
reference measurement path for revised experiments. A new bounded,
sequence-numbered UDP echo protocol uses a common monotonic clock
across Mininet namespaces on one Linux host. Forward unique receives
estimate one-way UDP loss within a declared grace period. Matched
echoes estimate RTT conditional on successful round trips; missing
echoes are not zero RTT. Echo loss is reported separately. The
measurement is of this probe traffic, not an application-independent
path guarantee. Token and sequence checks discard stale/duplicate
packets. All tunnel and primary streams use a common send window,
with a shared receive deadline and actual timestamps. A short window
is provisional; packet count, payload, window and grace are explicit.
100 samples resolve loss in 1 percentage-point increments, not a
guarantee of statistical precision. RTT and loss represent overlapping
supports, not an atomic physical state.

TX byte counter deltas over actual elapsed monotonic time use decimal
Mbps. Total TX includes probe/background/primary traffic and protocol
overhead as exposed by Linux interface counters. Nominal capacity
minus measured TX throughput (clipped at zero) is an estimated
residual interface capacity, not end-to-end available bandwidth.
Do not subtract UDP payload bytes from interface bytes, which have
different accounting layers. Probe offered payload rates and echo
bytes are reported separately. The runtime integration and paired
probe-on/off measurements are addressed in the next commits.

Unit tests cover count/units, timestamp age and a real localhost
UDP sequence/echo exchange. No Mininet validation or QoS claims.
