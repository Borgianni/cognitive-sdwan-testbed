# TNSM revision: Reviewer 1, Comments 16, 18, 19

Keep the logarithmic multi-objective reward. Define loss fraction ell=L/100
when telemetry L is in percent, and a common residual-bandwidth reference
B_ref. In normalized mode:

  R = w_b ln(T_D/D) - w_a[1+ln(1+ell/ell_ref)]
      + w_c ln(1+B_res/B_ref) - w_d c

All arguments are dimensionless. D is measured RTT, T_D=140 ms is the
value in the inspected public RLConfig (not reported in submitted paper).
It is a common reward reference, not an SLA/observation threshold. For
fixed positive D, replacing a common T_D changes each action's reward by
the same additive constant w_b ln(T_D_new/T_D_old). It does not change
instantaneous action ranking or the ideal fixed-horizon objective ranking,
but can affect learned values, optimization and reward ratios. Report it.

B_res=max(C-throughput,0) is an estimated residual interface capacity,
not measured end-to-end available bandwidth. B_ref=1 Mbps and ell_ref=.01
are explicit common positive physical scales chosen to preserve the
historical numerical tradeoff on valid positive non-floor samples.
They are not optimized or claimed as universal values. Converting
both a quantity and its reference to another unit leaves the ratio
unchanged. Normalized mode removes artificial floors for zero loss
and zero residual capacity; invalid selected RTT/throughput/loss fails
closed rather than treating zero RTT as excellent. Missing upstream
loss/throughput still may be encoded as numerical zero; telemetry
validation remains required. Legacy mode is a reproduction comparator
at historical parameters, not recommended for final runs. Reward
default is normalized. Existing records without the new reward fields
are interpreted as legacy, and incomplete new configurations are rejected.
No old checkpoint should be silently loaded under the new reward.

The four original coefficients remain (delay,loss,bandwidth,cost)
=(2,3,2,.2). Preserve the constant loss offset initially to avoid
changing the objective further. Explain it as an additive offset,
not a guarantee of conservative behavior or stability. C17 varies
weights and cost assumptions, not test-tuned values.

The submitted p.7 describes cost 7/20 as relative operator preferences.
There is no documented operator contract, price or energy measurement.
Treat them as synthetic positive dimensionless preference indices
(LEO/LTE=20/7), not empirical tariffs, energy or monetary costs.
The original reward used LTE 7 but Ryu logged LTE 5. The runner now
sends the same configured per-tunnel cost and reward used for learning;
the controller logs the provided values instead of a separate map.
Old clients without cost fields log missing values. No assertion of
actual economic savings or proof of data-plane forwarding follows.

Both entry points expose reward mode, references, weights and costs
via CLI (see --help), and existing checkpoint/trace config matching
protects the chosen reward. Greedy and oracle reuse SDWANEnv.reward.
C20 will report cost and QoS separately; C17 will plan controlled
sensitivity experiments. No emulation or manuscript edits in this commit.

Manuscript: p.5 equation (1) and dimensional consistency discussion;
p.7 reference case setup and Fig.3 (synthetic preference index);
p.8 experimental parameters; p.9-10 cost interpretation. Exact
LaTeX replacements deferred until the experimental protocol is validated.
