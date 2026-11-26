# Reviewer 1 C29-C34: bounded scalability extension

Keep the original dual-tunnel runner and its C01-C28 protocol for the
main comparison. Add a separate versioned scalability study, not an
unvalidated rewrite of its measurement and checkpoint formats.

Four cases: parallel2, parallel3, parallel4 (two sites); triangle3
(three sites, two direct paths and A-C-B transit candidate). A real
competing C->B pinned UDP flow shares the cb link with the transit
candidate. Direct paths carry independent competing flows. These
are explicitly provisioned logical paths, not arbitrary discovered
topologies or a multi-agent globally optimal WAN policy. In each
case there is one primary source-destination decision unit A->B.
Train/test schedules are distinct, fixed by a versioned generator,
with different affected link, severity and offered load where
applicable. Study budgets are provisional, not convergence claims.

DQN architecture 3N-64-32-N has 2144+225N trainable online parameters.
Binary observation space has at most 2^(3N) combinations; neither
parameter growth nor source-code reuse establishes sample efficiency
or transfer across N. Train fresh models per graph/path catalog and
seed, keep ordered path IDs in provenance; no reuse of a trained
checkpoint with another N or reordered action semantics. Continuous
features keep six/nine/twelve inputs without changing the DQN class.
For multi-hop paths estimate residual as minimum per-edge forward
residual; this is a bottleneck proxy, not an application throughput
or measured available-bandwidth guarantee.

C32-C33: model pre-provisioned site-to-site logical path choices,
traffic classes, policy steering, measured QoS and synthetic cost.
No GRE/IPsec/VXLAN establishment, encryption, orchestration, vendor
SLA protocol, radio scheduling, orbital model, Doppler or handover.
Synthetic link profiles inspired by terrestrial/LTE/LEO conditions,
not validated technology-specific emulators. Time-varying impairment
profiles are configurable abstractions, not orbital predictions.
Do not add protocol stacks merely to satisfy an overbroad claim.

C34: main revised experiments use pinned UDP competing flows instead
of per-packet striping; this is a workload design change, not a
reproduction of old figures. The old dual-site controller supports
UDP/9999 packet-in round robin but the main runner does not generate
that secondary stream. A dedicated optional diagnostic will compare
a fixed secondary path to packet-in round robin with identical
offered secondary load. Forward receiver order is measured by
sequence numbers; reverse echo is pinned to first path in RR.
No claim that reordering is absent or commercially representative.
