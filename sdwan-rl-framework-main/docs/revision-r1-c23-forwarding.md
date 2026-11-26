# C23-C28: isolate measurement rules and apply control before traffic

The revised dual-site topology explicitly assigns OpenFlow 1.3 ports
1=LTE, 2=LEO, 3=access on DPID 1/3, secure mode, STP off. Loop-free
rules send tunnel ingress to the local host, never to another tunnel.
Access links explicitly use their configured TCLink profiles; the old
constructor did not explicitly select TCLink for them. Measured RTTs
will therefore need recalibration. Static host ARP is required.

Start Ryu separately (OpenFlow TCP 6653 and REST 8080). The agent sends
a decision ID and action to /decision BEFORE the primary workload.
Ryu updates primary ingress rules on both switches and waits for
OpenFlow barriers; errors/timeouts fail the run. REST acknowledgement
is not data-plane proof. The collector reads back flow actions and
counter deltas, checks selected-path ingress, other-path receives
and host receives. Late packets can trigger a failure. No atomic
cross-switch update is claimed. /metrics only logs the applied
decision; no ML or duplicate reward/cost computation in Ryu.

UDP 5201/5202 are pinned by higher-priority bidirectional probe rules;
5101/5102 remain independent background traffic. Primary UDP echo
workload uses 5300 and follows the Ryu-installed path. Secondary UDP
9999 packet-level RR is retained on host ingress; validation is C34.
Primary is windowed, not continuous outside measurement windows.
Reward still uses selected-tunnel probe snapshots, not primary QoS.
Probe accounting reports payload bytes and OVS match byte/packet
counters (different accounting layers), not exact wire bytes.
Counter sample intervals enclose the common send/receive window.

Real Mininet/Ryu/OVS path and qdisc/flow readback must be smoke-tested.
Synthetic flow table tests do not establish actual forwarding.
