# C29/C34: physical Mininet realization and packet order diagnostic

The new scale runtime constructs the catalog as explicit TCLink edges,
one switch and host per site, static host ARP, secure OF13 bridges.
It applies child-NetEm impairment events and preserves readback.
For each candidate, pinned concurrent UDP probes travel its exact
forward/reverse route; primary follows the acknowledged controller
action. Competing iperf flows use independent UDP ports and exact
routes; C->B traffic shares the cb transit segment. Counts are checked
before/after every measurement. Forward interface TX measurements
yield per-edge residual and the minimum path residual, not actual
end-to-end throughput. This is a single controlled A->B decision
unit, not a joint multi-site policy.

C34: ordinary scale runs have secondary diagnostic off and pinned
competing flows. A separate fixed-action calibration run may add
a UDP/9999 secondary stream, either pinned to path 1 or distributed
packet-by-packet by a Ryu packet-in and PacketOut rotation over
candidate ingress ports. The two arms use equal packet counts,
payload, primary route, background profiles and send window.
In RR, the echo path is fixed to path 1. The destination counts
unique sequences smaller than the maximum sequence already seen
and records the maximum sequence lag. This is a transparent
late-arrival indicator, not an RFC 4737-complete reordering metric.
Loss, RTT, late-arrival fraction and receive coverage are separate.
Controller dispatch counts are compared to downstream flow
matches and receiver counts; dispatch is not delivery. No result
or absence of perturbation is asserted.

This runtime is a candidate implementation; actual OVS/NetEm and
Ryu compatibility, counter timing, packet-in load and primary
forwarding still need a real smoke test.
