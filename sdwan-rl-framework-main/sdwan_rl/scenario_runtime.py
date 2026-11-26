"""Apply version-2 profiles and independent background UDP traffic.

Uses checked commands and readback; actual QoS/forwarding must still be
validated in the full emulation. No Mininet import is needed for unit tests.
"""
import json
import re
import subprocess
import time

from .scenario_profiles import TUNNELS, episode_profiles, normalize_v2

INTERFACES = {"lte": ("s1-eth1", "s3-eth1"), "satellite": ("s1-eth2", "s3-eth2")}
PORTS = {"lte": 5101, "satellite": 5102}
PRIORITY = 40000


def checked(node, args):
    out, err, status = node.pexec(args)
    if status != 0:
        raise RuntimeError(f"{node.name}: {args!r} failed ({status}): {err}")
    return out


def netem_child(node, iface):
    qdiscs = json.loads(checked(node, ["tc", "-j", "qdisc", "show", "dev", iface]))
    netem = [q for q in qdiscs if q.get("kind") == "netem"]
    if len(netem) != 1 or not netem[0].get("parent") or not netem[0].get("handle"):
        raise RuntimeError(f"{iface}: expected exactly one child netem qdisc")
    child = netem[0]
    parent_major = child["parent"].split(":")[0]
    if not any(q.get("kind") == "htb" and q.get("handle", "").split(":")[0] == parent_major
               for q in qdiscs):
        raise RuntimeError(f"{iface}: expected HTB parent, refusing to replace root")
    return child


def change_netem(node, iface, profile):
    before = netem_child(node, iface)
    checked(node, ["tc", "qdisc", "change", "dev", iface,
                   "parent", before["parent"], "handle", before["handle"],
                   "netem", "delay", f'{profile["delay_ms"]:g}ms',
                   f'{profile["jitter_ms"]:g}ms', "loss", f'{profile["loss_pct"]:g}%',
                   "limit", str(profile["max_queue"])])
    after = netem_child(node, iface)
    if (before["parent"], before["handle"]) != (after["parent"], after["handle"]):
        raise RuntimeError(f"{iface}: netem hierarchy changed unexpectedly")
    # Raw kernel readback is retained; exact option decoding varies by tc version.
    return after


def ofport(node, iface):
    port = int(checked(node, ["ovs-vsctl", "get", "Interface", iface, "ofport"]).strip())
    if port <= 0:
        raise RuntimeError(f"Invalid OVS port for {iface}")
    return port


def flow_counter(node, cookie):
    output = checked(node, ["ovs-ofctl", "dump-flows", node.name])
    matches = []
    for line in output.splitlines():
        m = re.search(r"cookie=0x([0-9a-fA-F]+)", line)
        if m and int(m.group(1), 16) == cookie:
            count = re.search(r"n_packets=(\d+)", line)
            if count:
                matches.append(int(count.group(1)))
    if len(matches) != 1:
        raise RuntimeError(f"{node.name}: expected one flow with cookie {cookie:#x}")
    return matches[0]


class ScenarioRuntime:
    def __init__(self, net, scenario):
        self.scenario = normalize_v2(scenario)
        self.s1, self.s3, self.h1, self.h2 = net.get("s1", "s3", "h1", "h2")
        self.processes = []
        self.rules = []
        self.forward_rules = {}
        self.last_counters = {}
        self.current = None
        self.readback = {}

    def _rule(self, node, incoming, outgoing, port, direction, cookie):
        match = f"priority={PRIORITY},udp,in_port={incoming},tp_{direction}={port}"
        checked(node, ["ovs-ofctl", "add-flow", node.name,
                       f"cookie={cookie:#x},{match},actions=output:{outgoing}"])
        self.rules.append((node, match))
        return (node, cookie)

    def start_traffic(self):
        for i, tunnel in enumerate(TUNNELS):
            rate = self.scenario["traffic_mbps"][tunnel]
            if rate == 0:
                continue
            left, right = INTERFACES[tunnel]
            a, b = ofport(self.s1, "s1-h1"), ofport(self.s3, "s3-h2")
            l, r = ofport(self.s1, left), ofport(self.s3, right)
            port, cookie = PORTS[tunnel], 0xC100 + 4 * i
            self.forward_rules[tunnel] = [
                self._rule(self.s1, a, l, port, "dst", cookie),
                self._rule(self.s3, r, b, port, "dst", cookie + 1),
            ]
            self._rule(self.s3, b, r, port, "src", cookie + 2)
            self._rule(self.s1, l, a, port, "src", cookie + 3)
            self.last_counters[tunnel] = [flow_counter(*rule) for rule in self.forward_rules[tunnel]]
            server = self.h2.popen(["iperf", "-s", "-u", "-p", str(port)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.processes.append(server)
            time.sleep(0.3)
            if server.poll() is not None:
                raise RuntimeError(f"{tunnel}: UDP server failed to start")
            client = self.h1.popen(["iperf", "-c", self.h2.IP(), "-u", "-p", str(port),
                                    "-b", f"{rate:g}M", "-t", "86400"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.processes.append(client)

    def apply_episode(self, episode):
        desired = episode_profiles(self.scenario, episode)
        for tunnel in TUNNELS:
            if self.current is None or desired[tunnel] != self.current[tunnel]:
                self.readback[tunnel] = {
                    iface: change_netem(node, iface, desired[tunnel])
                    for node, iface in zip((self.s1, self.s3), INTERFACES[tunnel])
                }
        self.current = desired
        return {"requested_profiles": desired, "netem_readback": self.readback,
                "background_offered_mbps": self.scenario["traffic_mbps"]}

    def check_traffic(self):
        if any(proc.poll() is not None for proc in self.processes):
            raise RuntimeError("Background iperf process stopped before run completion")
        report = {}
        for tunnel, rules in self.forward_rules.items():
            counts = [flow_counter(*rule) for rule in rules]
            delta = [new - old for new, old in zip(counts, self.last_counters[tunnel])]
            if delta[0] <= 0 or delta[1] < 0:
                raise RuntimeError(f"{tunnel}: no background ingress traffic or counters reset")
            # Zero egress matches can be legitimate under 100% loss. Record them
            # rather than silently treating offered load as delivered throughput.
            report[tunnel] = {"ingress_flow_packets": delta[0], "egress_flow_packets": delta[1]}
            self.last_counters[tunnel] = counts
        return report

    def close(self):
        for proc in reversed(self.processes):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        for node, match in reversed(self.rules):
            try:
                checked(node, ["ovs-ofctl", "--strict", "del-flows", node.name, match])
            except RuntimeError as exc:
                print(f"[scenario cleanup] {exc}")
