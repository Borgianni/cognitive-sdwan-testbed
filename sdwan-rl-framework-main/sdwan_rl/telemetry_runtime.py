"""Concurrent telemetry and independent primary forwarding evidence."""
import json
import re
import secrets
import select
import subprocess
import sys
import time
from pathlib import Path

from .scenario_runtime import checked, ofport
from .telemetry_core import TelemetryConfig, TUNNELS, PORTS, stream_specs, stream_measurement, throughput_mbps

PROBE_PRIORITY = 41000
INTERFACES = {"lte": "s1-eth1", "satellite": "s1-eth2"}
WORKER = Path(__file__).with_name("udp_measure.py").resolve()


def flow_table(node):
    text = checked(node, ["ovs-ofctl", "-O", "OpenFlow13", "--no-names", "dump-flows", node.name])
    entries = {}
    for line in text.splitlines():
        cookie = re.search(r"cookie=0x([0-9a-fA-F]+)", line)
        packets = re.search(r"n_packets=(\d+)", line)
        size = re.search(r"n_bytes=(\d+)", line)
        output = re.search(r"actions=output:(\d+)\s*$", line)
        if cookie and packets and size and output:
            key = int(cookie.group(1), 16)
            if key in entries:
                raise RuntimeError("Duplicate monitored cookie")
            entries[key] = {"packets": int(packets.group(1)), "bytes": int(size.group(1)),
                            "output": int(output.group(1))}
    return entries


def require_flow(table, cookie, output):
    row = table.get(cookie)
    if row is None or row["output"] != output:
        raise RuntimeError(f"Missing/wrong forwarding rule {cookie:#x}, expected output {output}")
    return row


def counter_delta(before, after):
    values = {k: after[k] - before[k] for k in ("packets", "bytes")}
    if min(values.values()) < 0:
        raise RuntimeError("Monitored flow counter reset during measurement")
    return values


def flow_evidence(before, after, action, probes=True):
    selected = action + 1
    result = {"primary": {}, "probes": {}}
    for dpid in (1, 3):
        a, b = before[dpid], after[dpid]
        for cookie in (0xd100 + dpid, 0xd300 + dpid):
            require_flow(a, cookie, selected)
            require_flow(b, cookie, selected)
        ingress = counter_delta(a[0xd300 + dpid], b[0xd300 + dpid])
        paths = {}
        for port in (1, 2):
            cookie = 0xd200 + dpid * 16 + port
            paths[port] = counter_delta(require_flow(a, cookie, 3), require_flow(b, cookie, 3))
        if paths[3 - selected]["packets"] != 0:
            raise RuntimeError("Primary packets observed on non-selected tunnel (including late packets)")
        if dpid == 1 and ingress["packets"] <= 0:
            raise RuntimeError("No primary traffic matched the controller forwarding rule")
        result["primary"][str(dpid)] = {"ingress": ingress,
                                           "selected_tunnel_receive": paths[selected],
                                           "other_tunnel_receive": paths[3 - selected]}
    if probes:
        for index, tunnel in enumerate(TUNNELS):
            port, cookie = index + 1, 0xc200 + index * 4
            rules = ((1, cookie, port), (3, cookie + 1, 3),
                     (3, cookie + 2, port), (1, cookie + 3, 3))
            values = [counter_delta(require_flow(before[d], c, p),
                                    require_flow(after[d], c, p)) for d, c, p in rules]
            if values[0]["packets"] <= 0:
                raise RuntimeError(f"No {tunnel} probe traffic matched pinned ingress")
            result["probes"][tunnel] = dict(zip(("forward_ingress", "forward_egress",
                                                "reverse_ingress", "reverse_egress"), values))
    return result


def read_interface_counters():
    counters = {}
    for tunnel, iface in INTERFACES.items():
        start = time.monotonic()
        value = int(Path(f"/sys/class/net/{iface}/statistics/tx_bytes").read_text().strip())
        end = time.monotonic()
        counters[tunnel] = {"time_s": (start + end) / 2, "tx_bytes": value,
                            "read_start_s": start, "read_end_s": end}
    return counters


class TelemetryCollector:
    def __init__(self, net, cfg=None, probes=True):
        self.cfg = cfg if cfg is not None else TelemetryConfig()
        self.probes = probes
        self.h1, self.h2, self.s1, self.s3 = net.get("h1", "h2", "s1", "s3")
        self.rules = []
        for node, host in ((self.s1, "h1"), (self.s3, "h2")):
            for iface, port in ((f"{node.name}-eth1", 1), (f"{node.name}-eth2", 2),
                                (f"{node.name}-{host}", 3)):
                if ofport(node, iface) != port:
                    raise RuntimeError(f"Unexpected OpenFlow port mapping: {iface}")
            if node.inNamespace:
                raise RuntimeError("Interface counters require root-namespace OVS switches")

    def _rule(self, node, incoming, outgoing, udp_port, direction, cookie):
        match = f"priority={PROBE_PRIORITY},udp,in_port={incoming},tp_{direction}={udp_port}"
        checked(node, ["ovs-ofctl", "-O", "OpenFlow13", "add-flow", node.name,
                       f"cookie={cookie:#x},{match},actions=output:{outgoing}"])
        self.rules.append((node, match))

    def install(self):
        if not self.probes:
            return
        for index, tunnel in enumerate(TUNNELS):
            port, udp_port, cookie = index + 1, PORTS[tunnel], 0xc200 + index * 4
            self._rule(self.s1, 3, port, udp_port, "dst", cookie)
            self._rule(self.s3, port, 3, udp_port, "dst", cookie + 1)
            self._rule(self.s3, 3, port, udp_port, "src", cookie + 2)
            self._rule(self.s1, port, 3, udp_port, "src", cookie + 3)

    def _tables(self):
        return {1: flow_table(self.s1), 3: flow_table(self.s3)}

    def collect(self, action):
        if type(action) is not int or action not in (0, 1):
            raise ValueError("Invalid selected tunnel")
        begin = time.monotonic()
        before_flows = self._tables()
        for dpid in (1, 3):
            require_flow(before_flows[dpid], 0xd100 + dpid, action + 1)
            require_flow(before_flows[dpid], 0xd300 + dpid, action + 1)
        cfg = self.cfg
        start = time.monotonic() + 1.0
        deadline = start + cfg.window_s + cfg.grace_s
        spec = {"start_s": start, "window_s": cfg.window_s, "grace_s": cfg.grace_s,
                "token": secrets.token_hex(16), "client_ip": self.h1.IP(), "server_ip": self.h2.IP(),
                "streams": stream_specs(cfg, self.probes)}
        processes = []
        def spawn(host, role):
            proc = host.popen([sys.executable, str(WORKER), role, json.dumps(spec)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(proc)
            return proc
        try:
            server = spawn(self.h2, "server")
            readable, _, _ = select.select([server.stdout], [], [], .75)
            if not readable or server.stdout.readline().strip() != "READY":
                raise RuntimeError("UDP receiver did not become ready")
            client = spawn(self.h1, "client")
            time.sleep(max(0.0, start - .03 - time.monotonic()))
            before = read_interface_counters()
            if any(v["read_end_s"] >= start for v in before.values()):
                raise RuntimeError("Counter acquisition missed common send-window start")
            def receive(proc):
                out, err = proc.communicate(timeout=max(1, deadline - time.monotonic() + 3))
                if proc.returncode != 0:
                    raise RuntimeError(f"UDP worker failed: {err}")
                return json.loads(out)
            client_result = receive(client)
            server_result = receive(server)
            after = read_interface_counters()
            after_flows = self._tables()
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate()
        streams = {name: stream_measurement(client_result["streams"][name],
                                             server_result["streams"][name], stream,
                                             start, deadline, cfg.window_s)
                   for name, stream in spec["streams"].items()}
        evidence = flow_evidence(before_flows, after_flows, action, self.probes)
        if (streams["primary"]["received_packets"] > evidence["primary"]["3"]["selected_tunnel_receive"]["packets"]
                or streams["primary"]["echo_received_packets"] > evidence["primary"]["1"]["selected_tunnel_receive"]["packets"]):
            raise RuntimeError("Primary receives exceed selected-path forwarding evidence")
        if self.probes:
            for tunnel in TUNNELS:
                counts = evidence["probes"][tunnel]
                if (streams[tunnel]["received_packets"] > counts["forward_egress"]["packets"]
                        or streams[tunnel]["echo_received_packets"] > counts["reverse_egress"]["packets"]):
                    raise RuntimeError(f"{tunnel} receives exceed pinned-path forwarding evidence")
        metrics, windows = {}, {}
        for tunnel in TUNNELS:
            if after[tunnel]["time_s"] < deadline:
                raise RuntimeError("Interface counter ended before receive deadline")
            if self.probes:
                stream = streams[tunnel]
                metrics[tunnel] = {"delay": stream["rtt_ms"], "packet_loss": stream["loss_pct"],
                                   "throughput": throughput_mbps(before[tunnel], after[tunnel])}
                windows[tunnel] = {
                    "rtt": {"start_s": stream["send_start_s"], "end_s": deadline},
                    "loss": {"start_s": stream["send_start_s"], "end_s": deadline},
                    "throughput": {"start_s": before[tunnel]["time_s"], "end_s": after[tunnel]["time_s"]},
                }
        probe_payload = sum(streams[t]["forward_payload_bytes"] + streams[t]["reverse_payload_bytes"]
                            for t in TUNNELS if t in streams)
        return {"metrics": metrics, "measurement_windows": windows if self.probes else None,
                "streams": streams, "flow_evidence": evidence,
                "interface_counters": {"before": before, "after": after},
                "probe_payload_bytes_both_directions": probe_payload,
                "probe_payload_mbps_both_directions": probe_payload * 8 / (1e6 * cfg.window_s),
                "window_start_s": start, "window_end_s": deadline,
                "collector_elapsed_s": time.monotonic() - begin}

    def close(self):
        for node, match in reversed(self.rules):
            checked(node, ["ovs-ofctl", "-O", "OpenFlow13", "--strict", "del-flows", node.name, match])
        self.rules.clear()
