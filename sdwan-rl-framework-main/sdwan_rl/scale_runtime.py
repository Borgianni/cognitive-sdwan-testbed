"""Mininet/OVS runtime for explicit multi-path scalability cases."""
import json
import re
import secrets
import select
import subprocess
import sys
import time
from pathlib import Path

import requests

from .scale_cases import (ACCESS, PRIMARY_PORT, SECONDARY_PORT, PRIMARY_COOKIE,
                          digest, validate_scenario, primary_rules, static_rules,
                          episode_profiles, path_residual, walk, host_ip)
from .scenario_runtime import checked, ofport, change_netem
from .telemetry_core import throughput_mbps, stream_measurement

WORKER = Path(__file__).with_name("udp_measure.py").resolve()


def create_network(graph):
    from mininet.topo import Topo
    from mininet.link import TCLink
    from mininet.node import OVSSwitch
    from .network_guard import prepare_network

    class ScaleTopo(Topo):
        def build(self):
            for site in graph["sites"]:
                s, h = f"s{site}", f"h{site}"
                self.addSwitch(s, dpid=f"{site:016x}", stp=False, failMode="secure",
                               protocols="OpenFlow13", cls=OVSSwitch)
                self.addHost(h, ip=host_ip(site) + "/24")
                self.addLink(h, s, port1=0, port2=ACCESS, cls=TCLink,
                             intfName1=f"{h}-access", intfName2=f"{s}-access",
                             bw=250, delay="50ms", loss=1, jitter="10ms",
                             max_queue_size=1000, use_htb=True)
            for edge in graph["edges"]:
                a, b = f"s{edge['a']}", f"s{edge['b']}"
                self.addLink(a, b, port1=edge["a_port"], port2=edge["b_port"], cls=TCLink,
                             intfName1=f"{a}-{edge['id']}", intfName2=f"{b}-{edge['id']}",
                             bw=edge["capacity_mbps"], delay=f'{edge["delay_ms"]}ms',
                             loss=edge["loss_pct"], jitter=f'{edge["jitter_ms"]}ms',
                             max_queue_size=edge["max_queue"], use_htb=True)

    return prepare_network(ScaleTopo())


class ScaleAPI:
    def __init__(self, scenario, secondary):
        self.scenario, self.secondary = scenario, secondary
        self.hash = digest(scenario)
        self.sites = scenario["graph"]["sites"]
        self.session = requests.Session()
        self.base = "http://127.0.0.1:8080"

    def wait_ready(self, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = self.session.get(self.base + "/ready", timeout=1)
                r.raise_for_status()
                data = r.json()
                if data.get("scenario_sha256") != self.hash or data.get("secondary") != self.secondary:
                    raise RuntimeError("Wrong Ryu app/scenario/secondary mode")
                if data.get("datapaths") == self.sites:
                    return
            except (requests.RequestException, ValueError):
                pass
            time.sleep(.2)
        raise RuntimeError("Scalability controller not ready")

    def apply(self, action, identifier):
        response = self.session.post(self.base + "/decision", json={"action": action,
                                     "decision_id": identifier}, timeout=10)
        response.raise_for_status()
        ack = response.json()
        expected = {"action": action, "decision_id": identifier, "scenario_sha256": self.hash,
                    "path_id": self.scenario["graph"]["paths"][action]["id"],
                    "datapaths": self.sites, "barrier_acknowledged": True}
        if ack != expected:
            raise RuntimeError("Mismatched controller acknowledgement")
        return ack

    def rr_stats(self):
        response = self.session.get(self.base + "/rr_stats", timeout=5)
        response.raise_for_status()
        counts = response.json()["dispatched_by_path"]
        if len(counts) != len(self.scenario["graph"]["paths"]) or any(type(n) is not int or n < 0 for n in counts):
            raise RuntimeError("Invalid RR dispatch counters")
        return counts

    def close(self):
        self.session.close()


def parse_table(text):
    result = {}
    for line in text.splitlines():
        cookie = re.search(r"cookie=0x([0-9a-fA-F]+)", line)
        if not cookie:
            continue
        c = int(cookie.group(1), 16)
        if c & 0xf0000000 != PRIMARY_COOKIE:
            continue
        packets, size = re.search(r"n_packets=(\d+)", line), re.search(r"n_bytes=(\d+)", line)
        output = re.search(r"actions=output:(\d+)\s*$", line)
        controller = re.search(r"actions=CONTROLLER(?::\d+)?\s*$", line, re.I)
        if not packets or not size or not (output or controller) or c in result:
            raise RuntimeError(f"Invalid monitored OpenFlow rule {c:#x}")
        result[c] = {"packets": int(packets.group(1)), "bytes": int(size.group(1)),
                     "out_port": int(output.group(1)) if output else "controller"}
    return result


def rule_deltas(rules, before, after):
    """Readback verifies exact expected cookie sets and output ports."""
    expected = {site: {} for site in before}
    for rule in rules:
        expected[rule["site"]][rule["cookie"]] = rule["out_port"]
    if set(before) != set(after) or set(before) != set(expected):
        raise RuntimeError("Incomplete datapath readback")
    result = {}
    for site, ports in expected.items():
        if set(before[site]) != set(ports) or set(after[site]) != set(ports):
            raise RuntimeError("Missing or unexpected monitored flow cookie")
        for cookie, port in ports.items():
            a, b = before[site][cookie], after[site][cookie]
            if a["out_port"] != port or b["out_port"] != port:
                raise RuntimeError("Forwarding action differs from route plan")
            row = {k: b[k] - a[k] for k in ("packets", "bytes")}
            if min(row.values()) < 0:
                raise RuntimeError("Flow counters reset during measurement")
            result[str(cookie)] = {**row, "site": site, "out_port": port}
    return result


def read_edge_counters(graph):
    result = {}
    for edge in graph["edges"]:
        # All catalog edges are oriented in the forward direction of their
        # path from A toward B (including ac and cb).
        iface = f"s{edge['a']}-{edge['id']}"
        start = time.monotonic()
        value = int(Path(f"/sys/class/net/{iface}/statistics/tx_bytes").read_text())
        end = time.monotonic()
        result[edge["id"]] = {"time_s": (start + end) / 2, "tx_bytes": value,
                              "read_start_s": start, "read_end_s": end}
    return result


class ScaleRuntime:
    def __init__(self, net, scenario, cfg, secondary, api):
        self.net, self.scenario, self.cfg, self.api = net, validate_scenario(scenario), cfg, api
        self.graph, self.secondary = scenario["graph"], secondary
        self.switches = {i: net.get(f"s{i}") for i in self.graph["sites"]}
        self.static = static_rules(scenario, secondary)
        self.processes, self.profiles = [], {}
        for site, switch in self.switches.items():
            if ofport(switch, f"s{site}-access") != ACCESS:
                raise RuntimeError("Unexpected access port")
        for edge in self.graph["edges"]:
            for side in ("a", "b"):
                site = edge[side]
                if ofport(self.switches[site], f"s{site}-{edge['id']}") != edge[side + "_port"]:
                    raise RuntimeError("Unexpected edge port")

    def apply_episode(self, episode):
        desired = episode_profiles(self.scenario, episode)
        result = {}
        for name, profile in desired.items():
            if self.profiles.get(name) != profile:
                result[name] = {}
                for side in ("a", "b"):
                    site = profile[side]
                    result[name][str(site)] = change_netem(self.switches[site], f"s{site}-{name}", profile)
        self.profiles = desired
        return {"requested": desired, "changed_qdisc_readback": result}

    def start_background(self):
        for load in self.scenario["traffic"]:
            host = self.net.get(f"h{load['src']}")
            target = self.net.get(f"h{load['dst']}")
            port = str(load["port"])
            server = target.popen(["iperf", "-s", "-u", "-p", port],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.processes.append(server)
            time.sleep(.25)
            if server.poll() is not None:
                raise RuntimeError("Background server failed")
            client = host.popen(["iperf", "-c", target.IP(), "-u", "-p", port,
                                 "-b", f'{load["rate_mbps"]:g}M', "-t", "86400"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.processes.append(client)

    def tables(self):
        return {site: parse_table(checked(switch, ["ovs-ofctl", "-O", "OpenFlow13", "--no-names",
                                                   "dump-flows", switch.name]))
                for site, switch in self.switches.items()}

    def collect(self, action):
        cfg = self.cfg
        paths = self.graph["paths"]
        streams = {"primary": {"port": PRIMARY_PORT, "packets": cfg.primary_packets,
                               "payload_bytes": cfg.primary_payload_bytes}}
        for i, path in enumerate(paths):
            streams[path["id"]] = {"port": 5201 + i, "packets": cfg.probe_packets,
                                    "payload_bytes": cfg.probe_payload_bytes}
        if self.secondary != "off":
            streams["secondary"] = {"port": SECONDARY_PORT, "packets": cfg.primary_packets,
                                    "payload_bytes": cfg.primary_payload_bytes}
        rules = self.static + primary_rules(self.graph, action)
        before = self.tables()
        # Validate the installed route before any primary/probe sends.
        rule_deltas(rules, before, before)
        rr_before = self.api.rr_stats() if self.secondary == "rr" else None
        start = time.monotonic() + 1.0
        deadline = start + cfg.window_s + cfg.grace_s
        spec = {"start_s": start, "window_s": cfg.window_s, "grace_s": cfg.grace_s,
                "token": secrets.token_hex(16), "client_ip": host_ip(1), "server_ip": host_ip(2),
                "streams": streams}
        processes = []
        def spawn(site, role):
            p = self.net.get(f"h{site}").popen([sys.executable, str(WORKER), role, json.dumps(spec)],
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(p)
            return p
        try:
            server = spawn(2, "server")
            ready, _, _ = select.select([server.stdout], [], [], .75)
            if not ready or server.stdout.readline().strip() != "READY":
                raise RuntimeError("Scale UDP receiver not ready")
            client = spawn(1, "client")
            time.sleep(max(0.0, start - .03 - time.monotonic()))
            counters0 = read_edge_counters(self.graph)
            if any(c["read_end_s"] >= start for c in counters0.values()):
                raise RuntimeError("Missed common measurement window")
            def receive(p):
                out, err = p.communicate(timeout=max(1, deadline - time.monotonic() + 3))
                if p.returncode:
                    raise RuntimeError(f"UDP measurement worker failed: {err}")
                return json.loads(out)
            cr, sr = receive(client), receive(server)
            counters1 = read_edge_counters(self.graph)
            after = self.tables()
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate()
        if any(p.poll() is not None for p in self.processes):
            raise RuntimeError("Competing traffic terminated early")
        rates = {}
        for name in counters0:
            if counters1[name]["time_s"] < deadline:
                raise RuntimeError("Counter support ended before receive deadline")
            rates[name] = throughput_mbps(counters0[name], counters1[name])
        samples = {name: stream_measurement(cr["streams"][name], sr["streams"][name],
                                             stream, start, deadline, cfg.window_s)
                   for name, stream in streams.items()}
        deltas = rule_deltas(rules, before, after)
        # Check that all forward source rules were actually exercised and
        # that host receives are supported by destination switch counts.
        for load in self.scenario["traffic"]:
            origin = [r for r in rules if r["site"] == load["src"]
                      and r["match"].get("udp_dst") == load["port"]
                      and r["match"]["in_port"] == ACCESS]
            if len(origin) != 1 or deltas[str(origin[0]["cookie"])]["packets"] <= 0:
                raise RuntimeError("No competing flow matched the installed route")
        for name, stream in streams.items():
            port = stream["port"]
            ingress = [r for r in rules if r["site"] == 1 and r["match"].get("udp_dst") == port
                       and r["match"]["in_port"] == ACCESS]
            egress = [r for r in rules if r["site"] == 2 and r["match"].get("udp_dst") == port
                      and r["out_port"] == ACCESS]
            reverse_egress = [r for r in rules if r["site"] == 1 and r["match"].get("udp_src") == port
                              and r["out_port"] == ACCESS]
            if (len(ingress) != 1 or deltas[str(ingress[0]["cookie"])]["packets"] <= 0
                    or samples[name]["received_packets"] > sum(deltas[str(r["cookie"])]["packets"] for r in egress)
                    or samples[name]["echo_received_packets"] > sum(deltas[str(r["cookie"])]["packets"] for r in reverse_egress)):
                raise RuntimeError("Traffic does not match the expected ingress/egress evidence")
        rr, rr_egress = None, None
        if rr_before is not None:
            rr = [b - a for a, b in zip(rr_before, self.api.rr_stats())]
            origin = [r for r in rules if r["site"] == 1 and r["match"].get("udp_dst") == SECONDARY_PORT
                      and r["out_port"] == "controller"]
            rr_egress = []
            for path in paths:
                incoming = walk(self.graph, path["edges"], 1, 2)[-1][1]
                final = [r for r in rules if r["site"] == 2 and r["match"].get("udp_dst") == SECONDARY_PORT
                         and r["match"]["in_port"] == incoming]
                if len(final) != 1:
                    raise RuntimeError("Missing unique RR path receive rule")
                rr_egress.append(deltas[str(final[0]["cookie"])]["packets"])
            if (len(origin) != 1 or min(rr) < 0 or sum(rr) <= 0 or max(rr) - min(rr) > 1
                    or sum(rr) > deltas[str(origin[0]["cookie"])]["packets"]
                    or any(received > dispatched for received, dispatched in zip(rr_egress, rr))):
                raise RuntimeError("RR dispatch/receive counters are inconsistent")
        metrics = {}
        for path in paths:
            name = path["id"]
            metrics[name] = {"delay": samples[name]["rtt_ms"], "packet_loss": samples[name]["loss_pct"],
                             **path_residual(self.graph, path, rates)}
        probe_bytes = sum(samples[p["id"]]["forward_payload_bytes"] + samples[p["id"]]["reverse_payload_bytes"]
                          for p in paths)
        # All probe source/echo matches, not transit/egress matches.
        probe_ports = {5201 + i for i in range(len(paths))}
        match_bytes = sum(deltas[str(r["cookie"])]["bytes"] for r in rules
                          if r["match"]["in_port"] == ACCESS and
                          (r["match"].get("udp_dst") in probe_ports or r["match"].get("udp_src") in probe_ports))
        return {"metrics": metrics, "streams": samples, "flow_deltas": deltas, "rr_dispatched": rr, "rr_path_egress_matches": rr_egress,
                "edge_tx_mbps": rates, "counters_before": counters0, "counters_after": counters1,
                "window_start_s": start, "window_end_s": deadline,
                "probe_payload_bytes_both_directions": probe_bytes,
                # Align with existing primary_summary accounting interface.
                "probe_ovs_ingress_match_bytes": match_bytes}

    def close(self):
        for proc in reversed(self.processes):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
