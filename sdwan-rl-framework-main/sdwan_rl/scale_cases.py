"""Explicit, bounded multi-path experimental scenarios (C29-C34).

No Mininet, Ryu or ML imports: topology/route compilation is pure data.
One controlled source-destination pair, optional transit/competing site.
These are logical SD-WAN paths, not encapsulated or radio-level tunnels.
"""
import hashlib
import json
import math

CASES = ("parallel2", "parallel3", "parallel4", "triangle3")
SPLITS = ("train", "calibration", "held_out_test")
ACCESS = 1
PRIMARY_PORT = 5300
SECONDARY_PORT = 9999
PRIMARY_COOKIE = 0xA0000000
STREAM_MASK = 0xfffff000
# Representative synthetic profiles, not measured carrier technologies.
PROFILES = {
    "lte_like": (250, 25, 2, 5, 750, 7),
    "leo_like": (200, 60, 3, 30, 1000, 20),
    "terrestrial_like": (180, 35, 1, 5, 750, 10),
    "alternate_like": (150, 45, 2, 8, 750, 12),
    "transit_like": (180, 18, .5, 3, 750, 5),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def host_ip(site):
    return f"10.0.0.{site}"


def make_edge(identifier, a, b, technology):
    cap, delay, loss, jitter, queue, cost = PROFILES[technology]
    return {"id": identifier, "a": a, "b": b, "profile": technology,
            "capacity_mbps": cap, "delay_ms": delay, "loss_pct": loss,
            "jitter_ms": jitter, "max_queue": queue, "cost_index": cost}


def catalog(case):
    if case not in CASES:
        raise ValueError("Unknown scalability case")
    n = 3 if case == "triangle3" else int(case[-1])
    sites = [1, 2, 3] if case == "triangle3" else [1, 2]
    direct = 2 if case == "triangle3" else n
    profiles = ("lte_like", "leo_like", "terrestrial_like", "alternate_like")
    edges = [make_edge(f"p{i+1}", 1, 2, profiles[i]) for i in range(direct)]
    paths = [{"id": f"path{i+1}", "edges": [e["id"]], "cost_index": e["cost_index"]}
             for i, e in enumerate(edges)]
    if case == "triangle3":
        edges += [make_edge("ac", 1, 3, "transit_like"), make_edge("cb", 3, 2, "transit_like")]
        paths.append({"id": "path3", "edges": ["ac", "cb"], "cost_index": 10})
    # Access port 1; assign stable numeric ports independently at each switch.
    next_port = {site: 2 for site in sites}
    for edge in edges:
        for side in ("a", "b"):
            site = edge[side]
            edge[side + "_port"] = next_port[site]
            next_port[site] += 1
    return {"schema": 1, "case": case, "sites": sites, "primary_pair": [1, 2],
            "edges": edges, "paths": paths}


def make_scenario(case, split, episodes=12, ticks=10):
    """Separate fixed-size study; lock before running any held-out case.

    Small default budget is a diagnostic/study starting point, not a
    claim that a policy converges in this number of updates.
    """
    if split not in SPLITS or type(episodes) is not int or episodes < 6 or type(ticks) is not int or ticks < 2:
        raise ValueError("Invalid split or experiment budget")
    graph = catalog(case)
    if split == "train":
        edge, start, end, delay, loss, rate = "p1", 2, max(2, episodes // 2), 120, 15, 8
    elif split == "calibration":
        edge, start, end, delay, loss, rate = "p1", 3, max(3, episodes // 2 + 1), 140, 18, 10
    else:
        edge = "cb" if case == "triangle3" else graph["edges"][-1]["id"]
        start, end, delay, loss, rate = max(2, episodes // 2), max(3, episodes - 1), 180, 22, 14
    # Every direct candidate has an independent pinned UDP flow. In the
    # triangle the C->B flow shares the cb segment with A->C->B traffic.
    traffic = [{"id": f"load_{e['id']}", "src": 1, "dst": 2,
                "edges": [e["id"]], "port": 6100 + i, "rate_mbps": rate}
               for i, e in enumerate(graph["edges"]) if e["a"] == 1 and e["b"] == 2]
    if case == "triangle3":
        traffic.append({"id": "load_cb", "src": 3, "dst": 2, "edges": ["cb"],
                        "port": 6190, "rate_mbps": rate * 2})
    return {"schema": 1, "id": f"{case}_{split}", "split": split, "graph": graph,
            "episodes": episodes, "ticks_per_episode": ticks, "traffic": traffic,
            "events": [{"edge": edge, "start_episode": start, "end_episode": end,
                        "delay_ms": delay, "loss_pct": loss}]}


def validate_scenario(scenario):
    """Restrict to the versioned catalog and reproducible scenario generator."""
    if not isinstance(scenario, dict) or scenario.get("schema") != 1:
        raise ValueError("Unsupported scalability scenario schema")
    expected = make_scenario(scenario["graph"]["case"], scenario["split"],
                             scenario["episodes"], scenario["ticks_per_episode"])
    if canonical(expected) != canonical(scenario):
        raise ValueError("Scenario differs from the versioned case generator")
    return scenario


def edge_map(graph):
    return {e["id"]: e for e in graph["edges"]}


def walk(graph, edge_ids, src, dst):
    """Return ordered (site, in_port, out_port) including local access hops."""
    edges = edge_map(graph)
    if src == dst or src not in graph["sites"] or dst not in graph["sites"] or not edge_ids:
        raise ValueError("Invalid path endpoints")
    site, incoming = src, ACCESS
    visited, result = {src}, []
    for name in edge_ids:
        edge = edges.get(name)
        if edge is None:
            raise ValueError("Unknown edge")
        if site == edge["a"]:
            following, outgoing, peer = edge["b"], edge["a_port"], edge["b_port"]
        elif site == edge["b"]:
            following, outgoing, peer = edge["a"], edge["b_port"], edge["a_port"]
        else:
            raise ValueError("Disconnected path")
        if following in visited:
            raise ValueError("Routing loops are not supported")
        result.append((site, incoming, outgoing))
        visited.add(following)
        site, incoming = following, peer
    if site != dst:
        raise ValueError("Path does not reach destination")
    result.append((dst, incoming, ACCESS))
    return result


def route(graph, edge_ids, src, dst, udp_port, stream_id):
    """Exact bidirectional UDP forwarding matches with unique stream cookies."""
    if type(stream_id) is not int or not 0 <= stream_id < 65536:
        raise ValueError("Invalid stream id")
    out = []
    for direction in (0, 1):
        origin, destination = (src, dst) if direction == 0 else (dst, src)
        edges = edge_ids if direction == 0 else list(reversed(edge_ids))
        for hop, (site, incoming, outgoing) in enumerate(walk(graph, edges, origin, destination)):
            match = {"in_port": incoming, "eth_type": 0x0800, "ip_proto": 17,
                     "ipv4_src": host_ip(origin), "ipv4_dst": host_ip(destination),
                     ("udp_dst" if direction == 0 else "udp_src"): udp_port}
            out.append({"site": site, "match": match, "out_port": outgoing,
                        "cookie": PRIMARY_COOKIE | (stream_id << 12) | (direction << 11) | hop})
    return out


def primary_rules(graph, action):
    if type(action) is not int or not 0 <= action < len(graph["paths"]):
        raise ValueError("Invalid action for this topology")
    return route(graph, graph["paths"][action]["edges"], 1, 2, PRIMARY_PORT, 0)


def secondary_rules(graph, mode):
    if mode == "off":
        return []
    if mode == "pinned":
        return route(graph, graph["paths"][0]["edges"], 1, 2, SECONDARY_PORT, 200)
    if mode != "rr":
        raise ValueError("Unknown secondary mode")
    # Forward packets are round-robined at A. Transit rules and B ingress
    # matches cover each candidate. Echoes use the fixed first path.
    first = route(graph, graph["paths"][0]["edges"], 1, 2, SECONDARY_PORT, 200)
    source = {**first[0], "out_port": "controller"}
    result = [source] + [r for r in first if "udp_src" in r["match"]]
    seen = {}
    for i, path in enumerate(graph["paths"]):
        for r in route(graph, path["edges"], 1, 2, SECONDARY_PORT, 201 + i):
            if "udp_src" in r["match"] or r["site"] == 1:
                continue
            key = (r["site"], canonical(r["match"]))
            if key in seen:
                if seen[key] != r["out_port"]:
                    raise ValueError("Ambiguous RR transit routing")
            else:
                seen[key] = r["out_port"]
                result.append(r)
    return result


def static_rules(scenario, secondary="off"):
    validate_scenario(scenario)
    graph = scenario["graph"]
    rules = []
    for i, p in enumerate(graph["paths"]):
        rules.extend(route(graph, p["edges"], 1, 2, 5201 + i, i + 1))
    for i, load in enumerate(scenario["traffic"]):
        rules.extend(route(graph, load["edges"], load["src"], load["dst"], load["port"], 100 + i))
    rules.extend(secondary_rules(graph, secondary))
    identities = [(r["site"], canonical(r["match"])) for r in rules]
    if len(set(identities)) != len(identities):
        raise ValueError("Conflicting static flow matches")
    return rules


def episode_profiles(scenario, episode):
    validate_scenario(scenario)
    if type(episode) is not int or not 1 <= episode <= scenario["episodes"]:
        raise ValueError("Invalid episode")
    result = {e["id"]: dict(e) for e in scenario["graph"]["edges"]}
    for event in scenario["events"]:
        if event["start_episode"] <= episode <= event["end_episode"]:
            result[event["edge"]].update(delay_ms=event["delay_ms"], loss_pct=event["loss_pct"])
    return result


def path_residual(graph, path, edge_tx_mbps):
    """Minimum forward residual over edges; not available bandwidth."""
    edges = edge_map(graph)
    rates = [edge_tx_mbps.get(e) for e in path["edges"]]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in rates):
        raise ValueError("Missing or invalid edge TX measurement")
    capacity = min(edges[e]["capacity_mbps"] for e in path["edges"])
    residual = min(max(0.0, edges[e]["capacity_mbps"] - rate)
                   for e, rate in zip(path["edges"], rates))
    return {"nominal_bottleneck_mbps": capacity, "residual_mbps": residual,
            "equivalent_load_mbps": capacity - residual}


def architecture(n):
    if type(n) is not int or n <= 0:
        raise ValueError("N must be a positive integer")
    return {"input": 3 * n, "hidden": [64, 32], "output": n,
            "online_trainable_parameters": 2144 + 225 * n,
            "binary_observation_count_upper_bound": 2 ** (3 * n)}
