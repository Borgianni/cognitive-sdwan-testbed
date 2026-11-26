"""Ryu app for the bounded C29-C34 scalability scenarios.

Start a fresh process for each run with SDWAN_SCALE_SCENARIO and
SDWAN_SCALE_SECONDARY (off/pinned/rr). No TensorFlow dependency.
"""
import json
import os
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from webob import Response
from sdwan_rl.scale_cases import (validate_scenario, digest, static_rules, primary_rules,
                                  walk, PRIMARY_COOKIE, STREAM_MASK, SECONDARY_PORT)


def response(value, status=200):
    return Response(status=status, content_type="application/json",
                    body=json.dumps(value).encode("utf-8"))


class REST(ControllerBase):
    def __init__(self, req, link, data, **kwargs):
        super().__init__(req, link, data, **kwargs)
        self.app = data["app"]

    @route("ready", "/ready", methods=["GET"])
    def ready(self, request):
        return response({"scenario_sha256": self.app.scenario_hash,
                         "secondary": self.app.secondary, "datapaths": sorted(self.app.datapaths)})

    @route("decision", "/decision", methods=["POST"])
    def decision(self, request):
        try:
            return response(self.app.apply(request.json_body))
        except Exception as exc:
            self.app.logger.exception("Scalability decision failed")
            return response({"error": str(exc)}, 503)

    @route("rr_stats", "/rr_stats", methods=["GET"])
    def rr_stats(self, request):
        return response({"dispatched_by_path": list(self.app.rr_counts)})


class ScaleController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(os.environ["SDWAN_SCALE_SCENARIO"], encoding="utf-8") as stream:
            self.scenario = validate_scenario(json.load(stream))
        self.graph = self.scenario["graph"]
        self.secondary = os.environ.get("SDWAN_SCALE_SECONDARY", "off")
        if self.secondary != "off" and self.scenario["split"] != "calibration":
            raise ValueError("Secondary packet-order diagnostic is calibration only")
        self.scenario_hash = digest(self.scenario)
        self.static = static_rules(self.scenario, self.secondary)
        self.datapaths, self.errors, self.pending = {}, {}, {}
        self.lock = hub.Semaphore(1)
        self.rr_index = 0
        self.rr_counts = [0] * len(self.graph["paths"])
        self.rr_ports = [walk(self.graph, p["edges"], 1, 2)[0][2] for p in self.graph["paths"]]
        kwargs["wsgi"].register(REST, {"app": self})

    def add(self, dp, rule):
        p, ofp = dp.ofproto_parser, dp.ofproto
        out = rule["out_port"]
        actions = ([p.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
                   if out == "controller" else [p.OFPActionOutput(out)])
        dp.send_msg(p.OFPFlowMod(datapath=dp, cookie=rule["cookie"], priority=200,
                                 match=p.OFPMatch(**rule["match"]),
                                 instructions=[p.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]))

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def features(self, ev):
        dp = ev.msg.datapath
        if dp.id not in self.graph["sites"]:
            return
        self.errors.pop(dp.id, None)
        p = dp.ofproto_parser
        # Fresh bridges on each run. No L2 learning/flooding loops.
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=0, match=p.OFPMatch(), instructions=[]))
        for rule in self.static:
            if rule["site"] == dp.id:
                self.add(dp, rule)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def change(self, ev):
        dp = ev.datapath
        if dp.id not in self.graph["sites"]:
            return
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif self.datapaths.get(dp.id) is dp:
            self.datapaths.pop(dp.id, None)
            self.errors[dp.id] = "disconnected"

    @set_ev_cls(ofp_event.EventOFPErrorMsg, [CONFIG_DISPATCHER, MAIN_DISPATCHER])
    def error(self, ev):
        msg = ev.msg
        self.errors[msg.datapath.id] = f"{msg.type}:{msg.code}:{msg.xid}"
        self.logger.error("OpenFlow error %s", self.errors[msg.datapath.id])

    @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
    def barrier(self, ev):
        event = self.pending.get((ev.msg.datapath.id, ev.msg.xid))
        if event is not None:
            event.set()

    def apply(self, data):
        action, decision_id = data.get("action"), data.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("Decision identifier required")
        rules = primary_rules(self.graph, action)
        with self.lock:
            if sorted(self.datapaths) != self.graph["sites"] or self.errors:
                raise RuntimeError("Missing datapath or OpenFlow error")
            waits = []
            try:
                for site in self.graph["sites"]:
                    dp = self.datapaths[site]
                    p, ofp = dp.ofproto_parser, dp.ofproto
                    # Delete only primary stream; leave all probes and competing
                    # flows installed. Windowed primary traffic is not running.
                    dp.send_msg(p.OFPFlowMod(datapath=dp, cookie=PRIMARY_COOKIE,
                                             cookie_mask=STREAM_MASK, table_id=0,
                                             command=ofp.OFPFC_DELETE, out_port=ofp.OFPP_ANY,
                                             out_group=ofp.OFPG_ANY, match=p.OFPMatch()))
                    for rule in rules:
                        if rule["site"] == site:
                            self.add(dp, rule)
                    req = p.OFPBarrierRequest(dp)
                    dp.set_xid(req)
                    key, event = (site, req.xid), hub.Event()
                    self.pending[key] = event
                    waits.append((key, event))
                    dp.send_msg(req)
                timeout = hub.Timeout(5)
                try:
                    for _, event in waits:
                        event.wait()
                except hub.Timeout:
                    raise RuntimeError("OpenFlow barrier timeout")
                finally:
                    timeout.cancel()
                if self.errors or sorted(self.datapaths) != self.graph["sites"]:
                    raise RuntimeError("Datapath error during installation")
            finally:
                for key, _ in waits:
                    self.pending.pop(key, None)
        return {"action": action, "decision_id": decision_id, "scenario_sha256": self.scenario_hash,
                "path_id": self.graph["paths"][action]["id"], "datapaths": self.graph["sites"],
                "barrier_acknowledged": True}

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg, dp = ev.msg, ev.msg.datapath
        if self.secondary != "rr" or dp.id != 1 or msg.match.get("in_port") != 1:
            return
        from ryu.lib.packet import packet, ipv4, udp
        pkt = packet.Packet(msg.data)
        ip, segment = pkt.get_protocol(ipv4.ipv4), pkt.get_protocol(udp.udp)
        if ip is None or segment is None or ip.src != "10.0.0.1" or ip.dst != "10.0.0.2" or segment.dst_port != SECONDARY_PORT:
            return
        index = self.rr_index
        self.rr_index = (index + 1) % len(self.rr_ports)
        self.rr_counts[index] += 1
        p, ofp = dp.ofproto_parser, dp.ofproto
        dp.send_msg(p.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=1,
                                  actions=[p.OFPActionOutput(self.rr_ports[index])],
                                  data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None))
