"""Dual-site OpenFlow 1.3 controller, independent of ML libraries."""
import csv
import json
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from webob import Response

DPIDS = (1, 3)
ACCESS = 3
PRIMARY_PORT = 5300
CSV_FIELDS = ("episode", "tick", "decision_id", "action", "delay_lte", "throughput_lte",
              "packet_loss_lte", "delay_satellite", "throughput_satellite", "packet_loss_satellite",
              "cost_lte", "cost_satellite", "reward", "total_reward", "selected_probe_loss_pct")


def reply(value, status=200):
    return Response(status=status, content_type="application/json",
                    body=json.dumps(value).encode("utf-8"))


class MetricsController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app = data["sdwan_controller"]

    @route("ready", "/ready", methods=["GET"])
    def ready(self, request):
        return reply({"datapaths": sorted(self.app.datapaths)})

    @route("decision", "/decision", methods=["POST"])
    def receive_decision(self, request):
        try:
            return reply(self.app.apply_decision(request.json_body))
        except Exception as exc:
            self.app.logger.exception("Decision failed")
            return reply({"error": str(exc)}, 503)

    @route("metrics", "/metrics", methods=["POST"])
    def receive_metrics(self, request):
        try:
            link = self.app.process_metrics(request.json_body)
            return Response(status=200, body=link.encode("utf-8"), content_type="text/plain")
        except Exception as exc:
            self.app.logger.exception("Metrics rejected")
            return reply({"error": str(exc)}, 409)

    @route("current", "/link_decision", methods=["GET"])
    def current(self, request):
        return Response(status=200, body=self.app.current_link.encode("utf-8"),
                        content_type="text/plain")


class SDWANController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.errors = {}
        self.pending = {}
        self.lock = hub.Semaphore(1)
        self.current_link = "lte"
        self.current_decision = None
        self.rr_next = {1: 1, 3: 1}
        self.log_file = "network_metrics.log"
        with open(self.log_file, "w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=CSV_FIELDS).writeheader()
        kwargs["wsgi"].register(MetricsController, {"sdwan_controller": self})

    def add_flow(self, dp, priority, match, outputs, cookie):
        ofp, parser = dp.ofproto, dp.ofproto_parser
        actions = [parser.OFPActionOutput(port) for port in outputs]
        instructions = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=priority, match=match,
                                     cookie=cookie, instructions=instructions))

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        if dp.id not in DPIDS:
            return
        self.errors.pop(dp.id, None)
        p, ofp = dp.ofproto_parser, dp.ofproto
        self.add_flow(dp, 0, p.OFPMatch(), [], 0)
        for port in (1, 2):
            self.add_flow(dp, 100, p.OFPMatch(in_port=port, eth_type=0x0800),
                          [ACCESS], 0xd000 + dp.id * 16 + port)
            udp = {"udp_dst": PRIMARY_PORT} if dp.id == 3 else {"udp_src": PRIMARY_PORT}
            self.add_flow(dp, 150, p.OFPMatch(in_port=port, eth_type=0x0800, ip_proto=17, **udp),
                          [ACCESS], 0xd200 + dp.id * 16 + port)
        self.add_flow(dp, 200, p.OFPMatch(in_port=ACCESS, eth_type=0x0800, ip_proto=17, udp_dst=9999),
                      [ofp.OFPP_CONTROLLER], 0xd400 + dp.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change(self, ev):
        dp = ev.datapath
        if dp.id not in DPIDS:
            return
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif self.datapaths.get(dp.id) is dp:
            self.datapaths.pop(dp.id, None)
            self.current_decision = None

    @set_ev_cls(ofp_event.EventOFPErrorMsg, [CONFIG_DISPATCHER, MAIN_DISPATCHER])
    def openflow_error(self, ev):
        msg = ev.msg
        self.errors[msg.datapath.id] = f"type={msg.type}, code={msg.code}, xid={msg.xid}"
        self.logger.error("OpenFlow error: %s", self.errors[msg.datapath.id])

    @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
    def barrier_reply(self, ev):
        event = self.pending.get((ev.msg.datapath.id, ev.msg.xid))
        if event is not None:
            event.set()

    def apply_decision(self, data):
        action, decision_id = data.get("action"), data.get("decision_id")
        if type(action) is not int or action not in (0, 1) or not isinstance(decision_id, str) or not decision_id:
            raise ValueError("Expected action 0/1 and a decision identifier")
        with self.lock:
            if sorted(self.datapaths) != list(DPIDS) or self.errors:
                raise RuntimeError("Datapaths not ready or prior OpenFlow error")
            port, waits = action + 1, []
            self.current_decision = None
            try:
                for dpid in DPIDS:
                    dp = self.datapaths[dpid]
                    p = dp.ofproto_parser
                    self.add_flow(dp, 100, p.OFPMatch(in_port=ACCESS, eth_type=0x0800),
                                  [port], 0xd100 + dpid)
                    udp = {"udp_dst": PRIMARY_PORT} if dpid == 1 else {"udp_src": PRIMARY_PORT}
                    self.add_flow(dp, 150, p.OFPMatch(in_port=ACCESS, eth_type=0x0800, ip_proto=17, **udp),
                                  [port], 0xd300 + dpid)
                    req = p.OFPBarrierRequest(dp)
                    dp.set_xid(req)
                    event = hub.Event()
                    key = (dpid, req.xid)
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
                if self.errors or sorted(self.datapaths) != list(DPIDS):
                    raise RuntimeError("OpenFlow error or datapath disconnected")
            finally:
                for key, _ in waits:
                    self.pending.pop(key, None)
            self.current_link = "lte" if action == 0 else "satellite"
            self.current_decision = (decision_id, action)
            return {"decision_id": decision_id, "action": action, "link": self.current_link,
                    "datapaths": list(DPIDS), "barrier_acknowledged": True}

    def process_metrics(self, metrics):
        episode = metrics["episode"]
        action = episode["chosen link"]
        with self.lock:
            if self.current_decision != (episode["decision_id"], action):
                raise ValueError("Metrics do not correspond to the acknowledged decision")
            row = {"episode": episode["episode number"], "tick": episode["tick"],
                   "decision_id": episode["decision_id"], "action": action,
                   "reward": episode["reward"], "total_reward": episode["total reward"]}
            for name in ("lte", "satellite"):
                m = metrics[name]
                for key in ("delay", "throughput", "packet_loss"):
                    row[f"{key}_{name}"] = m[key]
                row[f"cost_{name}"] = m["cost_index"]
            row["selected_probe_loss_pct"] = metrics[self.current_link]["packet_loss"]
            with open(self.log_file, "a", newline="") as stream:
                csv.DictWriter(stream, fieldnames=CSV_FIELDS).writerow(row)
            return self.current_link

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg, dp = ev.msg, ev.msg.datapath
        if dp.id not in DPIDS or msg.match.get("in_port") != ACCESS:
            return
        from ryu.lib.packet import packet, udp
        segment = packet.Packet(msg.data).get_protocol(udp.udp)
        if segment is None or segment.dst_port != 9999:
            return
        port = self.rr_next[dp.id]
        self.rr_next[dp.id] = 3 - port
        p, ofp = dp.ofproto_parser, dp.ofproto
        dp.send_msg(p.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=ACCESS,
                                  actions=[p.OFPActionOutput(port)],
                                  data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None))
