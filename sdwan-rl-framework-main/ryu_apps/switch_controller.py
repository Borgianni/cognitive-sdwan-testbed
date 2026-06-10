
from ryu.controller import controller
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from ryu.lib.packet import packet 
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
import json
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from webob import Response
import os
import numpy as np
import tensorflow as tf


import random
import csv
#from sdwan_topology import get_link_cost  # Importing the function
import sys
import gym
import logging
from gym import spaces
from tensorflow.keras import layers, models, optimizers
tf.config.set_visible_devices([], 'GPU')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from collections import deque
import random
import matplotlib.pyplot as plt
import re
sys.setrecursionlimit(10000)




class MetricsController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(MetricsController, self).__init__(req, link, data, **config)
        self.sdwan_controller = data['sdwan_controller']
        self.logger = self.sdwan_controller.logger  

    # Route to receive metrics from Mininet and make a decision
    @route('metrics', '/metrics', methods=['POST'])
    def receive_metrics(self, request):
        try:
            metrics = request.json_body
            chosen_tunnel = self.sdwan_controller.process_metrics(metrics) 
            self.logger.info("Chosen tunnel: %s", chosen_tunnel) 
            if chosen_tunnel:
                return Response(status=200, body=str(chosen_tunnel), content_type='text/plain')
            else:
                return Response(status=500, body="No decision could be made.", content_type='text/plain')

        except Exception as e:
            self.logger.error("Error processing metrics: %s", e)
            return Response(status=500, body=str(e))
    @route('decision', '/decision', methods=['POST'])
    def receive_decision(self, request):
        try:
            decision = request.json_body
            self.sdwan_controller.process_decision(decision) 


        except Exception as e:
            self.logger.error("Error processing decision: %s", e)

            
    # Route to allow Mininet to query the chosen link
    @route('decision', '/link_decision', methods=['GET'])
    def link_decision(self, request):
        try:
            chosen_tunnel = self.sdwan_controller.get_link_decision()
            self.logger.info("GET request: Chosen tunnel is %s", chosen_tunnel)
            return Response(status=200, body=chosen_tunnel, content_type='text/plain')
        except Exception as e:
            self.logger.error("Error fetching link decision: %s", e)
            return Response(status=500, body=str(e))

           # Route to receive metrics from Mininet and make a decision


class SDWANController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}
    
    def __init__(self, *args, **kwargs):
        super(SDWANController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.current_link = 'lte'  # Default link to use
        self.wsgi = kwargs['wsgi']
        self.wsgi.register(MetricsController, {'sdwan_controller': self})
        self.log_file = 'network_metrics.log'
        self.log_file1 = 'network_decision1.log'
        self.initialize_log()
        self.list_lost=[]
        self.rr_toggle = True  # Used for round-robin switching for port 9999

    def initialize_log(self):
        with open(self.log_file, 'w') as f:
            
            f.write('episode,tick,delay_lte,throughput_lte,packet_loss_lte,delay_satellite,throughput_satellite,packet_loss_satellite,cost_lte,cost_satellite,action,reward,total_reward,Loss,total_loss\n')           
        with open(self.log_file1, 'w') as f1:     
            f1.write('action,reward\n')
    def process_metrics(self, metrics):
        lte_delay = metrics.get('lte', {}).get('delay', float('inf'))
        lte_throughput = metrics.get('lte', {}).get('throughput', 0)
        lte_packet_loss = metrics.get('lte', {}).get('packet_loss', 100)
        
        satellite_delay = metrics.get('satellite', {}).get('delay', float('inf'))
        satellite_throughput = metrics.get('satellite', {}).get('throughput', 0)
        satellite_packet_loss = metrics.get('satellite', {}).get('packet_loss', 100)

        episode = metrics.get('episode', {}).get('episode number', float('inf'))
        reward = metrics.get('episode', {}).get('reward', float('inf'))
        total_reward = metrics.get('episode', {}).get('total reward', float('inf'))
        action = metrics.get('episode', {}).get('chosen link', )
        tick = metrics.get('episode', {}).get('tick', float('inf'))
        # Print the metrics
        print("LTE - Delay:", lte_delay, "Throughput:", lte_throughput, "Packet Loss:", lte_packet_loss)
        print("Satellite - Delay:", satellite_delay, "Throughput:", satellite_throughput, "Packet Loss:", satellite_packet_loss)
        print("episode number:", episode, "reward:", reward, "total reward:", total_reward, 'action', action)
        # Define costs
        cost_lte = self.get_link_cost('lte')
        cost_satellite = self.get_link_cost('satellite')
        

        if int(action) == 1:
            self.current_link = 'satellite'
        elif int(action) == 0:
            self.current_link = 'lte'
        Loss = (lte_packet_loss + satellite_packet_loss) / 2
        #list_lost=[]
        self.list_lost.append(Loss)
    
        if tick == 19:
            total_loss = sum(self.list_lost)/20
            self.list_lost.clear()
        else:
            total_loss=0

        # Log the metrics and decision for both LTE and satellite
        self.log_metrics( episode, tick, lte_delay, lte_throughput, lte_packet_loss,cost_lte, satellite_delay, satellite_throughput, satellite_packet_loss, cost_satellite, action,reward,total_reward,Loss,total_loss)

        return self.current_link






    def process_decision(self, decision):
        action = decision.get('action', 0)
        reward = decision.get('reward', 0)

       
           

        # Log the metrics and decision for both LTE and satellite
        self.log_decision( action, reward)

        return action, reward

    def calculate_reward(self, delay, throughput, packet_loss, cost):
        # Define a simple reward function
        reward = (throughput - (delay / 10) - (packet_loss / 10)) - cost
        return reward
       
       
       
       


 
    def log_metrics(self, episode, tick, lte_delay, lte_throughput, lte_packet_loss, cost_lte, satellite_delay, satellite_throughput, satellite_packet_loss, cost_satellite ,action,reward,total_reward,Loss,total_loss):



        log_entry = {
            'episode': episode,
            'tick': tick,
            'delay_lte': lte_delay,
            'throughput_lte': lte_throughput,
            'packet_loss_lte': lte_packet_loss,
            'delay_satellite': satellite_delay,
            'throughput_satellite': satellite_throughput,
            'packet_loss_satellite': satellite_packet_loss,
            'cost_lte': cost_lte,
            'cost_satellite': cost_satellite,
            'action': action,
            'reward': reward,
            'total_reward': total_reward,
            'Loss': Loss,
            'total loss': total_loss
        }
        print("log entry",log_entry)

        
        # Append log entry to the file
        with open(self.log_file, 'a') as f:
            f.write(','.join(str(log_entry[key]) for key in [
            'episode', 'tick', 'delay_lte', 'throughput_lte', 'packet_loss_lte',
            'delay_satellite', 'throughput_satellite', 'packet_loss_satellite',
            'cost_lte', 'cost_satellite', 'action', 'reward', 'total_reward', 'Loss', 'total loss']) + '\n')
       




    def log_decision(self,action,reward):
       


        log_entry1 = {
            'action': action,
            'reward': reward,
        }

        
        # Append log entry to the file
        with open(self.log_file1, 'a') as f1:
            f1.write(','.join(str(log_entry1[key]) for key in [
             'action', 'reward']) + '\n')
       
       

    def get_link_decision(self):
        return self.current_link


    def get_link_cost(self,link_type):
        """Return the cost associated with the link type."""
        costs = {
        'lte': 5,
        'satellite': 20
    }
        return costs.get(link_type, 0)  # Return 0 if the link_type is not found

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.logger.info("Switch connected: %s", datapath.id)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                       ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        actions = []
        pkt = packet.Packet(msg.data)
        udp_pkt = pkt.get_protocols(ethernet.ethernet)[0]
        # Parse transport layer for UDP port
        try:
            from ryu.lib.packet import ipv4, udp
            ip = pkt.get_protocol(ipv4.ipv4)
            udp_seg = pkt.get_protocol(udp.udp)
            dst_port = udp_seg.dst_port if udp_seg else None
        except Exception as e:
            dst_port = None

    # Handle background traffic on port 9999 with round-robin
        if dst_port == 9999:
            if self.rr_toggle:
                actions = [parser.OFPActionOutput(1)]  # LTE
            else:
                actions = [parser.OFPActionOutput(2)]  # Satellite
            self.rr_toggle = not self.rr_toggle  # Toggle for next packet
        else:
        # RL-controlled routing
            if self.current_link == 'lte':
                actions = [parser.OFPActionOutput(1)]
            else:
                actions = [parser.OFPActionOutput(2)]

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                              in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
