#!/usr/bin/env python3
"""
Intelligent SDN Controller with AI-based Traffic Classification
Uses Ryu framework
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, tcp, udp
from ryu.lib import hub
import time
import pickle
import numpy as np
from collections import defaultdict
import json
import os

class IntelligentController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(IntelligentController, self).__init__(*args, **kwargs)
        
        # MAC to port mapping
        self.mac_to_port = {}
        
        # Flow statistics storage
        self.flow_stats = defaultdict(dict)
        
        # Traffic classification model (will be loaded)
        self.classifier = None
        
        # Flow features for classification
        self.flow_features = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'start_time': time.time(),
            'last_seen': time.time(),
            'packets': [],
            'inter_arrival_times': [],
            'packet_sizes': []
        })
        
        # Traffic type to priority mapping
        self.traffic_priority = {
            'VoIP': 3,      # Highest priority
            'Gaming': 3,
            'Video': 2,
            'HTTP': 1,
            'FTP': 0        # Lowest priority
        }
        
        # Load ML model if available
        self._load_classifier()
        
        # Start monitoring thread
        self.monitor_thread = hub.spawn(self._monitor)
        
        self.logger.info("Intelligent SDN Controller Started")
    
    def _load_classifier(self):
        """Load pre-trained ML classifier"""
        model_paths = [
            'ml_models/traffic_classifier_real.pkl',
            'ml_models/traffic_classifier.pkl'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    if isinstance(model_data, dict):
                        self.classifier = model_data.get('model')
                        self.scaler = model_data.get('scaler')
                        self.label_encoder = model_data.get('label_encoder')
                    else:
                        self.classifier = model_data
                    
                    self.logger.info(f"✓ Loaded classifier from {model_path}")
                    return
                except Exception as e:
                    self.logger.error(f"Failed to load {model_path}: {e}")
        
        self.logger.warning("No ML model loaded, using rule-based classification")
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                         ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        self.logger.info("Switch connected: %s", datapath.id)
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None, 
                 idle_timeout=0, hard_timeout=0):
        """Add a flow entry to the switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                            actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                   priority=priority, match=match,
                                   instructions=inst, idle_timeout=idle_timeout,
                                   hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                   match=match, instructions=inst,
                                   idle_timeout=idle_timeout,
                                   hard_timeout=hard_timeout)
        
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle incoming packets"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        self.mac_to_port.setdefault(dpid, {})
        
        # Learn MAC address
        self.mac_to_port[dpid][src] = in_port
        
        # Extract flow features and classify
        flow_key = self._extract_flow_key(pkt)
        priority = 1  # Default priority
        
        if flow_key:
            self._update_flow_features(flow_key, pkt, time.time())
            
            # Classify traffic if enough packets collected
            if self.flow_features[flow_key]['packet_count'] >= 10:
                traffic_type = self._classify_traffic(flow_key)
                priority = self.traffic_priority.get(traffic_type, 1)
                self.logger.info(f"Flow {flow_key[0]}:{flow_key[2]} -> {flow_key[1]}:{flow_key[3]}: "
                               f"Classified as {traffic_type}, Priority: {priority}")
        
        # Determine output port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install flow rule if destination is known
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, priority, match, actions, 
                             msg.buffer_id, idle_timeout=30, hard_timeout=60)
                return
            else:
                self.add_flow(datapath, priority, match, actions,
                             idle_timeout=30, hard_timeout=60)
        
        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                 in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    def _extract_flow_key(self, pkt):
        """Extract 5-tuple flow key from packet"""
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if not ip_pkt:
            return None
        
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        
        if tcp_pkt:
            return (ip_pkt.src, ip_pkt.dst, tcp_pkt.src_port, 
                   tcp_pkt.dst_port, 'TCP')
        elif udp_pkt:
            return (ip_pkt.src, ip_pkt.dst, udp_pkt.src_port, 
                   udp_pkt.dst_port, 'UDP')
        
        return None
    
    def _update_flow_features(self, flow_key, pkt, timestamp):
        """Update flow features for classification"""
        features = self.flow_features[flow_key]
        
        # Update counters
        features['packet_count'] += 1
        features['byte_count'] += len(pkt)
        features['last_seen'] = timestamp
        
        # Calculate inter-arrival time
        if features['packets']:
            iat = timestamp - features['packets'][-1]
            features['inter_arrival_times'].append(iat)
        
        features['packets'].append(timestamp)
        features['packet_sizes'].append(len(pkt))
    
    def _classify_traffic(self, flow_key):
        """Classify traffic based on flow features"""
        features = self.flow_features[flow_key]
        src_ip, dst_ip, src_port, dst_port, protocol = flow_key
        
        # Rule-based classification (fallback)
        if dst_port == 80 or dst_port == 443 or src_port == 80 or src_port == 443:
            return 'HTTP'
        elif dst_port == 554 or (dst_port >= 5000 and dst_port <= 5100):
            return 'Video'
        elif dst_port == 5060 or (dst_port >= 16384 and dst_port <= 32767):
            return 'VoIP'
        elif dst_port == 21 or dst_port == 20:
            return 'FTP'
        elif dst_port >= 27000 and dst_port <= 28000:
            return 'Gaming'
        
        # Feature-based classification
        avg_packet_size = np.mean(features['packet_sizes']) if features['packet_sizes'] else 0
        avg_iat = np.mean(features['inter_arrival_times']) if features['inter_arrival_times'] else 0
        
        if avg_packet_size < 200 and avg_iat < 0.05:
            return 'Gaming' if protocol == 'UDP' else 'VoIP'
        elif avg_packet_size > 1000 and avg_iat < 0.1:
            return 'Video'
        elif avg_packet_size > 1200 and avg_iat < 0.01:
            return 'FTP'
        
        return 'HTTP'
    
    def _monitor(self):
        """Monitor network statistics"""
        while True:
            hub.sleep(10)
            
            if len(self.flow_features) == 0:
                continue
                
            self.logger.info("=== Flow Statistics ===")
            for flow_key, features in list(self.flow_features.items()):
                duration = features['last_seen'] - features['start_time']
                if duration > 0:
                    throughput = features['byte_count'] / duration
                    self.logger.info(
                        f"Flow: {flow_key[0]}:{flow_key[2]} -> {flow_key[1]}:{flow_key[3]} "
                        f"| Packets: {features['packet_count']} "
                        f"| Bytes: {features['byte_count']} "
                        f"| Throughput: {throughput:.2f} B/s"
                    )
            
            # Clean old flows (older than 60 seconds)
            current_time = time.time()
            expired_flows = [k for k, v in self.flow_features.items() 
                           if current_time - v['last_seen'] > 60]
            for flow in expired_flows:
                del self.flow_features[flow]