#!/usr/bin/env python3
"""
Intelligent SDN Controller with AI-based Traffic Classification
Python 3.12 Compatible Version
Uses Ryu framework
"""

import json
import os
import pickle
import sys
import time
from collections import defaultdict

import networkx as nx
import numpy as np
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ether_types, ethernet, ipv4, packet, tcp, udp
from ryu.ofproto import ofproto_v1_3


class IntelligentController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(IntelligentController, self).__init__(*args, **kwargs)

        self.logger.info(f"Python version: {sys.version}")

        # MAC to port mapping
        self.mac_to_port = {}

        # Datapath references
        self.datapaths = {}

        # Flow statistics storage
        self.flow_stats = defaultdict(dict)

        # Traffic classification model
        self.classifier = None
        self.scaler = None
        self.label_encoder = None

        # Flow features for classification
        self.flow_features = defaultdict(
            lambda: {
                "packet_count": 0,
                "byte_count": 0,
                "start_time": time.time(),
                "last_seen": time.time(),
                "packets": [],
                "inter_arrival_times": [],
                "packet_sizes": [],
            }
        )

        # Traffic type to priority mapping
        self.traffic_priority = {
            "VoIP": 3,  # Highest priority
            "Gaming": 3,
            "Video": 2,
            "HTTP": 1,
            "FTP": 0,  # Lowest priority
        }

        # Network topology graph
        self.topology = self._build_topology()

        # RL Q-learning for routing
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

        # Switch port mappings (hardcoded for topology)
        self.switch_ports = {
            's1': {'s2': 1, 's3': 2, 's4': 3},
            's2': {'s1': 1, 's3': 2, 's4': 3},
            's3': {'s2': 1, 's4': 2, 's1': 3},
            's4': {'s2': 1, 's3': 2, 's1': 3}
        }

        # Load ML model if available
        self._load_classifier()

        # Start monitoring thread
        self.monitor_thread = hub.spawn(self._monitor)

        self.logger.info("✓ Intelligent SDN Controller Started (Python 3.12)")

    def _build_topology(self):
        """Build network topology graph with multiple paths"""
        G = nx.Graph()
        # Switches
        switches = ['s1', 's2', 's3', 's4']
        G.add_nodes_from(switches)
        # Links
        G.add_edge('s1', 's2', weight=1, bw=100)
        G.add_edge('s2', 's3', weight=1, bw=50)
        G.add_edge('s2', 's4', weight=1, bw=50)
        # Add more links for multiple paths
        G.add_edge('s1', 's3', weight=2, bw=50)  # Longer path
        G.add_edge('s1', 's4', weight=2, bw=50)
        G.add_edge('s3', 's4', weight=1, bw=50)
        # Host to switch
        host_switch = {'h1': 's1', 'h2': 's1', 'h3': 's3', 'h4': 's3', 'h5': 's4', 'h6': 's4'}
        for h, s in host_switch.items():
            G.add_edge(h, s, weight=1, bw=10)
        return G

    def _load_classifier(self):
        """Load pre-trained ML classifier"""
        model_paths = [
            "ml_models/traffic_classifier_real.pkl",
            "ml_models/traffic_classifier.pkl",
            "../ml_models/traffic_classifier_real.pkl",
            "../ml_models/traffic_classifier.pkl",
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    with open(model_path, "rb") as f:
                        model_data = pickle.load(f)

                    if isinstance(model_data, dict):
                        self.classifier = model_data.get("model")
                        self.scaler = model_data.get("scaler")
                        self.label_encoder = model_data.get("label_encoder")
                        self.logger.info(
                            f"✓ Loaded classifier: {model_data.get('model_name', 'Unknown')}"
                        )
                    else:
                        self.classifier = model_data
                        self.logger.info("✓ Loaded classifier (legacy format)")

                    self.logger.info(f"✓ Model loaded from: {model_path}")
                    return
                except Exception as e:
                    self.logger.warning(f"Could not load {model_path}: {e}")

        self.logger.warning("⚠ No ML model loaded, using rule-based classification")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)
        ]
        self.add_flow(datapath, 0, match, actions)

        self.logger.info(f"✓ Switch connected: {datapath.id}")

    def add_flow(
        self,
        datapath,
        priority,
        match,
        actions,
        buffer_id=None,
        idle_timeout=0,
        hard_timeout=0,
    ):
        """Add a flow entry to the switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                instructions=inst,
                idle_timeout=idle_timeout,
                hard_timeout=hard_timeout,
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=inst,
                idle_timeout=idle_timeout,
                hard_timeout=hard_timeout,
            )

        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle incoming packets"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

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
            if self.flow_features[flow_key]["packet_count"] >= 10:
                traffic_type = self._classify_traffic(flow_key)
                priority = self.traffic_priority.get(traffic_type, 1)

                # Log classification (only once per flow)
                if self.flow_features[flow_key]["packet_count"] == 10:
                    self.logger.info(
                        f"📊 Flow {flow_key[0]}:{flow_key[2]} -> "
                        f"{flow_key[1]}:{flow_key[3]} | "
                        f"Type: {traffic_type} | Priority: {priority}"
                    )

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
                self.add_flow(
                    datapath,
                    priority,
                    match,
                    actions,
                    msg.buffer_id,
                    idle_timeout=30,
                    hard_timeout=60,
                )
                return
            else:
                self.add_flow(
                    datapath, priority, match, actions, idle_timeout=30, hard_timeout=60
                )

        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(out)

    def _install_path(self, path, src_ip, dst_ip, priority, datapath, parser, in_port):
        """Install flow rules along the path"""
        # For simplicity, assume path is list of switches, and install on current datapath if in path
        # But since packet_in is per switch, we need to install on all switches in path
        # For now, just install on current switch assuming it's the entry point
        match = parser.OFPMatch(in_port=in_port, eth_type=0x0800, ipv4_src=src_ip, ipv4_dst=dst_ip)
        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]  # Use normal forwarding for now
        self.add_flow(datapath, priority, match, actions)

    def _get_path_rl(self, src_host, dst_host, traffic_type):
        """Get path using Q-learning"""
        src_switch = self._get_switch_for_host(src_host)
        dst_switch = self._get_switch_for_host(dst_host)
        if src_switch == dst_switch:
            return [src_switch]

        state = (src_switch, dst_switch, traffic_type)
        paths = list(nx.all_simple_paths(self.topology, src_switch, dst_switch, cutoff=4))
        if not paths:
            return nx.shortest_path(self.topology, src_switch, dst_switch)

        # Choose action (path) using epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(paths))
        else:
            action = max(range(len(paths)), key=lambda i: self.q_table[state][i])

        path = paths[action]
        return path

    def _get_switch_for_host(self, host):
        """Get switch for host"""
        host_switch = {'h1': 's1', 'h2': 's1', 'h3': 's3', 'h4': 's3', 'h5': 's4', 'h6': 's4'}
        return host_switch.get(host, 's1')

    def _update_q_table(self, state, action, reward, next_state):
        """Update Q-table"""
        best_next = max(self.q_table[next_state].values(), default=0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * best_next - self.q_table[state][action])

    def _extract_flow_key(self, pkt):
        """Extract 5-tuple flow key from packet"""
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if not ip_pkt:
            return None

        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)

        if tcp_pkt:
            return (ip_pkt.src, ip_pkt.dst, tcp_pkt.src_port, tcp_pkt.dst_port, "TCP")
        elif udp_pkt:
            return (ip_pkt.src, ip_pkt.dst, udp_pkt.src_port, udp_pkt.dst_port, "UDP")

        return None

    def _update_flow_features(self, flow_key, pkt, timestamp):
        """Update flow features for classification"""
        features = self.flow_features[flow_key]

        # Update counters
        features["packet_count"] += 1
        features["byte_count"] += len(pkt)
        features["last_seen"] = timestamp

        # Calculate inter-arrival time
        if features["packets"]:
            iat = timestamp - features["packets"][-1]
            features["inter_arrival_times"].append(iat)

        features["packets"].append(timestamp)
        features["packet_sizes"].append(len(pkt))

    def _classify_traffic(self, flow_key):
        """Classify traffic based on flow features"""
        features = self.flow_features[flow_key]
        src_ip, dst_ip, src_port, dst_port, protocol = flow_key

        # Rule-based classification (fallback)
        if dst_port == 80 or dst_port == 443 or src_port == 80 or src_port == 443:
            return "HTTP"
        elif dst_port == 554 or (dst_port >= 5000 and dst_port <= 5100):
            return "Video"
        elif dst_port == 5060 or (dst_port >= 16384 and dst_port <= 32767):
            return "VoIP"
        elif dst_port == 21 or dst_port == 20:
            return "FTP"
        elif dst_port >= 27000 and dst_port <= 28000:
            return "Gaming"

        # Feature-based classification
        if features["packet_sizes"]:
            avg_packet_size = np.mean(features["packet_sizes"])
        else:
            avg_packet_size = 0

        if features["inter_arrival_times"]:
            avg_iat = np.mean(features["inter_arrival_times"])
        else:
            avg_iat = 0

        if avg_packet_size < 200 and avg_iat < 0.05:
            return "Gaming" if protocol == "UDP" else "VoIP"
        elif avg_packet_size > 1000 and avg_iat < 0.1:
            return "Video"
        elif avg_packet_size > 1200 and avg_iat < 0.01:
            return "FTP"

        return "HTTP"

    def _monitor(self):
        """Monitor network statistics"""
        while True:
            hub.sleep(10)

            if len(self.flow_features) == 0:
                continue

            self.logger.info("=" * 60)
            self.logger.info("📈 Flow Statistics Summary")
            self.logger.info("=" * 60)

            total_flows = len(self.flow_features)
            total_packets = sum(f["packet_count"] for f in self.flow_features.values())
            total_bytes = sum(f["byte_count"] for f in self.flow_features.values())

            self.logger.info(f"Active Flows: {total_flows}")
            self.logger.info(f"Total Packets: {total_packets}")
            self.logger.info(f"Total Bytes: {total_bytes:,}")
            self.logger.info("-" * 60)

            for flow_key, features in list(self.flow_features.items())[:5]:
                duration = features["last_seen"] - features["start_time"]
                if duration > 0:
                    throughput = features["byte_count"] / duration
                    self.logger.info(
                        f"Flow: {flow_key[0]}:{flow_key[2]} → {flow_key[1]}:{flow_key[3]} | "
                        f"Pkts: {features['packet_count']} | "
                        f"Bytes: {features['byte_count']:,} | "
                        f"Throughput: {throughput:.2f} B/s"
                    )

            if total_flows > 5:
                self.logger.info(f"... and {total_flows - 5} more flows")

            self.logger.info("=" * 60)

            # Clean old flows (older than 60 seconds)
            current_time = time.time()
            expired_flows = [
                k
                for k, v in self.flow_features.items()
                if current_time - v["last_seen"] > 60
            ]
            for flow in expired_flows:
                del self.flow_features[flow]

            if expired_flows:
                self.logger.info(f"🧹 Cleaned {len(expired_flows)} expired flows")
