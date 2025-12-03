#!/usr/bin/env python3
"""
Full SDN AI Traffic Classifier Simulation
Simulates the complete system behavior including controller, topology, and traffic
"""

import sys
import os
import time
import threading
import random
import numpy as np
from collections import defaultdict
import json

# Add project root to path
sys.path.append('/home/puneeth8055/Desktop/sdn-ai-traffic-classifier')

class MockSwitch:
    """Mock OpenFlow switch for simulation"""
    def __init__(self, switch_id, name):
        self.switch_id = switch_id
        self.name = name
        self.flows = {}
        self.ports = {}
        self.connected = False
        
    def connect(self, controller):
        """Connect to controller"""
        self.controller = controller
        self.connected = True
        print(f"🔗 Switch {self.name} connected to controller")
        
    def add_flow(self, flow_id, match, actions, priority=1):
        """Add flow entry"""
        self.flows[flow_id] = {
            'match': match,
            'actions': actions,
            'priority': priority,
            'packets': 0,
            'bytes': 0
        }
        print(f"📝 Flow added to {self.name}: Priority {priority}, Match: {match}")
        
    def process_packet(self, src_ip, dst_ip, src_port, dst_port, protocol):
        """Process incoming packet"""
        if not self.connected:
            return
            
        # Find matching flow
        matching_flow = None
        for flow_id, flow in self.flows.items():
            if (flow['match'].get('nw_dst') == dst_ip and 
                flow['match'].get('tp_dst') == dst_port):
                matching_flow = flow
                break
                
        if matching_flow:
            matching_flow['packets'] += 1
            matching_flow['bytes'] += 64  # Assume 64-byte packet
            return matching_flow['actions']
        else:
            # Send to controller for classification
            return self.controller.classify_flow(src_ip, dst_ip, src_port, dst_port, protocol)

class MockHost:
    """Mock host for simulation"""
    def __init__(self, host_id, name, ip, mac):
        self.host_id = host_id
        self.name = name
        self.ip = ip
        self.mac = mac
        self.switch = None
        self.port = None
        
    def connect_to_switch(self, switch, port):
        """Connect host to switch"""
        self.switch = switch
        self.port = port
        switch.ports[port] = self
        print(f"🖥️  Host {self.name} ({self.ip}) connected to {switch.name} on port {port}")
        
    def send_traffic(self, dst_ip, traffic_type, duration=5):
        """Generate traffic to destination"""
        print(f"📤 {self.name} generating {traffic_type} traffic to {dst_ip}")
        
        # Traffic patterns for different types
        traffic_patterns = {
            'http': {'dst_port': 80, 'protocol': 'tcp', 'packet_rate': 50},
            'video': {'dst_port': 554, 'protocol': 'tcp', 'packet_rate': 200},
            'voip': {'dst_port': 5060, 'protocol': 'udp', 'packet_rate': 100},
            'gaming': {'dst_port': 27015, 'protocol': 'udp', 'packet_rate': 150},
            'ftp': {'dst_port': 21, 'protocol': 'tcp', 'packet_rate': 30}
        }
        
        pattern = traffic_patterns.get(traffic_type, traffic_patterns['http'])
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Send packet through switch
            src_port = random.randint(1024, 65535)
            actions = self.switch.process_packet(
                self.ip, dst_ip, src_port, pattern['dst_port'], pattern['protocol']
            )
            
            if actions:
                print(f"  📦 Packet routed: {actions}")
            
            time.sleep(1.0 / pattern['packet_rate'])

class SDNController:
    """Mock SDN Controller with AI classification"""
    def __init__(self):
        self.switches = {}
        self.flow_classifier = None
        self.flow_stats = defaultdict(dict)
        self.traffic_priority = {
            'VoIP': 3,      # Highest priority
            'Gaming': 3,
            'Video': 2,
            'HTTP': 1,
            'FTP': 1        # Lowest priority
        }
        self.load_model()
        
    def load_model(self):
        """Load the trained ML model"""
        try:
            import pickle
            model_path = '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/traffic_classifier.pkl'
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.flow_classifier = {
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'label_encoder': model_data['label_encoder']
            }
            print("✅ ML model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            
    def register_switch(self, switch):
        """Register a new switch"""
        self.switches[switch.switch_id] = switch
        print(f"📋 Switch {switch.name} registered with controller")
        
    def classify_flow(self, src_ip, dst_ip, src_port, dst_port, protocol):
        """Classify traffic and install flow rules"""
        print(f"🧠 Classifying flow: {src_ip}:{src_port} → {dst_ip}:{dst_port} ({protocol})")
        
        if not self.flow_classifier:
            # Default classification if no model
            traffic_type = 'HTTP'
            priority = 1
        else:
            # Extract features for ML classification
            features = self.extract_flow_features(src_ip, dst_ip, src_port, dst_port, protocol)
            
            # Classify
            features_scaled = self.flow_classifier['scaler'].transform([features])
            predicted_index = self.flow_classifier['model'].predict(features_scaled)[0]
            traffic_type = self.flow_classifier['label_encoder'].inverse_transform([predicted_index])[0]
            priority = self.traffic_priority.get(traffic_type, 1)
            
        print(f"🎯 Flow classified as {traffic_type} with priority {priority}")
        
        # Install flow rule on all switches
        flow_id = f"{src_ip}-{dst_ip}-{dst_port}"
        for switch in self.switches.values():
            switch.add_flow(
                flow_id,
                match={'nw_dst': dst_ip, 'tp_dst': dst_port},
                actions=['output:2'],  # Forward to appropriate port
                priority=priority
            )
            
        return ['output:2']  # Forward action
        
    def extract_flow_features(self, src_ip, dst_ip, src_port, dst_port, protocol):
        """Extract features for ML classification"""
        # Simplified feature extraction
        features = [
            random.uniform(1, 10),      # duration
            random.randint(10, 1000),   # packet_count
            random.randint(1000, 100000), # byte_count
            random.uniform(50, 1500),   # avg_packet_size
            random.uniform(0.1, 2.0),   # std_packet_size
            random.uniform(0.01, 0.1),  # avg_inter_arrival_time
            random.uniform(0.001, 0.01), # std_inter_arrival_time
            src_port,                    # src_port
            dst_port,                    # dst_port
            1 if protocol == 'tcp' else 0,  # protocol_tcp
            1 if protocol == 'udp' else 0,  # protocol_udp
            random.randint(100, 1000),  # forward_packets
            random.randint(100, 1000),  # backward_packets
            random.uniform(1000, 100000) # flow_bytes_per_sec
        ]
        return features
        
    def show_statistics(self):
        """Display flow statistics"""
        print("\n📊 Flow Statistics:")
        print("-" * 50)
        
        for switch_id, switch in self.switches.items():
            print(f"\nSwitch {switch.name}:")
            if switch.flows:
                for flow_id, flow in switch.flows.items():
                    print(f"  Flow {flow_id[:20]}...: {flow['packets']} packets, {flow['bytes']} bytes")
            else:
                print("  No flows installed")

class NetworkTopology:
    """Mock network topology"""
    def __init__(self):
        self.controller = SDNController()
        self.switches = {}
        self.hosts = {}
        self.create_topology()
        
    def create_topology(self):
        """Create the network topology"""
        print("🏗️  Creating network topology...")
        
        # Create switches
        switch1 = MockSwitch(1, "s1")
        switch2 = MockSwitch(2, "s2")
        
        # Connect switches to controller
        switch1.connect(self.controller)
        switch2.connect(self.controller)
        self.controller.register_switch(switch1)
        self.controller.register_switch(switch2)
        
        # Create hosts
        hosts_config = [
            (1, "h1", "10.0.0.1", "00:00:00:00:00:01"),
            (2, "h2", "10.0.0.2", "00:00:00:00:00:02"),
            (3, "h3", "10.0.0.3", "00:00:00:00:00:03"),
            (4, "h4", "10.0.0.4", "00:00:00:00:00:04"),
            (5, "h5", "10.0.0.5", "00:00:00:00:00:05"),
            (6, "h6", "10.0.0.6", "00:00:00:00:00:06")
        ]
        
        for host_id, name, ip, mac in hosts_config:
            host = MockHost(host_id, name, ip, mac)
            self.hosts[name] = host
            
            # Connect hosts to switches (h1-h3 to s1, h4-h6 to s2)
            if host_id <= 3:
                host.connect_to_switch(switch1, host_id)
            else:
                host.connect_to_switch(switch2, host_id - 3)
                
        # Connect switches together
        switch1.ports[4] = switch2
        switch2.ports[4] = switch1
        
        print("✅ Topology created successfully")
        
    def run_traffic_simulation(self):
        """Run traffic simulation"""
        print("\n🚀 Starting traffic simulation...")
        
        # Define traffic scenarios
        scenarios = [
            ("h1", "10.0.0.3", "http", 3),
            ("h2", "10.0.0.4", "video", 4),
            ("h3", "10.0.0.5", "voip", 2),
            ("h4", "10.0.0.6", "gaming", 3),
            ("h5", "10.0.0.1", "ftp", 2),
            ("h6", "10.0.0.2", "http", 2)
        ]
        
        # Run scenarios in parallel
        threads = []
        for src_host, dst_ip, traffic_type, duration in scenarios:
            host = self.hosts[src_host]
            thread = threading.Thread(
                target=host.send_traffic,
                args=(dst_ip, traffic_type, duration)
            )
            threads.append(thread)
            thread.start()
            time.sleep(0.5)  # Stagger start times
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        print("\n✅ Traffic simulation completed")

def main():
    """Main simulation function"""
    print("=" * 60)
    print("🎯 SDN AI Traffic Classifier - Full System Simulation")
    print("=" * 60)
    
    # Create and run the simulation
    topology = NetworkTopology()
    
    print("\n" + "=" * 60)
    print("🌐 Network Status:")
    print("=" * 60)
    print(f"  Switches: {len(topology.switches)}")
    print(f"  Hosts: {len(topology.hosts)}")
    print(f"  Controller: {'✅ Active' if topology.controller else '❌ Inactive'}")
    
    # Run traffic simulation
    topology.run_traffic_simulation()
    
    # Show final statistics
    topology.controller.show_statistics()
    
    print("\n" + "=" * 60)
    print("🎉 Full System Simulation Complete!")
    print("=" * 60)
    print("\n📝 Summary:")
    print("  • SDN Controller with AI-based traffic classification")
    print("  • OpenFlow switches with dynamic flow installation")
    print("  • Multiple traffic types with priority-based routing")
    print("  • Real-time flow statistics and monitoring")
    print("\n🔧 For real deployment:")
    print("  1. Install Mininet: sudo apt install mininet")
    print("  2. Install Ryu: pip install ryu")
    print("  3. Run: ./start.sh (Terminal 1)")
    print("  4. Run: sudo python3 topology/simple_topology.py (Terminal 2)")

if __name__ == "__main__":
    main()