#!/usr/bin/env python3
"""
Demo Traffic Classifier Controller (without Ryu dependencies)
Simulates the intelligent controller functionality
"""

import time
import pickle
import numpy as np
from collections import defaultdict
import json
import os

class DemoController:
    """Demo version of intelligent controller without Ryu dependencies"""
    
    def __init__(self):
        # MAC to port mapping
        self.mac_to_port = {}
        
        # Flow statistics storage
        self.flow_stats = defaultdict(dict)
        
        # Traffic classification model
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        
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
        
        print("🚀 Demo SDN Controller Started")
        print("=" * 50)
    
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
                    
                    print(f"✅ Loaded classifier from {model_path}")
                    return
                except Exception as e:
                    print(f"❌ Failed to load {model_path}: {e}")
        
        print("⚠️  No ML model loaded, using rule-based classification")
    
    def simulate_packet_in(self, src_ip, dst_ip, src_port, dst_port, protocol, packet_size):
        """Simulate packet-in event"""
        flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        timestamp = time.time()
        
        # Update flow features
        self._update_flow_features(flow_key, packet_size, timestamp)
        
        # Classify traffic if enough packets collected
        if self.flow_features[flow_key]['packet_count'] >= 10:
            traffic_type = self._classify_traffic(flow_key)
            priority = self.traffic_priority.get(traffic_type, 1)
            
            print(f"📊 Flow {src_ip}:{src_port} → {dst_ip}:{dst_port}")
            print(f"   Classified as: {traffic_type}")
            print(f"   Priority: {priority}")
            print(f"   Packets: {self.flow_features[flow_key]['packet_count']}")
            print(f"   Bytes: {self.flow_features[flow_key]['byte_count']}")
            print("-" * 40)
            
            return traffic_type, priority
        
        return None, None
    
    def _update_flow_features(self, flow_key, packet_size, timestamp):
        """Update flow features for classification"""
        features = self.flow_features[flow_key]
        
        # Update counters
        features['packet_count'] += 1
        features['byte_count'] += packet_size
        features['last_seen'] = timestamp
        
        # Calculate inter-arrival time
        if features['packets']:
            iat = timestamp - features['packets'][-1]
            features['inter_arrival_times'].append(iat)
        
        features['packets'].append(timestamp)
        features['packet_sizes'].append(packet_size)
    
    def _classify_traffic(self, flow_key):
        """Classify traffic based on flow features"""
        features = self.flow_features[flow_key]
        src_ip, dst_ip, src_port, dst_port, protocol = flow_key
        
        # Try ML classification first
        if self.classifier is not None:
            try:
                # Extract features for ML model
                feature_vector = self._extract_ml_features(features, src_port, dst_port, protocol)
                
                # Scale features
                feature_vector_scaled = self.scaler.transform([feature_vector])
                
                # Predict
                prediction = self.classifier.predict(feature_vector_scaled)[0]
                confidence = np.max(self.classifier.predict_proba(feature_vector_scaled)[0])
                
                print(f"🤖 ML Prediction: {prediction} (confidence: {confidence:.2f})")
                return prediction
            except Exception as e:
                print(f"⚠️  ML classification failed: {e}")
        
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
    
    def _extract_ml_features(self, features, src_port, dst_port, protocol):
        """Extract features for ML model"""
        duration = features['last_seen'] - features['start_time']
        avg_packet_size = np.mean(features['packet_sizes']) if features['packet_sizes'] else 0
        std_packet_size = np.std(features['packet_sizes']) if features['packet_sizes'] else 0
        avg_iat = np.mean(features['inter_arrival_times']) if features['inter_arrival_times'] else 0
        std_iat = np.std(features['inter_arrival_times']) if features['inter_arrival_times'] else 0
        
        return [
            duration,
            features['packet_count'],
            features['byte_count'],
            avg_packet_size,
            std_packet_size,
            avg_iat,
            std_iat,
            src_port,
            dst_port,
            1 if protocol == 'TCP' else 0,  # protocol_tcp
            1 if protocol == 'UDP' else 0,  # protocol_udp
            features['packet_count'] // 2,  # forward_packets (estimate)
            features['packet_count'] // 2,  # backward_packets (estimate)
            features['byte_count'] / duration if duration > 0 else 0  # flow_bytes_per_sec
        ]
    
    def simulate_traffic_scenarios(self):
        """Simulate different traffic scenarios"""
        print("\n🎯 Simulating Traffic Classification Scenarios")
        print("=" * 50)
        
        scenarios = [
            # HTTP Traffic
            {"name": "HTTP Web Browsing", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", 
             "src_port": 12345, "dst_port": 80, "protocol": "TCP", "packet_size": 1200},
            
            # Video Streaming
            {"name": "Video Streaming", "src_ip": "10.0.0.3", "dst_ip": "10.0.0.4", 
             "src_port": 54321, "dst_port": 554, "protocol": "TCP", "packet_size": 1400},
            
            # VoIP Call
            {"name": "VoIP Call", "src_ip": "10.0.0.5", "dst_ip": "10.0.0.6", 
             "src_port": 16384, "dst_port": 5060, "protocol": "UDP", "packet_size": 160},
            
            # FTP Transfer
            {"name": "FTP File Transfer", "src_ip": "10.0.0.7", "dst_ip": "10.0.0.8", 
             "src_port": 54321, "dst_port": 21, "protocol": "TCP", "packet_size": 1500},
            
            # Gaming Traffic
            {"name": "Online Gaming", "src_ip": "10.0.0.9", "dst_ip": "10.0.0.10", 
             "src_port": 27005, "dst_port": 27015, "protocol": "UDP", "packet_size": 100}
        ]
        
        for scenario in scenarios:
            print(f"\n📡 Scenario: {scenario['name']}")
            print("-" * 30)
            
            # Simulate multiple packets for this flow
            for i in range(15):
                self.simulate_packet_in(
                    scenario['src_ip'], scenario['dst_ip'],
                    scenario['src_port'], scenario['dst_port'],
                    scenario['protocol'], scenario['packet_size']
                )
                time.sleep(0.01)  # Small delay between packets
    
    def show_statistics(self):
        """Show flow statistics"""
        print("\n📈 Flow Statistics Summary")
        print("=" * 50)
        
        for flow_key, features in list(self.flow_features.items()):
            if features['packet_count'] >= 10:
                duration = features['last_seen'] - features['start_time']
                if duration > 0:
                    throughput = features['byte_count'] / duration
                    avg_pkt_size = np.mean(features['packet_sizes']) if features['packet_sizes'] else 0
                    
                    print(f"Flow: {flow_key[0]}:{flow_key[2]} → {flow_key[1]}:{flow_key[3]}")
                    print(f"  Packets: {features['packet_count']}")
                    print(f"  Bytes: {features['byte_count']}")
                    print(f"  Duration: {duration:.2f}s")
                    print(f"  Throughput: {throughput:.2f} B/s")
                    print(f"  Avg Packet Size: {avg_pkt_size:.0f} bytes")
                    print("-" * 30)

def main():
    """Main demo function"""
    print("🌐 SDN AI Traffic Classifier Demo")
    print("================================")
    
    # Initialize controller
    controller = DemoController()
    
    # Simulate traffic scenarios
    controller.simulate_traffic_scenarios()
    
    # Show statistics
    controller.show_statistics()
    
    print("\n✅ Demo Complete!")
    print("This demonstrates the core functionality of the intelligent SDN controller:")
    print("- Real-time traffic classification")
    print("- Priority-based QoS assignment")
    print("- Flow feature extraction")
    print("- ML-based decision making")

if __name__ == "__main__":
    main()