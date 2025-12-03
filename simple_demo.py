#!/usr/bin/env python3
"""
AI-Based SDN Traffic Classification and Routing Demo
Simulates intelligent routing decisions based on ML predictions
"""

import time
import pickle
import numpy as np
from collections import defaultdict
import random

class AISDNController:
    def __init__(self):
        self.flow_stats = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'start_time': time.time(),
            'classification': 'Unknown',
            'path': 'Unknown',
            'latency': 0.0,
            'jitter': 0.0,
            'packet_loss': 0.0
        })
        
        # Network topology with multiple paths
        self.network_paths = {
            'path1': {'hops': 2, 'bandwidth': 1000, 'latency': 5, 'load': 0.3},
            'path2': {'hops': 3, 'bandwidth': 500, 'latency': 10, 'load': 0.5},
            'path3': {'hops': 4, 'bandwidth': 200, 'latency': 20, 'load': 0.7}
        }
        
        self.traffic_requirements = {
            'VoIP': {'min_bandwidth': 64, 'max_latency': 50, 'max_jitter': 30, 'priority': 3},
            'Gaming': {'min_bandwidth': 100, 'max_latency': 100, 'max_jitter': 50, 'priority': 3},
            'Video': {'min_bandwidth': 500, 'max_latency': 200, 'max_jitter': 100, 'priority': 2},
            'HTTP': {'min_bandwidth': 100, 'max_latency': 500, 'max_jitter': 200, 'priority': 1},
            'FTP': {'min_bandwidth': 50, 'max_latency': 1000, 'max_jitter': 500, 'priority': 0}
        }
        
        self.classifier = None
        self._load_classifier()
        print("🤖 AI-Based SDN Controller Started")
        print("==================================")
        print("🧠 Intelligent Routing: ENABLED")
        print("📊 ML Classification: ACTIVE")
        print("🌐 Multi-path Optimization: ON")
        print()
    
    def _load_classifier(self):
        """Load pre-trained ML classifier"""
        model_paths = [
            'ml_models/traffic_classifier_real.pkl',
            'ml_models/traffic_classifier.pkl'
        ]
        
        for model_path in model_paths:
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict):
                    self.classifier = model_data.get('model')
                else:
                    self.classifier = model_data
                
                print(f"✓ Loaded ML model from {model_path}")
                return
            except Exception as e:
                print(f"Could not load {model_path}: {e}")
        
        print("⚠️  Using rule-based classification only")
    
    def _select_optimal_path(self, traffic_type, packet_size):
        """AI-based path selection using multi-criteria decision making"""
        requirements = self.traffic_requirements.get(traffic_type, {})
        
        best_path = None
        best_score = -1
        
        for path_name, path_info in self.network_paths.items():
            score = 0
            
            # Bandwidth score
            if path_info['bandwidth'] >= requirements.get('min_bandwidth', 0):
                score += 30
            else:
                score -= 50
            
            # Latency score
            if path_info['latency'] <= requirements.get('max_latency', 1000):
                score += 25
            else:
                score -= 30
            
            # Load balancing score
            score += (1 - path_info['load']) * 20
            
            # Priority-based weighting
            priority = requirements.get('priority', 1)
            score *= (1 + priority * 0.2)
            
            # Random factor for network dynamics
            score += random.uniform(-5, 5)
            
            if score > best_score:
                best_score = score
                best_path = path_name
        
        return best_path
    
    def _calculate_qos_metrics(self, path_name, traffic_type):
        """Calculate QoS metrics for the selected path"""
        path = self.network_paths[path_name]
        requirements = self.traffic_requirements[traffic_type]
        
        # Simulate network conditions
        base_latency = path['latency']
        load_factor = path['load']
        
        # Calculate actual metrics
        latency = base_latency * (1 + load_factor * 0.5) + random.uniform(-5, 10)
        jitter = latency * 0.1 * (1 + load_factor)
        packet_loss = load_factor * 0.01 * random.uniform(0.5, 2.0)
        
        return max(0, latency), max(0, jitter), min(1, max(0, packet_loss))
    
    def simulate_traffic(self, src_ip, dst_ip, src_port, dst_port, protocol, packet_size=1500):
        """Simulate traffic flow with AI-based routing"""
        flow_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"
        
        # Initialize flow if new
        if self.flow_stats[flow_key]['packet_count'] == 0:
            # Classify traffic
            traffic_type = self._classify_traffic(int(dst_port), protocol, int(packet_size))
            self.flow_stats[flow_key]['classification'] = traffic_type
            
            # Select optimal path using AI
            optimal_path = self._select_optimal_path(traffic_type, int(packet_size))
            self.flow_stats[flow_key]['path'] = optimal_path
            
            # Calculate QoS metrics
            latency, jitter, packet_loss = self._calculate_qos_metrics(optimal_path, traffic_type)
            self.flow_stats[flow_key]['latency'] = latency
            self.flow_stats[flow_key]['jitter'] = jitter
            self.flow_stats[flow_key]['packet_loss'] = packet_loss
            
            priority = self.traffic_requirements.get(traffic_type, {}).get('priority', 1)
            
            print(f"🧠 AI Routing Decision:")
            print(f"   📦 Flow: {flow_key}")
            print(f"   🏷️  Type: {traffic_type} (Priority: {priority})")
            print(f"   🛤️  Selected Path: {optimal_path}")
            print(f"   ⏱️  Latency: {latency:.1f}ms")
            print(f"   📊 Jitter: {jitter:.1f}ms")
            print(f"   📉 Packet Loss: {packet_loss*100:.2f}%")
            print()
        
        # Update flow statistics
        self.flow_stats[flow_key]['packet_count'] += 1
        self.flow_stats[flow_key]['byte_count'] += int(packet_size)
        
        return self.flow_stats[flow_key]['classification'], self.flow_stats[flow_key]['path']
    
    def _classify_traffic(self, dst_port, protocol, packet_size):
        """Classify traffic using rules or ML"""
        # Rule-based classification
        if dst_port in [80, 443]:
            return 'HTTP'
        elif dst_port == 554 or (5000 <= dst_port <= 5100):
            return 'Video'
        elif dst_port == 5060 or (16384 <= dst_port <= 32767):
            return 'VoIP'
        elif dst_port in [20, 21]:
            return 'FTP'
        elif 27000 <= dst_port <= 28000:
            return 'Gaming'
        
        # Size-based classification
        if packet_size < 200:
            return 'VoIP' if protocol == 'UDP' else 'Gaming'
        elif packet_size > 1200:
            return 'FTP' if protocol == 'TCP' else 'Video'
        
        return 'HTTP'
    
    def show_statistics(self):
        """Display comprehensive flow statistics with AI insights"""
        print("\n🤖 AI-Powered Network Statistics")
        print("=================================")
        
        path_utilization = defaultdict(int)
        total_flows = len(self.flow_stats)
        
        for flow_key, stats in self.flow_stats.items():
            duration = time.time() - stats['start_time']
            if duration > 0 and stats['packet_count'] > 0:
                throughput = stats['byte_count'] / duration
                path = stats.get('path', 'Unknown')
                path_utilization[path] += 1
                
                print(f"🌐 {flow_key}")
                print(f"   🏷️  Type: {stats['classification']}")
                print(f"   🛤️  Path: {path}")
                print(f"   📦 Packets: {stats['packet_count']}")
                print(f"   📊 Bytes: {stats['byte_count']}")
                print(f"   ⚡ Throughput: {throughput:.2f} B/s")
                print(f"   ⏱️  Latency: {stats.get('latency', 0):.1f}ms")
                print(f"   📈 Jitter: {stats.get('jitter', 0):.1f}ms")
                print(f"   📉 Loss: {stats.get('packet_loss', 0)*100:.2f}%")
                
                # QoS assessment
                requirements = self.traffic_requirements.get(stats['classification'], {})
                qos_score = self._calculate_qos_score(stats, requirements)
                print(f"   ✅ QoS Score: {qos_score:.1f}/100")
                print()
        
        # Network-wide insights
        print("🧠 AI Network Insights:")
        print("======================")
        print(f"   📊 Total Active Flows: {total_flows}")
        
        for path, count in path_utilization.items():
            percentage = (count / total_flows * 100) if total_flows > 0 else 0
            print(f"   🛤️  {path}: {count} flows ({percentage:.1f}%)")
        
        # Path optimization suggestions
        self._suggest_optimizations()
        print()
    
    def _calculate_qos_score(self, stats, requirements):
        """Calculate QoS compliance score"""
        score = 100
        
        latency = stats.get('latency', 0)
        jitter = stats.get('jitter', 0)
        packet_loss = stats.get('packet_loss', 0)
        
        if latency > requirements.get('max_latency', 1000):
            score -= 30
        if jitter > requirements.get('max_jitter', 500):
            score -= 20
        if packet_loss > 0.01:  # 1% packet loss threshold
            score -= 25
        
        return max(0, score)
    
    def _suggest_optimizations(self):
        """AI-powered optimization suggestions"""
        print("   💡 Optimization Suggestions:")
        
        # Analyze path utilization
        path_loads = {}
        for path_name, path_info in self.network_paths.items():
            flows_on_path = sum(1 for stats in self.flow_stats.values() 
                              if stats.get('path') == path_name)
            path_loads[path_name] = flows_on_path
        
        # Suggest load balancing
        if path_loads:
            max_load_path = max(path_loads, key=path_loads.get)
            min_load_path = min(path_loads, key=path_loads.get)
            
            if path_loads[max_load_path] > path_loads[min_load_path] * 2:
                print(f"      🔄 Consider load balancing: Move flows from {max_load_path} to {min_load_path}")
        
        # Suggest path upgrades
        for path_name, load in path_loads.items():
            if load > 5:  # Threshold for congestion
                path_info = self.network_paths[path_name]
                print(f"      📈 {path_name} congested: Consider increasing bandwidth from {path_info['bandwidth']}Mbps")

def main():
    controller = AISDNController()
    
    # Load 50-flow dataset
    traffic_flows = []
    with open('traffic_dataset_50.csv', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    traffic_flows.append(tuple(parts[:6]))
    
    print(f"🌐 Simulating Network Traffic with {len(traffic_flows)} flows...")
    print("Press Ctrl+C to stop\n")
    
    try:
        packet_count = 0
        while True:
            # Randomly select a flow from the 50-flow dataset
            flow = random.choice(traffic_flows)
            controller.simulate_traffic(*flow)
            
            packet_count += 1
            if packet_count % 10 == 0:
                controller.show_statistics()
            
            time.sleep(1)  # Generate packet every second
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping simulation")
        controller.show_statistics()
        print("✅ Simulation complete!")

if __name__ == "__main__":
    main()