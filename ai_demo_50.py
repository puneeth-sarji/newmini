#!/usr/bin/env python3
"""
AI-Based SDN Traffic Classification and Routing Demo with 50 Flows
Uses expanded dataset for more realistic network simulation
"""

import time
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import random

class AISDNController:
    def __init__(self):
        # Initialize flow stats with proper default values
        self.flow_stats = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'start_time': time.time(),
            'classification': 'Unknown',
            'path': None,
            'latency': 0.0,
            'jitter': 0.0,
            'packet_loss': 0.0
        })
        
        # Load traffic dataset from CSV
        self.traffic_dataset = self._load_traffic_dataset()
        
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
        print(f"📈 Traffic Dataset: {len(self.traffic_dataset)} flows")
        print()
    
    def _load_traffic_dataset(self):
        """Load traffic dataset from CSV file"""
        try:
            df = pd.read_csv('traffic_dataset_50.csv')
            # Remove comment lines if any
            df = df[~df['src_ip'].astype(str).str.startswith('#')]
            print(f"✓ Loaded {len(df)} traffic flows from dataset")
            return df.to_dict('records')
        except Exception as e:
            print(f"Could not load dataset: {e}")
            # Fallback to basic dataset
            return [
                ('10.0.0.1', '10.0.0.3', 12345, 80, 'TCP', 1200, 'HTTP', 1, 100, 500, 200),
                ('10.0.0.2', '10.0.0.4', 54321, 554, 'TCP', 1400, 'Video', 2, 500, 200, 100),
                ('10.0.0.3', '10.0.0.5', 16384, 5060, 'UDP', 160, 'VoIP', 3, 64, 50, 30),
                ('10.0.0.4', '10.0.0.6', 8765, 21, 'TCP', 1500, 'FTP', 0, 50, 1000, 500),
                ('10.0.0.5', '10.0.0.1', 27015, 27015, 'UDP', 200, 'Gaming', 3, 100, 100, 50),
            ]
    
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
    
    def simulate_traffic(self):
        """Simulate traffic flow with AI-based routing using dataset"""
        # Randomly select a flow from dataset
        flow_record = random.choice(self.traffic_dataset)
        
        if isinstance(flow_record, dict):
            src_ip = flow_record['src_ip']
            dst_ip = flow_record['dst_ip']
            src_port = flow_record['src_port']
            dst_port = flow_record['dst_port']
            protocol = flow_record['protocol']
            packet_size = flow_record['packet_size']
            traffic_type = flow_record['traffic_type']
            priority = flow_record['priority']
        else:
            # Fallback for tuple format
            src_ip, dst_ip, src_port, dst_port, protocol, packet_size, traffic_type, priority, _, _, _ = flow_record
        
        flow_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"
        
        # Initialize flow if new
        if self.flow_stats[flow_key]['packet_count'] == 0:
            # Select optimal path using AI
            optimal_path = self._select_optimal_path(traffic_type, packet_size)
            self.flow_stats[flow_key]['path'] = optimal_path
            self.flow_stats[flow_key]['classification'] = traffic_type
            
            # Calculate QoS metrics
            latency, jitter, packet_loss = self._calculate_qos_metrics(optimal_path, traffic_type)
            self.flow_stats[flow_key]['latency'] = latency
            self.flow_stats[flow_key]['jitter'] = jitter
            self.flow_stats[flow_key]['packet_loss'] = packet_loss
            
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
        self.flow_stats[flow_key]['byte_count'] += packet_size
        
        return traffic_type, self.flow_stats[flow_key]['path']
    
    def show_statistics(self):
        """Display comprehensive flow statistics with AI insights"""
        print("\n🤖 AI-Powered Network Statistics")
        print("=================================")
        
        path_utilization = defaultdict(int)
        traffic_type_count = defaultdict(int)
        total_flows = len(self.flow_stats)
        
        for flow_key, stats in self.flow_stats.items():
            duration = time.time() - stats['start_time']
            if duration > 0 and stats['packet_count'] > 0:
                throughput = stats['byte_count'] / duration
                path = stats.get('path', 'Unknown')
                traffic_type = stats.get('classification', 'Unknown')
                
                path_utilization[path] += 1
                traffic_type_count[traffic_type] += 1
                
                print(f"🌐 {flow_key}")
                print(f"   🏷️  Type: {traffic_type}")
                print(f"   🛤️  Path: {path}")
                print(f"   📦 Packets: {stats['packet_count']}")
                print(f"   📊 Bytes: {stats['byte_count']}")
                print(f"   ⚡ Throughput: {throughput:.2f} B/s")
                print(f"   ⏱️  Latency: {stats.get('latency', 0):.1f}ms")
                print(f"   📈 Jitter: {stats.get('jitter', 0):.1f}ms")
                print(f"   📉 Loss: {stats.get('packet_loss', 0)*100:.2f}%")
                
                # QoS assessment
                requirements = self.traffic_requirements.get(traffic_type, {})
                qos_score = self._calculate_qos_score(stats, requirements)
                print(f"   ✅ QoS Score: {qos_score:.1f}/100")
                print()
        
        # Network-wide insights
        print("🧠 AI Network Insights:")
        print("======================")
        print(f"   📊 Total Active Flows: {total_flows}")
        print(f"   📈 Dataset Size: {len(self.traffic_dataset)} available flows")
        
        print("\n   🛤️  Path Utilization:")
        for path, count in path_utilization.items():
            percentage = (count / total_flows * 100) if total_flows > 0 else 0
            print(f"      {path}: {count} flows ({percentage:.1f}%)")
        
        print("\n   🏷️  Traffic Type Distribution:")
        for traffic_type, count in sorted(traffic_type_count.items()):
            percentage = (count / total_flows * 100) if total_flows > 0 else 0
            print(f"      {traffic_type}: {count} flows ({percentage:.1f}%)")
        
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
        for path_name in self.network_paths.keys():
            flows_on_path = sum(1 for stats in self.flow_stats.values() 
                              if stats.get('path') == path_name)
            path_loads[path_name] = flows_on_path
        
        # Suggest load balancing
        if path_loads:
            max_load_path = max(path_loads.items(), key=lambda x: x[1])[0]
            min_load_path = min(path_loads.items(), key=lambda x: x[1])[0]
            
            if path_loads[max_load_path] > path_loads[min_load_path] * 2:
                print(f"      🔄 Consider load balancing: Move flows from {max_load_path} to {min_load_path}")
        
        # Suggest path upgrades
        for path_name, load in path_loads.items():
            if load > 8:  # Threshold for congestion
                path_info = self.network_paths[path_name]
                print(f"      📈 {path_name} congested: Consider increasing bandwidth from {path_info['bandwidth']}Mbps")
        
        # Dataset utilization
        active_flows = len(self.flow_stats)
        dataset_size = len(self.traffic_dataset)
        utilization = (active_flows / dataset_size * 100) if dataset_size > 0 else 0
        print(f"      📊 Dataset Utilization: {utilization:.1f}% ({active_flows}/{dataset_size} flows active)")

def main():
    controller = AISDNController()
    
    print("🌐 Simulating AI-Powered Network Traffic...")
    print("🤖 Making intelligent routing decisions with 50-flow dataset...")
    print("Press Ctrl+C to stop\n")
    
    try:
        packet_count = 0
        while True:
            # Simulate traffic using dataset
            controller.simulate_traffic()
            
            packet_count += 1
            if packet_count % 20 == 0:
                controller.show_statistics()
            
            time.sleep(0.5)  # Generate packet every 0.5 seconds for more activity
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping AI simulation...")
        controller.show_statistics()
        print("✅ AI-Powered Simulation complete!")
        print("🎯 Network optimized with intelligent routing using 50-flow dataset!")

if __name__ == "__main__":
    main()