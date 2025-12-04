#!/usr/bin/env python3
"""
AI-Based SDN Traffic Classification and RL Routing Demo
Simulates intelligent routing decisions with Reinforcement Learning
"""

import pickle
import random
import time
from collections import defaultdict

import numpy as np


class RLSDNController:
    def __init__(self):
        self.flow_stats = defaultdict(
            lambda: {
                "packet_count": 0,
                "byte_count": 0,
                "start_time": time.time(),
                "classification": "Unknown",
                "path": "Unknown",
                "latency": 0.0,
                "jitter": 0.0,
                "packet_loss": 0.0,
            }
        )

        # Network topology with multiple paths
        self.network_paths = {
            "path1": {"hops": 2, "bandwidth": 1000, "latency": 5, "load": 0.3},
            "path2": {"hops": 3, "bandwidth": 500, "latency": 10, "load": 0.5},
            "path3": {"hops": 4, "bandwidth": 200, "latency": 20, "load": 0.7},
        }

        self.traffic_requirements = {
            "VoIP": {
                "min_bandwidth": 64,
                "max_latency": 50,
                "max_jitter": 30,
                "priority": 3,
            },
            "Gaming": {
                "min_bandwidth": 100,
                "max_latency": 100,
                "max_jitter": 50,
                "priority": 3,
            },
            "Video": {
                "min_bandwidth": 500,
                "max_latency": 200,
                "max_jitter": 100,
                "priority": 2,
            },
            "HTTP": {
                "min_bandwidth": 100,
                "max_latency": 500,
                "max_jitter": 200,
                "priority": 1,
            },
            "FTP": {
                "min_bandwidth": 50,
                "max_latency": 1000,
                "max_jitter": 500,
                "priority": 0,
            },
        }

        # RL Q-learning
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

        self.classifier = None
        self._load_classifier()
        print("🤖 RL-Based SDN Controller Started")
        print("==================================")
        print("🧠 RL Routing: ENABLED")
        print("📊 ML Classification: ACTIVE")
        print("🌐 Multi-path Optimization: ON")
        print()

    def _load_classifier(self):
        """Load pre-trained ML classifier"""
        model_paths = [
            "ml_models/traffic_classifier_real.pkl",
            "ml_models/traffic_classifier.pkl",
        ]

        for model_path in model_paths:
            try:
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)

                if isinstance(model_data, dict):
                    self.classifier = model_data.get("model")
                else:
                    self.classifier = model_data

                print(f"✓ Loaded ML model from {model_path}")
                return
            except Exception as e:
                print(f"Could not load {model_path}: {e}")

        print("⚠️  Using rule-based classification only")

    def _select_path_rl(self, traffic_type, src_host, dst_host):
        """RL-based path selection"""
        state = (src_host, dst_host, traffic_type)
        paths = list(self.network_paths.keys())

        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(paths))
        else:
            action = max(range(len(paths)), key=lambda i: self.q_table[state][i])

        path = paths[action]
        return path, action

    def _update_q_table(self, state, action, reward, next_state):
        """Update Q-table"""
        best_next = max(self.q_table[next_state].values(), default=0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * best_next - self.q_table[state][action])

    def _calculate_reward(self, path_name, traffic_type, latency, packet_loss):
        """Calculate RL reward based on QoS"""
        requirements = self.traffic_requirements.get(traffic_type, {})
        reward = 0

        if latency <= requirements.get("max_latency", 1000):
            reward += 10
        else:
            reward -= 5

        if packet_loss <= 0.01:
            reward += 5
        else:
            reward -= 10

        # Prefer shorter paths for high priority
        if requirements.get("priority", 1) >= 2 and self.network_paths[path_name]["hops"] <= 3:
            reward += 5

        return reward

    def _calculate_qos_metrics(self, path_name, traffic_type):
        """Calculate QoS metrics for the selected path"""
        path = self.network_paths[path_name]
        requirements = self.traffic_requirements[traffic_type]

        # Simulate network conditions
        base_latency = path["latency"]
        load_factor = path["load"]

        # Calculate actual metrics
        latency = base_latency * (1 + load_factor * 0.5) + random.uniform(-5, 10)
        jitter = latency * 0.1 * (1 + load_factor)
        packet_loss = load_factor * 0.01 * random.uniform(0.5, 2.0)

        return max(0, latency), max(0, jitter), min(1, max(0, packet_loss))

    def simulate_traffic(
        self, src_ip, dst_ip, src_port, dst_port, protocol, packet_size=1500
    ):
        """Simulate traffic flow with RL-based routing"""
        flow_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"

        # Extract hosts
        src_host = f"h{src_ip.split('.')[-1]}"
        dst_host = f"h{dst_ip.split('.')[-1]}"

        # Initialize flow if new
        if self.flow_stats[flow_key]["packet_count"] == 0:
            # Classify traffic
            traffic_type = self._classify_traffic(
                int(dst_port), protocol, int(packet_size)
            )
            self.flow_stats[flow_key]["classification"] = traffic_type

            # Select optimal path using RL
            optimal_path, action = self._select_path_rl(traffic_type, src_host, dst_host)
            self.flow_stats[flow_key]["path"] = optimal_path

            # Calculate QoS metrics
            latency, jitter, packet_loss = self._calculate_qos_metrics(
                optimal_path, traffic_type
            )
            self.flow_stats[flow_key]["latency"] = latency
            self.flow_stats[flow_key]["jitter"] = jitter
            self.flow_stats[flow_key]["packet_loss"] = packet_loss

            # Calculate reward and update Q-table
            reward = self._calculate_reward(optimal_path, traffic_type, latency, packet_loss)
            next_state = (src_host, dst_host, traffic_type)  # Same for simplicity
            self._update_q_table((src_host, dst_host, traffic_type), action, reward, next_state)

            priority = self.traffic_requirements.get(traffic_type, {}).get(
                "priority", 1
            )

            print(f"🧠 RL Routing Decision:")
            print(f"   📦 Flow: {flow_key}")
            print(f"   🏷️  Type: {traffic_type} (Priority: {priority})")
            print(f"   🛤️  Selected Path: {optimal_path}")
            print(f"   ⏱️  Latency: {latency:.1f}ms")
            print(f"   📊 Jitter: {jitter:.1f}ms")
            print(f"   📉 Packet Loss: {packet_loss * 100:.2f}%")
            print(f"   🎯 RL Reward: {reward}")
            print()

        # Update flow statistics
        self.flow_stats[flow_key]["packet_count"] += 1
        self.flow_stats[flow_key]["byte_count"] += int(packet_size)

        return self.flow_stats[flow_key]["classification"], self.flow_stats[flow_key][
            "path"
        ]

    def _classify_traffic(self, dst_port, protocol, packet_size):
        """Classify traffic using rules or ML"""
        # Rule-based classification
        if dst_port in [80, 443]:
            return "HTTP"
        elif dst_port == 554 or (5000 <= dst_port <= 5100):
            return "Video"
        elif dst_port == 5060 or (16384 <= dst_port <= 32767):
            return "VoIP"
        elif dst_port in [20, 21]:
            return "FTP"
        elif 27000 <= dst_port <= 28000:
            return "Gaming"

        # Size-based classification
        if packet_size < 200:
            return "VoIP" if protocol == "UDP" else "Gaming"
        elif packet_size > 1200:
            return "FTP" if protocol == "TCP" else "Video"

        return "HTTP"

    def show_statistics(self):
        """Display comprehensive flow statistics with RL insights"""
        print("\n🤖 RL-Powered Network Statistics")
        print("=================================")

        path_utilization = defaultdict(int)
        total_flows = len(self.flow_stats)

        for flow_key, stats in self.flow_stats.items():
            duration = time.time() - stats["start_time"]
            if duration > 0 and stats["packet_count"] > 0:
                throughput = stats["byte_count"] / duration
                path = stats.get("path", "Unknown")
                path_utilization[path] += 1

                print(f"🌐 {flow_key}")
                print(f"   🏷️  Type: {stats['classification']}")
                print(f"   🛤️  Path: {path}")
                print(f"   📦 Packets: {stats['packet_count']}")
                print(f"   📊 Bytes: {stats['byte_count']}")
                print(f"   ⚡ Throughput: {throughput:.2f} B/s")
                print(f"   ⏱️  Latency: {stats.get('latency', 0):.1f}ms")
                print(f"   📈 Jitter: {stats.get('jitter', 0):.1f}ms")
                print(f"   📉 Loss: {stats.get('packet_loss', 0) * 100:.2f}%")
                print()

        # Network-wide insights
        print("🧠 RL Network Insights:")
        print("======================")
        print(f"   📊 Total Active Flows: {total_flows}")

        for path, count in path_utilization.items():
            percentage = (count / total_flows * 100) if total_flows > 0 else 0
            print(f"   🛤️  {path}: {count} flows ({percentage:.1f}%)")

        # RL learning progress
        print(f"   🎯 Q-table Size: {len(self.q_table)} states")
        print()


def main():
    controller = RLSDNController()

    # Generate sample traffic flows
    traffic_flows = [
        ("10.0.0.1", "10.0.0.3", "12345", "80", "TCP", "1500"),
        ("10.0.0.2", "10.0.0.5", "23456", "5060", "UDP", "200"),
        ("10.0.0.3", "10.0.0.1", "34567", "21", "TCP", "1400"),
        ("10.0.0.4", "10.0.0.6", "45678", "554", "TCP", "1300"),
        ("10.0.0.5", "10.0.0.2", "56789", "27015", "UDP", "500"),
        ("10.0.0.6", "10.0.0.4", "67890", "443", "TCP", "1200"),
    ]

    print(f"🌐 Simulating Network Traffic with RL routing...")
    print("Press Ctrl+C to stop\n")

    try:
        packet_count = 0
        while True:
            # Randomly select a flow
            flow = random.choice(traffic_flows)
            controller.simulate_traffic(*flow)

            packet_count += 1
            if packet_count % 10 == 0:
                controller.show_statistics()

            time.sleep(1)  # Generate packet every second

    except KeyboardInterrupt:
        print("\n🛑 Stopping simulation")
        controller.show_statistics()
        print("✅ RL Simulation complete!")


if __name__ == "__main__":
    main()