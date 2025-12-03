#!/usr/bin/env python3
"""
Network Topology Visualization for SDN AI Traffic Classifier
Shows the network paths, switches, and traffic flow distribution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import random

class NetworkTopologyVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.ax.set_xlim(0, 14)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Network paths configuration
        self.paths = {
            'path1': {
                'hops': 2, 
                'bandwidth': 1000, 
                'latency': 5, 
                'load': 0.3,
                'color': '#2ECC71',  # Green
                'y_position': 7
            },
            'path2': {
                'hops': 3, 
                'bandwidth': 500, 
                'latency': 10, 
                'load': 0.5,
                'color': '#3498DB',  # Blue
                'y_position': 5
            },
            'path3': {
                'hops': 4, 
                'bandwidth': 200, 
                'latency': 20, 
                'load': 0.7,
                'color': '#E74C3C',  # Red
                'y_position': 3
            }
        }
        
        # Traffic types with their characteristics
        self.traffic_types = {
            'VoIP': {'color': '#9B59B6', 'priority': 3, 'icon': '📞'},
            'Gaming': {'color': '#E67E22', 'priority': 3, 'icon': '🎮'},
            'Video': {'color': '#F39C12', 'priority': 2, 'icon': '📹'},
            'HTTP': {'color': '#1ABC9C', 'priority': 1, 'icon': '🌐'},
            'FTP': {'color': '#95A5A6', 'priority': 0, 'icon': '📁'}
        }
        
    def draw_controller(self):
        """Draw the SDN Controller"""
        controller_box = FancyBboxPatch(
            (6, 8.5), 2, 1,
            boxstyle="round,pad=0.1",
            facecolor='#34495E',
            edgecolor='#2C3E50',
            linewidth=2
        )
        self.ax.add_patch(controller_box)
        self.ax.text(7, 9, 'SDN\nController', 
                    ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
        
        # Add AI brain icon
        brain_circle = Circle((7, 8.2), 0.15, facecolor='#E74C3C', edgecolor='white')
        self.ax.add_patch(brain_circle)
        self.ax.text(7, 8.2, '🧠', ha='center', va='center', fontsize=12)
    
    def draw_switches(self):
        """Draw network switches for each path"""
        switch_positions = {
            'path1': [(3, 7), (7, 7), (11, 7)],
            'path2': [(2.5, 5), (5, 5), (9, 5), (11.5, 5)],
            'path3': [(2, 3), (4.5, 3), (7, 3), (9.5, 3), (12, 3)]
        }
        
        for path_name, positions in switch_positions.items():
            path_info = self.paths[path_name]
            for i, (x, y) in enumerate(positions):
                # Draw switch
                switch = Circle((x, y), 0.3, 
                              facecolor=path_info['color'], 
                              edgecolor='black', 
                              linewidth=1.5)
                self.ax.add_patch(switch)
                self.ax.text(x, y, f'S{i+1}', 
                           ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='white')
    
    def draw_paths(self):
        """Draw network paths with connections"""
        switch_positions = {
            'path1': [(3, 7), (7, 7), (11, 7)],
            'path2': [(2.5, 5), (5, 5), (9, 5), (11.5, 5)],
            'path3': [(2, 3), (4.5, 3), (7, 3), (9.5, 3), (12, 3)]
        }
        
        for path_name, positions in switch_positions.items():
            path_info = self.paths[path_name]
            
            # Draw connections between switches
            for i in range(len(positions) - 1):
                x1, y1 = positions[i]
                x2, y2 = positions[i + 1]
                
                # Draw connection line
                line = mlines.Line2D([x1, x2], [y1, y2], 
                                   color=path_info['color'], 
                                   linewidth=3, 
                                   alpha=0.7)
                self.ax.add_line(line)
                
                # Add bandwidth label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                self.ax.text(mid_x, mid_y + 0.3, 
                           f"{path_info['bandwidth']}Mbps", 
                           ha='center', va='center', 
                           fontsize=7, 
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor='white', 
                                   alpha=0.8))
    
    def draw_endpoints(self):
        """Draw source and destination endpoints"""
        # Source hosts
        for i in range(5):
            x = 0.5 + i * 0.3
            y = 9
            host = Circle((x, y), 0.15, facecolor='#95A5A6', edgecolor='black')
            self.ax.add_patch(host)
            self.ax.text(x, y, f'H{i+1}', ha='center', va='center', 
                        fontsize=6, fontweight='bold')
        
        # Destination hosts
        for i in range(5):
            x = 0.5 + i * 0.3
            y = 1
            host = Circle((x, y), 0.15, facecolor='#95A5A6', edgecolor='black')
            self.ax.add_patch(host)
            self.ax.text(x, y, f'H{i+6}', ha='center', va='center', 
                        fontsize=6, fontweight='bold')
        
        # Labels
        self.ax.text(1.5, 9.5, 'Source Hosts', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
        self.ax.text(1.5, 0.5, 'Destination Hosts', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
    
    def draw_traffic_flows(self):
        """Draw sample traffic flows on different paths"""
        # Sample flows for visualization
        sample_flows = [
            ('10.0.0.1', '10.0.0.3', 'HTTP', 'path1'),
            ('10.0.0.2', '10.0.0.4', 'Video', 'path2'),
            ('10.0.0.3', '10.0.0.5', 'VoIP', 'path1'),
            ('10.0.0.4', '10.0.0.6', 'FTP', 'path3'),
            ('10.0.0.5', '10.0.0.1', 'Gaming', 'path2')
        ]
        
        for src_ip, dst_ip, traffic_type, path_name in sample_flows:
            traffic_info = self.traffic_types[traffic_type]
            path_info = self.paths[path_name]
            
            # Draw flow arrow
            if path_name == 'path1':
                arrow_y = 7.5
            elif path_name == 'path2':
                arrow_y = 5.5
            else:
                arrow_y = 3.5
            
            arrow = FancyArrowPatch((1, arrow_y), (13, arrow_y),
                                  connectionstyle="arc3,rad=0.1",
                                  arrowstyle='->', 
                                  mutation_scale=20,
                                  linewidth=2,
                                  color=traffic_info['color'],
                                  alpha=0.6)
            self.ax.add_patch(arrow)
            
            # Add traffic type label
            self.ax.text(7, arrow_y + 0.3, 
                       f"{traffic_info['icon']} {traffic_type}", 
                       ha='center', va='center', 
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.2", 
                               facecolor=traffic_info['color'], 
                               alpha=0.3))
    
    def draw_legend(self):
        """Draw legend for paths and traffic types"""
        # Path legend
        legend_x = 0.5
        legend_y = 7.5
        
        self.ax.text(legend_x, legend_y + 0.5, 'Network Paths:', 
                    fontsize=10, fontweight='bold')
        
        for i, (path_name, path_info) in enumerate(self.paths.items()):
            y_pos = legend_y - i * 0.4
            
            # Path color box
            rect = patches.Rectangle((legend_x, y_pos - 0.1), 0.3, 0.2, 
                                   facecolor=path_info['color'], 
                                   edgecolor='black')
            self.ax.add_patch(rect)
            
            # Path info
            self.ax.text(legend_x + 0.4, y_pos, 
                       f"{path_name}: {path_info['hops']} hops, "
                       f"{path_info['latency']}ms, {path_info['load']*100:.0f}% load", 
                       fontsize=8, va='center')
        
        # Traffic types legend
        traffic_y = 5.5
        self.ax.text(legend_x, traffic_y + 0.5, 'Traffic Types:', 
                    fontsize=10, fontweight='bold')
        
        for i, (traffic_type, traffic_info) in enumerate(self.traffic_types.items()):
            y_pos = traffic_y - i * 0.3
            
            self.ax.text(legend_x, y_pos, 
                       f"{traffic_info['icon']} {traffic_type} "
                       f"(Priority: {traffic_info['priority']})", 
                       fontsize=8, va='center')
    
    def draw_metrics(self):
        """Draw network performance metrics"""
        metrics_x = 12
        metrics_y = 7.5
        
        self.ax.text(metrics_x, metrics_y + 0.5, 'Network Metrics:', 
                    fontsize=10, fontweight='bold')
        
        # Calculate total bandwidth and average latency
        total_bw = sum(path['bandwidth'] for path in self.paths.values())
        avg_latency = sum(path['latency'] for path in self.paths.values()) / len(self.paths)
        avg_load = sum(path['load'] for path in self.paths.values()) / len(self.paths)
        
        metrics = [
            f"Total BW: {total_bw} Mbps",
            f"Avg Latency: {avg_latency:.1f} ms",
            f"Avg Load: {avg_load*100:.0f}%",
            f"Active Paths: {len(self.paths)}",
            f"Max Hops: {max(path['hops'] for path in self.paths.values())}"
        ]
        
        for i, metric in enumerate(metrics):
            y_pos = metrics_y - i * 0.3
            self.ax.text(metrics_x, y_pos, metric, fontsize=8, va='center')
    
    def visualize(self):
        """Create the complete network topology visualization"""
        self.draw_controller()
        self.draw_switches()
        self.draw_paths()
        self.draw_endpoints()
        self.draw_traffic_flows()
        self.draw_legend()
        self.draw_metrics()
        
        # Add title
        self.ax.text(7, 9.8, 'SDN AI Traffic Classifier - Network Topology', 
                    ha='center', va='center', 
                    fontsize=14, fontweight='bold')
        
        # Add subtitle
        self.ax.text(7, 9.5, 'Multi-Path Intelligent Routing with Traffic Classification', 
                    ha='center', va='center', 
                    fontsize=10, style='italic')
        
        plt.tight_layout()
        return self.fig

def main():
    """Main function to generate and display network topology"""
    print("🌐 Generating Network Topology Visualization...")
    print("=" * 50)
    
    visualizer = NetworkTopologyVisualizer()
    fig = visualizer.visualize()
    
    # Save the visualization
    plt.savefig('network_topology.png', dpi=300, bbox_inches='tight')
    print("✅ Network topology saved as 'network_topology.png'")
    
    # Display the visualization
    plt.show()
    
    # Print topology summary
    print("\n📊 Network Topology Summary:")
    print("=" * 30)
    print("🔧 Controller: Centralized SDN Controller with AI")
    print("🛤️  Paths: 3 redundant paths with different characteristics")
    print("🔄 Switches: 12 switches distributed across paths")
    print("🌐 Endpoints: 10 source and destination hosts")
    print("📦 Traffic Types: 5 (HTTP, Video, VoIP, FTP, Gaming)")
    print("🧠 Features: AI-based routing, QoS optimization, load balancing")

if __name__ == "__main__":
    main()