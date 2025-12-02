#!/usr/bin/env python3
"""
Traffic Generator for Different Application Types
Generates various traffic patterns for ML training
"""

import random
import time
import socket
import threading
from scapy.all import IP, TCP, UDP, Raw, send, sendp, Ether

class TrafficGenerator:
    """Generate different types of network traffic"""
    
    def __init__(self, src_ip, dst_ip):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.stop_flag = False
    
    def generate_http_traffic(self, duration=10):
        """Simulate HTTP traffic - short flows, port 80"""
        print(f"Generating HTTP traffic for {duration}s")
        start_time = time.time()
        
        while time.time() - start_time < duration and not self.stop_flag:
            # HTTP typically uses random high ports to port 80
            src_port = random.randint(1024, 65535)
            dst_port = 80
            
            # Small packets typical of HTTP requests/responses
            packet_size = random.randint(100, 1500)
            
            packet = IP(src=self.src_ip, dst=self.dst_ip) / \
                    TCP(sport=src_port, dport=dst_port, flags='S') / \
                    Raw(b'X' * packet_size)
            
            send(packet, verbose=0)
            time.sleep(random.uniform(0.1, 0.5))  # Inter-packet delay
    
    def generate_video_traffic(self, duration=10):
        """Simulate video streaming - constant bitrate, UDP"""
        print(f"Generating Video traffic for {duration}s")
        start_time = time.time()
        
        while time.time() - start_time < duration and not self.stop_flag:
            src_port = random.randint(1024, 65535)
            dst_port = 554  # RTSP port
            
            # Large packets for video streaming
            packet_size = random.randint(1000, 1500)
            
            packet = IP(src=self.src_ip, dst=self.dst_ip) / \
                    UDP(sport=src_port, dport=dst_port) / \
                    Raw(b'V' * packet_size)
            
            send(packet, verbose=0)
            time.sleep(0.033)  # ~30 fps
    
    def generate_voip_traffic(self, duration=10):
        """Simulate VoIP traffic - small packets, regular intervals"""
        print(f"Generating VoIP traffic for {duration}s")
        start_time = time.time()
        
        while time.time() - start_time < duration and not self.stop_flag:
            src_port = random.randint(1024, 65535)
            dst_port = 5060  # SIP port
            
            # Small packets for VoIP
            packet_size = random.randint(50, 200)
            
            packet = IP(src=self.src_ip, dst=self.dst_ip) / \
                    UDP(sport=src_port, dport=dst_port) / \
                    Raw(b'A' * packet_size)
            
            send(packet, verbose=0)
            time.sleep(0.02)  # 20ms intervals (50 packets/sec)
    
    def generate_file_transfer_traffic(self, duration=10):
        """Simulate bulk file transfer - large continuous flow"""
        print(f"Generating File Transfer traffic for {duration}s")
        start_time = time.time()
        
        src_port = random.randint(1024, 65535)
        dst_port = 21  # FTP port
        
        while time.time() - start_time < duration and not self.stop_flag:
            # Maximum size packets for bulk transfer
            packet_size = 1500
            
            packet = IP(src=self.src_ip, dst=self.dst_ip) / \
                    TCP(sport=src_port, dport=dst_port, flags='PA') / \
                    Raw(b'F' * packet_size)
            
            send(packet, verbose=0)
            time.sleep(0.001)  # Minimal delay for bulk transfer
    
    def generate_gaming_traffic(self, duration=10):
        """Simulate online gaming - small packets, low latency"""
        print(f"Generating Gaming traffic for {duration}s")
        start_time = time.time()
        
        while time.time() - start_time < duration and not self.stop_flag:
            src_port = random.randint(1024, 65535)
            dst_port = random.randint(27000, 28000)  # Gaming ports
            
            # Very small packets for gaming
            packet_size = random.randint(50, 150)
            
            packet = IP(src=self.src_ip, dst=self.dst_ip) / \
                    UDP(sport=src_port, dport=dst_port) / \
                    Raw(b'G' * packet_size)
            
            send(packet, verbose=0)
            time.sleep(random.uniform(0.01, 0.05))
    
    def generate_mixed_traffic(self, duration=30):
        """Generate mixed traffic of all types"""
        print(f"Generating mixed traffic for {duration}s")
        
        threads = [
            threading.Thread(target=self.generate_http_traffic, args=(duration,)),
            threading.Thread(target=self.generate_video_traffic, args=(duration,)),
            threading.Thread(target=self.generate_voip_traffic, args=(duration,)),
            threading.Thread(target=self.generate_file_transfer_traffic, args=(duration,)),
            threading.Thread(target=self.generate_gaming_traffic, args=(duration,))
        ]
        
        for thread in threads:
            thread.start()
            time.sleep(1)  # Stagger starts
        
        for thread in threads:
            thread.join()
    
    def stop(self):
        """Stop traffic generation"""
        self.stop_flag = True

def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python3 traffic_generator.py <src_ip> <dst_ip> <traffic_type>")
        print("Traffic types: http, video, voip, ftp, gaming, mixed")
        sys.exit(1)
    
    src_ip = sys.argv[1]
    dst_ip = sys.argv[2]
    traffic_type = sys.argv[3].lower()
    
    gen = TrafficGenerator(src_ip, dst_ip)
    
    duration = 20
    
    if traffic_type == 'http':
        gen.generate_http_traffic(duration)
    elif traffic_type == 'video':
        gen.generate_video_traffic(duration)
    elif traffic_type == 'voip':
        gen.generate_voip_traffic(duration)
    elif traffic_type == 'ftp':
        gen.generate_file_transfer_traffic(duration)
    elif traffic_type == 'gaming':
        gen.generate_gaming_traffic(duration)
    elif traffic_type == 'mixed':
        gen.generate_mixed_traffic(duration)
    else:
        print(f"Unknown traffic type: {traffic_type}")

if __name__ == '__main__':
    main()