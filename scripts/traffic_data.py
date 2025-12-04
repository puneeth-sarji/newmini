# Traffic Dataset Configuration for SDN AI Traffic Classifier
# This file contains the traffic flow definitions used in the demo

# Traffic Flow Definitions
# Format: (src_ip, dst_ip, src_port, dst_port, protocol, packet_size)
# Each flow represents different traffic types with realistic characteristics

traffic_flows = [
    # HTTP Traffic - Web browsing, moderate bandwidth, standard latency requirements
    ('10.0.0.1', '10.0.0.3', 12345, 80, 'TCP', 1200),      # HTTP GET request
    ('10.0.0.1', '10.0.0.3', 12346, 443, 'TCP', 1400),     # HTTPS secure
    ('10.0.0.1', '10.0.0.3', 12347, 8080, 'TCP', 1100),   # HTTP alternative port
    
    # Video Streaming - High bandwidth, latency sensitive but tolerant
    ('10.0.0.2', '10.0.0.4', 54321, 554, 'TCP', 1400),     # RTSP streaming
    ('10.0.0.2', '10.0.0.4', 54322, 5004, 'UDP', 1300),    # RTP video
    ('10.0.0.2', '10.0.0.4', 54323, 1935, 'TCP', 1500),    # RTMP streaming
    
    # VoIP Traffic - Low bandwidth, very low latency and jitter requirements
    ('10.0.0.3', '10.0.0.5', 16384, 5060, 'UDP', 160),     # SIP signaling
    ('10.0.0.3', '10.0.0.5', 16385, 5061, 'UDP', 180),     # SIP over TLS
    ('10.0.0.3', '10.0.0.5', 20000, 16384, 'UDP', 200),    # RTP audio stream
    
    # FTP Traffic - Bulk data transfer, high bandwidth, tolerant of delays
    ('10.0.0.4', '10.0.0.6', 8765, 21, 'TCP', 1500),       # FTP control
    ('10.0.0.4', '10.0.0.6', 8766, 20, 'TCP', 1500),       # FTP data
    ('10.0.0.4', '10.0.0.6', 8767, 22, 'TCP', 1200),       # SFTP secure
    
    # Gaming Traffic - Low latency, low jitter, moderate bandwidth
    ('10.0.0.5', '10.0.0.1', 27015, 27015, 'UDP', 200),    # Steam gaming
    ('10.0.0.5', '10.0.0.1', 27016, 27016, 'UDP', 250),    # Game data
    ('10.0.0.5', '10.0.0.1', 27017, 27017, 'UDP', 180),    # Game voice chat
]

# Traffic Classification Rules
# Port-based classification for quick identification

port_classification = {
    # Web Traffic
    80: 'HTTP',
    443: 'HTTP',
    8080: 'HTTP',
    8443: 'HTTP',
    
    # Video Streaming
    554: 'Video',      # RTSP
    5004: 'Video',     # RTP
    1935: 'Video',     # RTMP
    5000: 'Video',     # RTSP range
    5010: 'Video',
    5020: 'Video',
    5030: 'Video',
    5040: 'Video',
    5050: 'Video',
    5060: 'Video',
    5070: 'Video',
    5080: 'Video',
    5090: 'Video',
    5100: 'Video',
    
    # VoIP Traffic
    5060: 'VoIP',      # SIP
    5061: 'VoIP',      # SIP over TLS
    16384: 'VoIP',     # RTP audio range start
    32767: 'VoIP',     # RTP audio range end
    
    # File Transfer
    20: 'FTP',         # FTP data
    21: 'FTP',         # FTP control
    22: 'FTP',         # SFTP/SSH
    
    # Gaming
    27015: 'Gaming',   # Steam default
    27016: 'Gaming',
    27017: 'Gaming',
    27018: 'Gaming',
    27019: 'Gaming',
    27020: 'Gaming',
    27021: 'Gaming',
    27022: 'Gaming',
    27023: 'Gaming',
    27024: 'Gaming',
    27025: 'Gaming',
    27026: 'Gaming',
    27027: 'Gaming',
    27028: 'Gaming',
    27029: 'Gaming',
    27030: 'Gaming',
    27031: 'Gaming',
    27032: 'Gaming',
    27033: 'Gaming',
    27034: 'Gaming',
    27035: 'Gaming',
    27036: 'Gaming',
    27037: 'Gaming',
    27038: 'Gaming',
    27039: 'Gaming',
    27040: 'Gaming',
    27041: 'Gaming',
    27042: 'Gaming',
    27043: 'Gaming',
    27044: 'Gaming',
    27045: 'Gaming',
    27046: 'Gaming',
    27047: 'Gaming',
    27048: 'Gaming',
    27049: 'Gaming',
    27050: 'Gaming',
}

# Traffic Requirements and QoS Parameters
# Defines the network requirements for each traffic type

traffic_requirements = {
    'VoIP': {
        'min_bandwidth': 64,      # 64 Kbps minimum
        'max_latency': 50,        # 50ms maximum latency
        'max_jitter': 30,         # 30ms maximum jitter
        'max_packet_loss': 0.01,  # 1% maximum packet loss
        'priority': 3,            # Highest priority
        'description': 'Real-time voice communication requiring very low latency and jitter'
    },
    
    'Gaming': {
        'min_bandwidth': 100,     # 100 Kbps minimum
        'max_latency': 100,       # 100ms maximum latency
        'max_jitter': 50,        # 50ms maximum jitter
        'max_packet_loss': 0.02,  # 2% maximum packet loss
        'priority': 3,            # Highest priority
        'description': 'Interactive gaming requiring low latency and responsive gameplay'
    },
    
    'Video': {
        'min_bandwidth': 500,     # 500 Kbps minimum for SD video
        'max_latency': 200,       # 200ms maximum latency
        'max_jitter': 100,       # 100ms maximum jitter
        'max_packet_loss': 0.03,  # 3% maximum packet loss
        'priority': 2,            # High priority
        'description': 'Video streaming requiring consistent bandwidth and moderate latency'
    },
    
    'HTTP': {
        'min_bandwidth': 100,     # 100 Kbps minimum
        'max_latency': 500,       # 500ms maximum latency
        'max_jitter': 200,       # 200ms maximum jitter
        'max_packet_loss': 0.05,  # 5% maximum packet loss
        'priority': 1,            # Normal priority
        'description': 'Web browsing with moderate bandwidth and latency requirements'
    },
    
    'FTP': {
        'min_bandwidth': 50,      # 50 Kbps minimum
        'max_latency': 1000,      # 1000ms maximum latency
        'max_jitter': 500,       # 500ms maximum jitter
        'max_packet_loss': 0.1,   # 10% maximum packet loss
        'priority': 0,            # Low priority
        'description': 'File transfer tolerant of delays and packet loss'
    }
}

# Network Path Characteristics
# Defines the properties of available network paths

network_paths = {
    'path1': {
        'hops': 2,
        'bandwidth': 1000,        # 1000 Mbps
        'latency': 5,             # 5ms base latency
        'load': 0.3,             # 30% current load
        'reliability': 0.99,      # 99% reliability
        'cost': 10,              # Relative cost unit
        'description': 'High-speed fiber optic link with low latency'
    },
    
    'path2': {
        'hops': 3,
        'bandwidth': 500,         # 500 Mbps
        'latency': 10,            # 10ms base latency
        'load': 0.5,             # 50% current load
        'reliability': 0.97,      # 97% reliability
        'cost': 7,               # Lower cost
        'description': 'Medium-speed connection with moderate latency'
    },
    
    'path3': {
        'hops': 4,
        'bandwidth': 200,         # 200 Mbps
        'latency': 20,            # 20ms base latency
        'load': 0.7,             # 70% current load
        'reliability': 0.95,      # 95% reliability
        'cost': 3,               # Lowest cost
        'description': 'Lower-speed backup connection with higher latency'
    }
}

# Packet Size Characteristics by Traffic Type
# Typical packet sizes for different traffic types

packet_sizes = {
    'VoIP': {
        'min': 60,
        'max': 200,
        'typical': 160,
        'description': 'Small packets for real-time voice'
    },
    
    'Gaming': {
        'min': 100,
        'max': 300,
        'typical': 200,
        'description': 'Small to medium packets for game data'
    },
    
    'Video': {
        'min': 800,
        'max': 1500,
        'typical': 1400,
        'description': 'Large packets for video streaming'
    },
    
    'HTTP': {
        'min': 500,
        'max': 1500,
        'typical': 1200,
        'description': 'Variable packet sizes for web traffic'
    },
    
    'FTP': {
        'min': 1000,
        'max': 1500,
        'typical': 1500,
        'description': 'Large packets for file transfer'
    }
}

# Usage Instructions:
# 1. Import this file in your main script: from traffic_data import *
# 2. Use traffic_flows list for simulation
# 3. Reference traffic_requirements for QoS parameters
# 4. Use network_paths for routing decisions
# 5. Apply port_classification for quick traffic identification