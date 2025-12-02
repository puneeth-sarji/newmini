# System Architecture Diagram

## Complete System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │   CLI    │  │  Logs    │  │  Graphs  │  │  Configuration   │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────┐
│                    APPLICATION LAYER                                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Machine Learning Module                         │  │
│  │                                                              │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │  │
│  │  │  Feature   │→ │   Model    │→ │  Classification    │   │  │
│  │  │ Extraction │  │ (RF/GB/NN) │  │   Output           │   │  │
│  │  └────────────┘  └────────────┘  └────────────────────┘   │  │
│  │                                                              │  │
│  │  Traffic Types:                                             │  │
│  │  • HTTP (Web)     • Video (Streaming)  • VoIP (Voice)      │  │
│  │  • Gaming         • FTP (File Transfer)                     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │           Routing & QoS Management                          │  │
│  │                                                              │  │
│  │  Priority Mapping:           Path Computation:              │  │
│  │  VoIP    → Priority 3       Dijkstra's Algorithm            │  │
│  │  Gaming  → Priority 3       K-Shortest Paths                │  │
│  │  Video   → Priority 2       Load Balancing                  │  │
│  │  HTTP    → Priority 1       Congestion Avoidance            │  │
│  │  FTP     → Priority 0                                       │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 │ API / Function Calls
                                 │
┌────────────────────────────────▼───────────────────────────────────┐
│                      CONTROL LAYER (Ryu SDN)                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Flow Management                                 │  │
│  │  • Packet-In Handler      • Flow Rule Installation          │  │
│  │  • Match Criteria         • Action Definition               │  │
│  │  • Timeout Management     • Buffer Management               │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │            Statistics Collection                             │  │
│  │  • Flow Statistics        • Port Statistics                  │  │
│  │  • Bandwidth Usage        • Packet Counts                    │  │
│  │  • Error Rates            • Duration Tracking                │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Topology Discovery                              │  │
│  │  • LLDP Protocol          • Link Detection                   │  │
│  │  • Switch Discovery       • Host Detection                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                MAC Learning Table                            │  │
│  │  Switch 1: {MAC1→Port1, MAC2→Port2, ...}                   │  │
│  │  Switch 2: {MAC3→Port1, MAC4→Port3, ...}                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 │ OpenFlow Protocol (Port 6653)
                                 │
┌────────────────────────────────▼───────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                               │
│                                                                     │
│  Network Topology:                                                  │
│                                                                     │
│         ┌───────┐      ┌───────┐                                  │
│         │  H1   │      │  H2   │                                  │
│         └───┬───┘      └───┬───┘                                  │
│             │              │                                        │
│         ┌───▼──────────────▼───┐                                  │
│         │      Switch 1         │                                  │
│         │   (OpenFlow 1.3)      │                                  │
│         └───────────┬───────────┘                                  │
│                     │                                               │
│         ┌───────────▼───────────┐                                  │
│         │      Switch 2         │                                  │
│         │   (OpenFlow 1.3)      │                                  │
│         └─────┬──────────┬──────┘                                  │
│               │          │                                          │
│     ┌─────────▼──┐   ┌──▼─────────┐                               │
│     │ Switch 3   │   │ Switch 4   │                               │
│     │(OF 1.3)    │   │(OF 1.3)    │                               │
│     └──┬────┬────┘   └────┬────┬──┘                               │
│        │    │             │    │                                   │
│     ┌──▼┐ ┌─▼──┐      ┌──▼┐ ┌─▼──┐                               │
│     │H3 │ │ H4 │      │H5 │ │ H6 │                                │
│     └───┘ └────┘      └───┘ └────┘                                │
│                                                                     │
│  Flow Tables (per switch):                                         │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Priority | Match Fields        | Actions        | Stats     │  │
│  ├──────────┼────────────────────┼────────────────┼──────────┤  │
│  │    3     │ IP_SRC, IP_DST,    │ OUTPUT: Port 2 │ 1000 pkts │  │
│  │          │ UDP_DST=5060       │ SET_QUEUE: 3   │ 50 KB     │  │
│  ├──────────┼────────────────────┼────────────────┼──────────┤  │
│  │    2     │ IP_SRC, IP_DST,    │ OUTPUT: Port 3 │ 5000 pkts │  │
│  │          │ TCP_DST=554        │ SET_QUEUE: 2   │ 5 MB      │  │
│  ├──────────┼────────────────────┼────────────────┼──────────┤  │
│  │    1     │ IP_SRC, IP_DST,    │ OUTPUT: Port 1 │ 2000 pkts │  │
│  │          │ TCP_DST=80         │ SET_QUEUE: 1   │ 1 MB      │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Processing Pipeline

```
┌─────────────┐
│   Packet    │
│   Arrives   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ No matching flow?   │───Yes──┐
│ (First packet)      │        │
└──────┬──────────────┘        │
       │ No                    │
       │ (Existing flow)       │
       ▼                       ▼
┌─────────────┐    ┌──────────────────────┐
│   Forward   │    │ Send to Controller   │
│  via Flow   │    │  (Packet-In)         │
│    Rule     │    └──────────┬───────────┘
└─────────────┘               │
                              ▼
                   ┌──────────────────────┐
                   │ Extract Features:    │
                   │ • IP addresses       │
                   │ • Port numbers       │
                   │ • Protocol           │
                   │ • Packet size        │
                   │ • Timing info        │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Collect 10+ packets │
                   │  Calculate stats:    │
                   │  • Avg packet size   │
                   │  • IAT mean/std      │
                   │  • Byte rate         │
                   │  • Packet rate       │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  ML Classification   │
                   │  Input: 18 features  │
                   │  Output: Traffic type│
                   │  Time: < 1ms         │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Get QoS Policy      │
                   │  • Priority level    │
                   │  • Queue assignment  │
                   │  • Bandwidth limit   │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Path Computation    │
                   │  • Find best path    │
                   │  • Check load        │
                   │  • Verify capacity   │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Install Flow Rules  │
                   │  • Match criteria    │
                   │  • Actions           │
                   │  • Priority          │
                   │  • Timeout           │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Forward Packet      │
                   │  Future packets use  │
                   │  installed rule      │
                   └──────────────────────┘
```

## ML Model Training Flow

```
┌─────────────────┐
│  Data Sources   │
├─────────────────┤
│ • CIC-IDS2017   │
│ • UNSW-NB15     │
│ • ISCX VPN      │
│ • Synthetic     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Dataset Processing     │
│ • Load CSV files        │
│ • Unify formats         │
│ • Handle missing values │
│ • Remove duplicates     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature Engineering    │
│ • Extract 18 features   │
│ • Calculate statistics  │
│ • Normalize values      │
│ • Encode categorical    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Data Balancing         │
│ • Equal samples/class   │
│ • Oversample minority   │
│ • Undersample majority  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Train/Test Split       │
│ • 80% Training          │
│ • 20% Testing           │
│ • Stratified split      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Model Training         │
│ • Random Forest         │
│ • Gradient Boosting     │
│ • Neural Network        │
│ • Decision Tree         │
│ • K-NN                  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Model Evaluation       │
│ • Accuracy              │
│ • Precision/Recall      │
│ • F1-Score              │
│ • Confusion Matrix      │
│ • Cross-validation      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Select Best Model      │
│ • Highest accuracy      │
│ • Lowest inference time │
│ • Best generalization   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Save Model             │
│ • Pickle format         │
│ • Include scaler        │
│ • Include encoder       │
│ • Save metadata         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Deploy to Controller   │
│ • Load at startup       │
│ • Ready for inference   │
└─────────────────────────┘
```

## Traffic Flow Example: VoIP Call

```
Step 1: Initial Packet
H1 (10.0.0.1:50000) ──► S1 ──► Controller
     UDP to 10.0.0.3:5060
     
Step 2: Feature Extraction
┌─────────────────────────┐
│ duration: 0             │
│ protocol: UDP (17)      │
│ src_port: 50000         │
│ dst_port: 5060          │
│ packet_size: 120 bytes  │
└─────────────────────────┘

Step 3: Collect 10 packets (0.2 seconds)
┌─────────────────────────┐
│ avg_pkt_size: 115 bytes │
│ iat_mean: 0.02s         │
│ iat_std: 0.001s         │
│ bytes/sec: 5750         │
│ pkts/sec: 50            │
└─────────────────────────┘

Step 4: ML Classification
Feature Vector → Random Forest → "VoIP"
Confidence: 98%

Step 5: QoS Policy
┌─────────────────────────┐
│ Priority: 3 (Highest)   │
│ Queue: 3                │
│ Max Latency: 100ms      │
│ Min Bandwidth: 64 Kbps  │
└─────────────────────────┘

Step 6: Path Selection
S1 → S2 → S3 (shortest, lowest latency)
Bandwidth: Available ✓
Latency: 15ms ✓

Step 7: Flow Installation
┌─────────────────────────────────────┐
│ Match:                              │
│   in_port=1, eth_type=0x0800,      │
│   ip_proto=17, ipv4_src=10.0.0.1,  │
│   ipv4_dst=10.0.0.3,                │
│   udp_src=50000, udp_dst=5060      │
│                                     │
│ Actions:                            │
│   set_queue=3                       │
│   output=2                          │
│                                     │
│ Priority: 3                         │
│ Idle Timeout: 60s                   │
│ Hard Timeout: 300s                  │
└─────────────────────────────────────┘

Step 8: Subsequent Packets
All packets match installed rule
→ Forwarded directly by switch
→ Low latency maintained
→ No controller involvement
```

## Key Components Interaction

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Mininet    │◄────►│     Ryu      │◄────►│   ML Model   │
│   Topology   │      │  Controller  │      │  Classifier  │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │                      │
       │ OpenFlow            │ Function             │ Predict
       │                     │ Calls                │
       ▼                     ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ OVS Switches │      │ Flow Tables  │      │  Features    │
│  (s1-s4)     │      │  Statistics  │      │  18 dims     │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │                      │
       │                     │                      │
       ▼                     ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│    Hosts     │      │   Actions    │      │   Classes    │
│  (h1-h6)     │      │  Priorities  │      │   (5 types)  │
└──────────────┘      └──────────────┘      └──────────────┘
```

This architecture enables:
- ✅ Real-time traffic classification
- ✅ Dynamic QoS enforcement
- ✅ Intelligent routing decisions
- ✅ Scalable performance
- ✅ Low latency operation