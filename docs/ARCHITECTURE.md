# System Architecture Diagram

## Complete System Overview (v2.0 - Enhanced with Large Scale ML)

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
│  │  • HTTP (Web)     • HTTPS (Secure)  • Video (Streaming)   │  │
│  │  • VoIP (Voice)   • SSH (Remote)     • FTP (Transfer)     │  │
│  │                                                              │  │
│  │  Available Models:                                           │  │
│  │  • Large Scale (99.83% acc, 6 classes, 14 features)        │  │
│  │  • Real Traffic (95-98% acc, 5 classes, 18 features)       │  │
│  │  • Basic (90-95% acc, 5 classes, 12 features)              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │           Routing & QoS Management                          │  │
│  │                                                              │  │
│  │  Priority Mapping:           Path Computation:              │  │
│  │  VoIP    → Priority 5       Dijkstra's Algorithm            │  │
│  │  Video   → Priority 4       K-Shortest Paths                │  │
│  │  HTTPS   → Priority 3       Load Balancing                  │  │
│  │  HTTP    → Priority 2       Congestion Avoidance            │  │
│  │  SSH     → Priority 1       ML-based Optimization           │  │
│  │  FTP     → Priority 0       Real-time Adaptation            │  │
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
                    │ • Flow statistics    │
                    │ • Directional data   │
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
                    │  • Flow duration     │
                    │  • Directional stats │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                    │  ML Classification   │
                    │  Input: 14-18 features│
                    │  Output: Traffic type│
                    │  Time: < 1ms         │
                    │  Models: RF, GB, NN  │
                    │  Accuracy: 90-99.83% │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                    │  Get QoS Policy      │
                    │  • Priority level    │
                    │  • Queue assignment  │
                    │  • Bandwidth limit   │
                    │  • Latency target    │
                    │  • Jitter control    │
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

## ML Model Training Flow (Enhanced v2.0)

```
┌─────────────────┐
│  Data Sources   │
├─────────────────┤
│ • CIC-IDS2017   │
│ • UNSW-NB15     │
│ • ISCX VPN      │
│ • Synthetic     │
│ • Custom Data   │
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
│ • Extract 14-18 features │
│ • Calculate statistics  │
│ • Normalize values      │
│ • Encode categorical    │
│ • Feature selection     │
│ • Dimensionality reduction│
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
│ • Random Forest (Primary)│
│ • Gradient Boosting     │
│ • Neural Network        │
│ • Decision Tree         │
│ • K-NN                  │
│ • Ensemble Methods      │
│ • Hyperparameter Tuning │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Model Evaluation       │
│ • Accuracy (90-99.83%)  │
│ • Precision/Recall      │
│ • F1-Score              │
│ • Confusion Matrix      │
│ • Cross-validation      │
│ • ROC/AUC analysis      │
│ • Per-class metrics     │
│ • Feature importance    │
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
│ • Version control       │
│ • Performance metrics   │
│ • Feature mapping       │
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
│ Priority: 5 (Highest)   │
│ Queue: 5                │
│ Max Latency: 50ms       │
│ Min Bandwidth: 64 Kbps  │
│ Jitter: <5ms           │
│ Packet Loss: <1%       │
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
│ Priority: 5                         │
│ Idle Timeout: 60s                   │
│ Hard Timeout: 300s                  │
│ ML Confidence: 98%                 │
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
│   Mininet    │◄────►│     Ryu      │◄────►│   ML Models  │
│   Topology   │      │  Controller  │      │  (3 Models)  │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │                      │
       │ OpenFlow            │ Function             │ Predict
       │                     │ Calls                │
       ▼                     ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ OVS Switches │      │ Flow Tables  │      │  Features    │
│  (s1-s4)     │      │  Statistics  │      │ 14-18 dims   │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │                      │
       │                     │                      │
       ▼                     ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│    Hosts     │      │   Actions    │      │   Classes    │
│  (h1-h6)     │      │  Priorities  │      │  (5-6 types) │
└──────────────┘      └──────────────┘      └──────────────┘
```

This architecture enables:
- ✅ Real-time traffic classification (<1ms)
- ✅ High accuracy (90-99.83%)
- ✅ Dynamic QoS enforcement
- ✅ Intelligent routing decisions
- ✅ Scalable performance (1000+ flows)
- ✅ Low latency operation (<10ms additional)
- ✅ Multiple model support
- ✅ Comprehensive testing framework
- ✅ Production-ready deployment