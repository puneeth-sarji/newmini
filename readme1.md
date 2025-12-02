# AI-Based Network Traffic Classification and Routing Management System in SDN

An intelligent Software-Defined Networking (SDN) system that uses Machine Learning to classify network traffic and optimize routing decisions in real-time.

## 🎯 Features

- **AI-Powered Traffic Classification**: Automatically classifies traffic into HTTP, Video, VoIP, File Transfer, and Gaming
- **Dynamic Routing**: Prioritizes traffic based on application requirements
- **Real-time Monitoring**: Continuous flow statistics and network monitoring
- **Scalable Architecture**: Built on Ryu controller and Mininet for easy testing and deployment
- **Machine Learning Models**: Uses Random Forest for accurate traffic classification

## 📋 Prerequisites

- Ubuntu 20.04 or 22.04 LTS
- Minimum 4GB RAM
- Python 3.8+
- Root/sudo access

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Clone or download all project files to your home directory
cd ~
mkdir -p sdn-ai-traffic-classifier
cd sdn-ai-traffic-classifier

# Make setup script executable
chmod +x setup.sh

# Run setup (requires sudo)
sudo ./setup.sh
```

### 2. Copy Project Files

After setup, copy these files to their respective directories:

```bash
# Copy files to appropriate locations
cp intelligent_controller.py controller/
cp train_classifier.py ml_models/
cp traffic_generator.py utils/
cp simple_topology.py topology/
```

### 3. Choose Your Training Approach

#### Option A: Quick Start (Synthetic Data)

```bash
cd ~/sdn-ai-traffic-classifier
python3 ml_models/train_classifier.py
```

This will:
- Generate synthetic training data
- Train a Random Forest classifier
- Display accuracy metrics and confusion matrix
- Save the model as `traffic_classifier.pkl`

#### Option B: Real-World Datasets (Recommended for Production)

```bash
# 1. Setup Kaggle API (optional but recommended)
chmod +x setup_kaggle.sh
./setup_kaggle.sh

# 2. Download datasets
python3 ml_models/dataset_downloader.py

# 3. Process datasets (unifies multiple sources)
python3 ml_models/dataset_processor.py

# 4. Train with real data (tests multiple algorithms)
python3 ml_models/train_classifier_real.py
```

This will:
- Download/use CIC-IDS2017, UNSW-NB15, and other datasets
- Process and unify different dataset formats
- Train multiple ML models (RF, GB, NN, etc.)
- Compare model performance
- Save the best model with 95%+ accuracy

**See [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed dataset instructions**

### 4. Verify Model Training

Check that the model was created successfully:

```bash
# For synthetic data approach
ls -lh ml_models/traffic_classifier.pkl

# For real dataset approach
ls -lh ml_models/traffic_classifier_real.pkl
```

### 5. Run the System

#### Option A: Using the run script

```bash
chmod +x run_system.sh
./run_system.sh
```

#### Option B: Manual start

**Terminal 1 - Start Controller:**
```bash
cd ~/sdn-ai-traffic-classifier
ryu-manager controller/intelligent_controller.py --verbose
```

**Terminal 2 - Start Network:**
```bash
cd ~/sdn-ai-traffic-classifier
sudo python3 topology/simple_topology.py
```

## 🔬 Testing the System

Once Mininet CLI appears, you can test the system:

### Basic Connectivity Test
```bash
mininet> pingall
```

### Generate Traffic Between Hosts

**HTTP Traffic:**
```bash
mininet> h1 python3 utils/traffic_generator.py 10.0.0.1 10.0.0.3 http &
```

**Video Traffic:**
```bash
mininet> h2 python3 utils/traffic_generator.py 10.0.0.2 10.0.0.4 video &
```

**VoIP Traffic:**
```bash
mininet> h1 python3 utils/traffic_generator.py 10.0.0.1 10.0.0.5 voip &
```

**Gaming Traffic:**
```bash
mininet> h3 python3 utils/traffic_generator.py 10.0.0.3 10.0.0.6 gaming &
```

**Mixed Traffic:**
```bash
mininet> h1 python3 utils/traffic_generator.py 10.0.0.1 10.0.0.6 mixed &
```

### Monitor Traffic

Check the controller terminal to see:
- Traffic classification results
- Flow statistics
- Routing decisions
- Priority assignments

### Performance Testing

**Test bandwidth:**
```bash
mininet> iperf h1 h3
```

**Test latency:**
```bash
mininet> h1 ping -c 10 h3
```

### View Flow Tables

```bash
mininet> sh ovs-ofctl dump-flows s1 -O OpenFlow13
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────┐
│        Application Layer (ML Model)          │
│  - Traffic Classification                    │
│  - Routing Optimization                      │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│     Control Layer (Ryu Controller)           │
│  - Flow Management                           │
│  - Statistics Collection                     │
│  - Policy Enforcement                        │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  Infrastructure Layer (OpenFlow Switches)    │
│  - Packet Forwarding                         │
│  - Flow Tables                               │
└─────────────────────────────────────────────┘
```

## 🎓 Traffic Classification Categories

| Traffic Type | Priority | Characteristics | Applications |
|-------------|----------|-----------------|--------------|
| VoIP | High (3) | Small packets, regular intervals | Voice calls, conferencing |
| Gaming | High (3) | Small packets, low latency | Online games |
| Video | Medium (2) | Large packets, constant bitrate | Streaming, video calls |
| HTTP | Low (1) | Variable packet sizes | Web browsing |
| FTP | Lowest (0) | Large continuous flows | File downloads |

## 🔧 Configuration

### Modify Network Topology

Edit `topology/simple_topology.py` to:
- Add more switches/hosts
- Change bandwidth constraints
- Modify network layout

### Adjust ML Model

Edit `ml_models/train_classifier.py` to:
- Change model type (Random Forest, SVM, Decision Tree)
- Adjust hyperparameters
- Add new traffic classes
- Modify feature extraction

### Controller Behavior

Edit `controller/intelligent_controller.py` to:
- Change priority mappings
- Modify flow timeout values
- Adjust monitoring intervals
- Implement custom routing algorithms

## 📈 Monitoring and Debugging

### View Controller Logs
```bash
tail -f logs/controller.log
```

### Check OpenVSwitch Status
```bash
sudo ovs-vsctl show
```

### Monitor Network Traffic
```bash
sudo tcpdump -i any -n
```

### Wireshark Analysis
```bash
sudo wireshark &
# Select interface and filter by OpenFlow port
```

## 🛠️ Troubleshooting

### Controller Won't Start
```bash
# Check if port 6653 is already in use
sudo netstat -tulpn | grep 6653

# Kill existing controller
pkill -f ryu-manager
```

### Mininet Issues
```bash
# Clean up Mininet
sudo mn -c

# Restart OpenVSwitch
sudo service openvswitch-switch restart
```

### Model Not Loading
```bash
# Retrain the model
cd ~/sdn-ai-traffic-classifier
python3 ml_models/train_classifier.py
```

### Permission Errors
```bash
# Fix permissions
sudo chown -R $USER:$USER ~/sdn-ai-traffic-classifier
```

## 🔬 Advanced Usage

### Custom Traffic Patterns

Create your own traffic generator in `utils/`:

```python
from traffic_generator import TrafficGenerator

gen = TrafficGenerator('10.0.0.1', '10.0.0.3')
gen.generate_custom_traffic(
    packet_size=500,
    interval=0.01,
    duration=30
)
```

### Integrate Real ML Model

Replace rule-based classification with trained model:

```python
# In controller/intelligent_controller.py
def _classify_traffic(self, flow_key):
    features = self._extract_features(flow_key)
    prediction = self.classifier.predict([features])
    return self.label_encoder.inverse_transform(prediction)[0]
```

### Multi-Path Routing

Implement K-shortest paths algorithm for load balancing.

### QoS Enforcement

Add queue configurations for different traffic classes.

## 📚 Project Structure

```
sdn-ai-traffic-classifier/
├── controller/
│   └── intelligent_controller.py    # Main SDN controller
├── ml_models/
│   ├── train_classifier.py          # ML model training
│   └── traffic_classifier.pkl       # Trained model (generated)
├── topology/
│   └── simple_topology.py           # Network topology
├── utils/
│   └── traffic_generator.py         # Traffic generation tool
├── logs/                             # Log files
├── traffic_data/                     # Captured traffic data
├── setup.sh                          # Installation script
├── run_system.sh                     # System startup script
└── README.md                         # This file
```

## 🎯 Future Enhancements

- [ ] Deep Learning models (CNN, LSTM) for better accuracy
- [ ] Real-time network visualization dashboard
- [ ] Integration with real network hardware
- [ ] Encrypted traffic classification
- [ ] DDoS detection and mitigation
- [ ] Multi-controller deployment
- [ ] REST API for external integration
- [ ] Performance benchmarking suite

## 📖 References

- [Ryu SDN Framework](https://ryu-sdn.org/)
- [Mininet Network Emulator](http://mininet.org/)
- [OpenFlow Specification](https://opennetworking.org/software-defined-standards/specifications/)
- [scikit-learn Documentation](https://scikit-learn.org/)

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is for educational and research purposes.

## 👨‍💻 Author

Created for SDN and AI research and education.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review controller logs
3. Verify all dependencies are installed
4. Ensure proper permissions

---

**Happy SDN-ing! 🚀**