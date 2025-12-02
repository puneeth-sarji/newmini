# Complete Execution Guide

Step-by-step guide to get your AI-based SDN Traffic Classification system running on Ubuntu.

## 📋 Prerequisites Check

Before starting, verify:

```bash
# Check Ubuntu version (20.04+ recommended)
lsb_release -a

# Check Python version (3.8+ required)
python3 --version

# Check available disk space (minimum 20GB for datasets)
df -h ~

# Check RAM (minimum 4GB recommended)
free -h

# Check if you have sudo access
sudo echo "Sudo access OK"
```

## 🚀 Installation Methods

Choose the method that suits your needs:

### Method 1: Fully Automated Setup (Recommended)

**Time**: 5-30 minutes depending on options  
**Difficulty**: ⭐ Easy

```bash
# 1. Download the automated setup script
cd ~
mkdir sdn-ai-project
cd sdn-ai-project

# 2. Copy all the scripts from artifacts:
# - auto_setup_with_datasets.sh
# - setup.sh
# - setup_kaggle.sh
# - All Python files

# 3. Make executable
chmod +x auto_setup_with_datasets.sh

# 4. Run automated setup
./auto_setup_with_datasets.sh
```

**During setup, you'll be asked:**
```
Choose setup option:
1) Quick setup (synthetic data only - 5 minutes)
2) Full setup with real datasets (requires Kaggle - 30+ minutes)
3) Full setup with manual dataset download
```

**Choose 1** for quick testing  
**Choose 2** for production with Kaggle API  
**Choose 3** for manual dataset management

### Method 2: Step-by-Step Manual Setup

**Time**: 15-45 minutes  
**Difficulty**: ⭐⭐ Moderate

#### Step 1: System Setup
```bash
cd ~
mkdir -p sdn-ai-traffic-classifier
cd sdn-ai-traffic-classifier

# Create directory structure
mkdir -p controller ml_models topology utils traffic_data logs
mkdir -p traffic_data/{cicids2017,unsw_nb15,processed}
```

#### Step 2: Install Dependencies
```bash
# System packages
sudo apt update
sudo apt install -y python3 python3-pip python3-dev \
    mininet openvswitch-switch net-tools iperf iperf3 \
    tcpdump git curl build-essential

# Python packages
pip3 install --upgrade pip
pip3 install ryu numpy pandas scikit-learn matplotlib \
    seaborn scapy requests tqdm kaggle
```

#### Step 3: Copy Project Files
```bash
# Copy all provided Python files to their directories:
# - controller/intelligent_controller.py
# - ml_models/train_classifier.py
# - ml_models/train_classifier_real.py
# - ml_models/dataset_processor.py
# - ml_models/dataset_downloader.py
# - topology/simple_topology.py
# - utils/traffic_generator.py
```

#### Step 4: Setup Kaggle (Optional)
```bash
# Install Kaggle CLI
pip3 install kaggle

# Create Kaggle directory
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle

# Get your API key:
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Test authentication
kaggle datasets list
```

#### Step 5: Download Datasets (Optional)
```bash
cd ~/sdn-ai-traffic-classifier

# Option A: Using Kaggle
python3 ml_models/dataset_downloader.py

# Option B: Manual download (see DATASET_GUIDE.md)
# Download and place CSVs in traffic_data/cicids2017/ etc.
```

#### Step 6: Process Datasets
```bash
# Process and unify all datasets
python3 ml_models/dataset_processor.py

# This creates:
# traffic_data/processed/unified_traffic_dataset.csv
```

#### Step 7: Train Model
```bash
# Quick training (synthetic data)
python3 ml_models/train_classifier.py

# OR Production training (real datasets)
python3 ml_models/train_classifier_real.py
```

## ▶️ Running the System

### Option A: Using Helper Scripts

**Terminal 1 - Start Controller:**
```bash
cd ~/sdn-ai-traffic-classifier
./start.sh
```

**Terminal 2 - Start Network:**
```bash
cd ~/sdn-ai-traffic-classifier
sudo python3 topology/simple_topology.py
```

### Option B: Manual Commands

**Terminal 1 - Controller:**
```bash
cd ~/sdn-ai-traffic-classifier
ryu-manager controller/intelligent_controller.py --verbose
```

Wait for: `"Intelligent SDN Controller Started"`

**Terminal 2 - Mininet:**
```bash
cd ~/sdn-ai-traffic-classifier
sudo python3 topology/simple_topology.py
```

Wait for: `mininet>` prompt

## 🧪 Testing the System

### 1. Basic Connectivity Test
```bash
# In Mininet CLI (Terminal 2)
mininet> pingall

# Expected output:
# *** Ping: testing ping reachability
# h1 -> h2 h3 h4 h5 h6
# h2 -> h1 h3 h4 h5 h6
# ... (all hosts reachable)
# *** Results: 0% dropped
```

### 2. Generate Traffic

**HTTP Traffic:**
```bash
mininet> h1 python3 utils/traffic_generator.py 10.0.0.1 10.0.0.3 http &
```

**Video Streaming:**
```bash
mininet> h2 python3 utils/traffic_generator.py 10.0.0.2 10.0.0.4 video &
```

**VoIP Call:**
```bash
mininet> h3 python3 utils/traffic_generator.py 10.0.0.3 10.0.0.5 voip &
```

**Online Gaming:**
```bash
mininet> h4 python3 utils/traffic_generator.py 10.0.0.4 10.0.0.6 gaming &
```

**File Transfer:**
```bash
mininet> h5 python3 utils/traffic_generator.py 10.0.0.5 10.0.0.1 ftp &
```

**Mixed Traffic:**
```bash
mininet> h1 python3 utils/traffic_generator.py 10.0.0.1 10.0.0.6 mixed &
```

### 3. Observe Controller Output

In **Terminal 1** (Controller), you should see:
```
Switch connected: 1
Flow classified as HTTP, Priority: 1
Flow classified as Video, Priority: 2
Flow classified as VoIP, Priority: 3
=== Flow Statistics ===
Flow: 10.0.0.1:xxxx -> 10.0.0.3:80 | Packets: 45 | Bytes: 12500
...
```

### 4. Check Flow Tables
```bash
# In a new terminal
sudo ovs-ofctl dump-flows s1 -O OpenFlow13
sudo ovs-ofctl dump-flows s2 -O OpenFlow13
```

Expected output shows flows with different priorities:
```
priority=3,udp,nw_dst=10.0.0.5,tp_dst=5060 actions=output:2  # VoIP
priority=2,tcp,nw_dst=10.0.0.4,tp_dst=554 actions=output:3  # Video
priority=1,tcp,nw_dst=10.0.0.3,tp_dst=80 actions=output:1   # HTTP
```

### 5. Performance Tests

**Bandwidth Test:**
```bash
mininet> iperf h1 h3
```

**Latency Test:**
```bash
mininet> h1 ping -c 100 h3
```

**Jitter Test:**
```bash
mininet> h1 ping -i 0.01 -c 1000 h3 | tail -1
```

## 📊 Monitoring and Verification

### View Real-time Statistics

**In Controller Terminal**, statistics update every 10 seconds showing:
- Flow classifications
- Packet counts
- Byte counts
- Throughput per flow

### Check Model Performance

**After training**, you should see:
```
Model Accuracy: 0.9500
Classification Report:
              precision    recall  f1-score   support
     HTTP       0.96      0.95      0.95      2000
     Video      0.95      0.96      0.95      2000
     VoIP       0.97      0.96      0.96      2000
     Gaming     0.94      0.95      0.94      2000
     FTP        0.93      0.93      0.93      2000
```

### Monitor Network with Wireshark

```bash
# Start Wireshark
sudo wireshark &

# Select interface: s1-eth1, s2-eth1, etc.
# Apply filter: openflow
```

### View System Resources

```bash
# CPU and memory usage
top -p $(pgrep -f ryu-manager)

# Network statistics
watch -n 1 'sudo ovs-ofctl dump-ports s1 -O OpenFlow13'
```

## 🎮 Advanced Testing Scenarios

### Scenario 1: QoS Verification

Test that VoIP gets priority over bulk transfers:

```bash
# Terminal 1: Start bulk transfer
mininet> h1 iperf -s &
mininet> h2 iperf -c 10.0.0.1 -t 60 &

# Terminal 2: Start VoIP (should maintain low latency)
mininet> h3 ping -i 0.02 10.0.0.5
```

Expected: VoIP ping times should remain low (<50ms) despite bulk transfer.

### Scenario 2: Load Balancing

```bash
# Generate traffic on multiple paths
mininet> h1 iperf -s &
mininet> h3 iperf -s &

mininet> h2 iperf -c 10.0.0.1 -t 30 &
mininet> h4 iperf -c 10.0.0.3 -t 30 &

# Check flow distribution
sudo ovs-ofctl dump-flows s2 -O OpenFlow13
```

### Scenario 3: Traffic Classification Accuracy

```bash
# Generate known traffic and verify classification
mininet> h1 python3 utils/traffic_generator.py 10.0.0.1 10.0.0.3 voip &

# Check controller log for correct classification
# Should show: "Flow classified as VoIP, Priority: 3"
```

## 🛑 Stopping the System

### Graceful Shutdown

**In Mininet CLI (Terminal 2):**
```bash
mininet> exit
```

**In Controller Terminal (Terminal 1):**
```
Press Ctrl+C
```

### Complete Cleanup

```bash
# Clean Mininet
sudo mn -c

# Kill any remaining processes
pkill -f ryu-manager
pkill -f python3

# Verify cleanup
sudo ovs-vsctl show  # Should show no bridges
```

## 🔄 Restarting After Changes

### After Code Changes
```bash
# No need to retrain model unless ML code changed
# Just restart controller
pkill -f ryu-manager
ryu-manager controller/intelligent_controller.py --verbose
```

### After Model Retraining
```bash
# Retrain model
python3 ml_models/train_classifier_real.py

# Restart controller (will load new model)
pkill -f ryu-manager
ryu-manager controller/intelligent_controller.py --verbose
```

### After Topology Changes
```bash
# Cleanup
sudo mn -c

# Restart network
sudo python3 topology/simple_topology.py
```

## ⚠️ Troubleshooting During Execution

### Issue: "Address already in use" (Port 6653)
```bash
# Find and kill the process
sudo netstat -tlnp | grep 6653
kill -9 <PID>

# Or simply
pkill -f ryu-manager
```

### Issue: Mininet won't start
```bash
# Clean everything
sudo mn -c
sudo service openvswitch-switch restart

# Try again
sudo python3 topology/simple_topology.py
```

### Issue: No traffic classification shown
```bash
# Check if model is loaded
ls -lh ml_models/*.pkl

# Verify controller can read it
python3 -c "import pickle; print(pickle.load(open('ml_models/traffic_classifier.pkl', 'rb')).keys())"

# Check controller logs for errors
```

### Issue: Hosts can't ping each other
```bash
# In Mininet
mininet> h1 ifconfig  # Check IP
mininet> net  # Check topology
mininet> links  # Check connections

# Verify switches are connected to controller
sudo ovs-vsctl show
```

### Issue: High latency for all traffic
```bash
# Check CPU usage
top

# Check for packet drops
sudo ovs-ofctl dump-ports s1 -O OpenFlow13 | grep drop

# Reduce traffic generation
# Kill some traffic generator processes
```

## 📝 Logging and Debugging

### Enable Debug Logging
```bash
# Start controller with debug
ryu-manager --verbose --observe-links controller/intelligent_controller.py

# Or modify controller code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Save Logs to File
```bash
# Controller logs
ryu-manager controller/intelligent_controller.py --verbose 2>&1 | tee logs/controller.log

# Mininet logs
sudo python3 topology/simple_topology.py 2>&1 | tee logs/mininet.log
```

### Capture Traffic
```bash
# Capture packets for analysis
sudo tcpdump -i s1-eth1 -w logs/traffic.pcap

# Analyze later with
wireshark logs/traffic.pcap
```

## ✅ Verification Checklist

Before considering your system working, verify:

- [ ] Controller starts without errors
- [ ] Mininet creates topology successfully
- [ ] All hosts can ping each other (pingall succeeds)
- [ ] Traffic generator runs without errors
- [ ] Controller logs show traffic classification
- [ ] Different priorities are assigned correctly
- [ ] Flow tables show entries with correct priorities
- [ ] Model accuracy is >90%
- [ ] System performance is acceptable

## 🎯 Next Steps

Once your system is running:

1. **Experiment with different traffic patterns**
2. **Modify QoS policies**
3. **Add new traffic classes**
4. **Test with different topologies**
5. **Measure and optimize performance**
6. **Integrate with real hardware** (if available)
7. **Develop visualization dashboard**
8. **Extend ML models**

## 📚 Additional Resources

- Controller API: Check Ryu documentation
- OpenFlow spec: OpenNetworking Foundation
- ML tuning: scikit-learn documentation
- Network testing: iperf3, tcpdump, Wireshark guides

---

**Your system should now be running successfully! 🎉**

If you encounter issues not covered here, check:
1. README.md for general information
2. DATASET_GUIDE.md for dataset issues
3. System logs in logs/ directory
4. GitHub issues (if using a repository)