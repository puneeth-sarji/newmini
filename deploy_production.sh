#!/bin/bash

# SDN AI Traffic Classifier - Production Deployment Script
# This script sets up the complete production environment

set -e  # Exit on any error

echo "============================================================"
echo "🚀 SDN AI Traffic Classifier - Production Deployment"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

# Get current user
CURRENT_USER=${SUDO_USER:-$USER}
PROJECT_DIR="/home/$CURRENT_USER/Desktop/sdn-ai-traffic-classifier"

print_header "Step 1: System Requirements Check"

# Check Ubuntu version
if ! command -v lsb_release &> /dev/null; then
    print_error "lsb_release not found. Installing..."
    apt-get install -y lsb-release
fi

UBUNTU_VERSION=$(lsb_release -rs)
print_status "Ubuntu Version: $UBUNTU_VERSION"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
print_status "Python Version: $PYTHON_VERSION"

# Check available memory
MEMORY=$(free -h | awk '/^Mem:/ {print $2}')
print_status "Available RAM: $MEMORY"

# Check disk space
DISK_SPACE=$(df -h "$PROJECT_DIR" | awk 'NR==2 {print $4}')
print_status "Available Disk Space: $DISK_SPACE"

print_header "Step 2: Installing System Dependencies"

print_status "Installing system packages..."
apt-get update -qq

# Install core dependencies
apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    curl \
    wget \
    net-tools \
    iperf \
    iperf3 \
    tcpdump \
    htop \
    tree

# Install SDN-specific packages
print_status "Installing SDN packages..."
apt-get install -y \
    mininet \
    openvswitch-switch \
    openvswitch-common \
    bridge-utils

# Install Ryu SDN framework
print_status "Installing Ryu SDN framework..."
pip3 install ryu

# Install network monitoring tools
print_status "Installing network monitoring tools..."
apt-get install -y \
    wireshark \
    nmap \
    iftop \
    bmon

print_header "Step 3: Python Environment Setup"

cd "$PROJECT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment and install Python packages
print_status "Installing Python packages..."
source .venv/bin/activate

pip install --upgrade pip

# Install ML and data science packages
pip install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    joblib \
    tqdm

# Install network packages
pip install \
    scapy \
    requests \
    kaggle

# Install Ryu in virtual environment
pip install ryu

print_header "Step 4: Network Configuration"

# Configure OpenVSwitch
print_status "Configuring OpenVSwitch..."
systemctl enable openvswitch-switch
systemctl start openvswitch-switch

# Check OpenVSwitch status
if systemctl is-active --quiet openvswitch-switch; then
    print_status "OpenVSwitch is running"
else
    print_error "OpenVSwitch failed to start"
    exit 1
fi

# Clean up any existing Mininet instances
print_status "Cleaning up existing Mininet instances..."
mn -c 2>/dev/null || true

print_header "Step 5: ML Model Training"

# Train the ML model
print_status "Training ML model..."
source .venv/bin/activate
python3 ml_models/train_classifier.py

# Verify model exists
if [ -f "traffic_classifier.pkl" ]; then
    print_status "ML model trained successfully"
    MODEL_SIZE=$(ls -lh traffic_classifier.pkl | awk '{print $5}')
    print_status "Model size: $MODEL_SIZE"
else
    print_error "ML model training failed"
    exit 1
fi

print_header "Step 6: Production Configuration"

# Create production configuration
print_status "Creating production configuration..."

# Create logs directory
mkdir -p logs
mkdir -p data

# Set proper permissions
chown -R "$CURRENT_USER:$CURRENT_USER" "$PROJECT_DIR"
chmod +x *.sh

# Create production config file
cat > config/production.json << EOF
{
    "controller": {
        "host": "0.0.0.0",
        "port": 6653,
        "ofp_version": "OpenFlow13",
        "verbose": true
    },
    "topology": {
        "switches": 2,
        "hosts_per_switch": 3,
        "links": [
            ["s1", "s2"]
        ]
    },
    "ml": {
        "model_path": "traffic_classifier.pkl",
        "feature_update_interval": 10,
        "classification_threshold": 0.8
    },
    "logging": {
        "level": "INFO",
        "file": "logs/production.log",
        "max_size": "10MB",
        "backup_count": 5
    },
    "monitoring": {
        "stats_interval": 5,
        "flow_timeout": 60,
        "enable_wireshark": false
    }
}
EOF

print_header "Step 7: Service Setup"

# Create systemd service for controller
print_status "Creating systemd service for SDN controller..."

cat > /etc/systemd/system/sdn-ai-controller.service << EOF
[Unit]
Description=SDN AI Traffic Classifier Controller
After=network.target openvswitch-switch.service
Wants=openvswitch-switch.service

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/.venv/bin
ExecStart=$PROJECT_DIR/.venv/bin/ryu-manager controller/intelligent_controller.py --verbose
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=process
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable sdn-ai-controller

print_header "Step 8: Monitoring Setup"

# Create monitoring script
cat > scripts/monitor.sh << 'EOF'
#!/bin/bash

# SDN AI System Monitoring Script

PROJECT_DIR="/home/$(whoami)/Desktop/sdn-ai-traffic-classifier"
LOG_FILE="$PROJECT_DIR/logs/monitoring.log"

echo "=== SDN AI System Monitoring ===" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"

# Check controller status
if pgrep -f "ryu-manager" > /dev/null; then
    echo "✅ Controller: RUNNING" | tee -a "$LOG_FILE"
else
    echo "❌ Controller: STOPPED" | tee -a "$LOG_FILE"
fi

# Check OpenVSwitch status
if systemctl is-active --quiet openvswitch-switch; then
    echo "✅ OpenVSwitch: RUNNING" | tee -a "$LOG_FILE"
else
    echo "❌ OpenVSwitch: STOPPED" | tee -a "$LOG_FILE"
fi

# Check flow rules
echo "" | tee -a "$LOG_FILE"
echo "Flow Rules:" | tee -a "$LOG_FILE"
sudo ovs-ofctl dump-flows s1 -O OpenFlow13 2>/dev/null | head -10 | tee -a "$LOG_FILE"

# Check system resources
echo "" | tee -a "$LOG_FILE"
echo "System Resources:" | tee -a "$LOG_FILE"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%" | tee -a "$LOG_FILE"
echo "Memory: $(free -h | awk '/^Mem:/ {print $3"/"$2}')" | tee -a "$LOG_FILE"

echo "================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
EOF

chmod +x scripts/monitor.sh

print_header "Step 9: Testing Setup"

# Test Ryu installation
print_status "Testing Ryu installation..."
if command -v ryu-manager &> /dev/null; then
    print_status "Ryu manager is available"
else
    print_error "Ryu manager not found"
    exit 1
fi

# Test Mininet installation
print_status "Testing Mininet installation..."
if command -v mn &> /dev/null; then
    print_status "Mininet is available"
else
    print_error "Mininet not found"
    exit 1
fi

# Test Python packages
print_status "Testing Python packages..."
source .venv/bin/activate

python3 -c "
import numpy, pandas, sklearn, matplotlib, ryu, scapy
print('✅ All Python packages imported successfully')
" || {
    print_error "Python package test failed"
    exit 1
}

print_header "Step 10: Production Deployment Complete"

print_status "Production deployment completed successfully!"
print_status "System is ready for production use"

echo ""
print_header "Next Steps:"

echo -e "${GREEN}1. Start the Controller:${NC}"
echo "   sudo systemctl start sdn-ai-controller"
echo "   sudo systemctl status sdn-ai-controller"

echo ""
echo -e "${GREEN}2. Start the Network:${NC}"
echo "   sudo python3 topology/simple_topology.py"

echo ""
echo -e "${GREEN}3. Monitor the System:${NC}"
echo "   ./scripts/monitor.sh"

echo ""
echo -e "${GREEN}4. View Logs:${NC}"
echo "   tail -f logs/production.log"

echo ""
echo -e "${GREEN}5. Check Flow Rules:${NC}"
echo "   sudo ovs-ofctl dump-flows s1 -O OpenFlow13"

echo ""
print_header "Production Commands:"

echo -e "${BLUE}Start Controller:${NC} sudo systemctl start sdn-ai-controller"
echo -e "${BLUE}Stop Controller:${NC} sudo systemctl stop sdn-ai-controller"
echo -e "${BLUE}Restart Controller:${NC} sudo systemctl restart sdn-ai-controller"
echo -e "${BLUE}Controller Status:${NC} sudo systemctl status sdn-ai-controller"
echo -e "${BLUE}View Logs:${NC} sudo journalctl -u sdn-ai-controller -f"

echo ""
print_status "🎉 SDN AI Traffic Classifier is ready for production!"
print_status "   Access the system at: $PROJECT_DIR"

# Create a quick start script
cat > start_production.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting SDN AI Traffic Classifier Production System"

# Start controller
echo "Starting SDN Controller..."
sudo systemctl start sdn-ai-controller

# Wait for controller to start
sleep 3

# Check status
if sudo systemctl is-active --quiet sdn-ai-controller; then
    echo "✅ Controller started successfully"
else
    echo "❌ Controller failed to start"
    exit 1
fi

echo ""
echo "🌐 Starting Mininet topology..."
echo "Open a new terminal and run:"
echo "sudo python3 topology/simple_topology.py"
echo ""
echo "📊 Monitor the system with:"
echo "./scripts/monitor.sh"
EOF

chmod +x start_production.sh

echo ""
print_status "Quick start script created: ./start_production.sh"

exit 0