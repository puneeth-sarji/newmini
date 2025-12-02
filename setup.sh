#!/bin/bash

# SDN AI Traffic Classifier - Setup Script
# Run this script to set up the complete environment

set -e

echo "================================"
echo "SDN AI Traffic Classifier Setup"
echo "================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Update system
echo "[1/8] Updating system packages..."
apt update && apt upgrade -y

# Install Python and dependencies
echo "[2/8] Installing Python and essential tools..."
apt install -y python3 python3-pip python3-dev git curl build-essential

# Install Mininet
echo "[3/8] Installing Mininet..."
apt install -y mininet

# Install OpenVSwitch
echo "[4/8] Installing OpenVSwitch..."
apt install -y openvswitch-switch openvswitch-common

# Install networking tools
echo "[5/8] Installing networking tools..."
apt install -y net-tools iperf iperf3 tcpdump wireshark

# Install Python packages
echo "[6/8] Installing Python packages..."
pip3 install --upgrade pip
pip3 install ryu eventlet numpy pandas scikit-learn matplotlib seaborn scapy kaggle requests tqdm

# Create project structure
echo "[7/8] Creating project structure..."
PROJECT_DIR="$HOME/sdn-ai-traffic-classifier"
mkdir -p "$PROJECT_DIR"/{controller,ml_models,traffic_data,topology,logs,utils}

# Set permissions
chown -R $SUDO_USER:$SUDO_USER "$PROJECT_DIR"

# Test installations
echo "[8/8] Testing installations..."
echo "Python version:"
python3 --version

echo "Ryu version:"
ryu-manager --version

echo "Mininet version:"
mn --version

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo "Project directory: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo ""
echo "OPTION A - Quick Start (Synthetic Data):"
echo "  1. cd $PROJECT_DIR"
echo "  2. python3 ml_models/train_classifier.py"
echo "  3. ryu-manager controller/intelligent_controller.py"
echo "  4. sudo python3 topology/simple_topology.py"
echo ""
echo "OPTION B - Real Datasets (Recommended):"
echo "  1. cd $PROJECT_DIR"
echo "  2. ./setup_kaggle.sh  # Setup Kaggle API"
echo "  3. python3 ml_models/dataset_downloader.py"
echo "  4. python3 ml_models/dataset_processor.py"
echo "  5. python3 ml_models/train_classifier_real.py"
echo "  6. ryu-manager controller/intelligent_controller.py"
echo "  7. sudo python3 topology/simple_topology.py"
echo ""
echo "See DATASET_GUIDE.md for detailed dataset instructions"
echo "For detailed instructions, check the README.md file"