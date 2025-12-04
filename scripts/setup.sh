#!/bin/bash

# SDN AI Traffic Classifier - Setup Script (Python 3.12 Compatible)
# Run this script to set up the complete environment

set -e

echo "================================"
echo "SDN AI Traffic Classifier Setup"
echo "Python 3.12 Compatible Version"
echo "================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Install Python 3.12 and dependencies
echo "[2/9] Installing Python 3.12 and essential tools..."
apt install -y python3.12 python3.12-dev python3-pip python3.12-venv git curl build-essential

# Update alternatives to use Python 3.12
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install Mininet
echo "[3/9] Installing Mininet..."
apt install -y mininet

# Install OpenVSwitch
echo "[4/9] Installing OpenVSwitch..."
apt install -y openvswitch-switch openvswitch-common

# Install networking tools
echo "[5/9] Installing networking tools..."
apt install -y net-tools iperf iperf3 tcpdump wireshark

# Upgrade pip for Python 3.12
echo "[6/9] Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install Python packages compatible with Python 3.12
echo "[7/9] Installing Python packages (Python 3.12 compatible)..."
pip3 install --upgrade \
    ryu \
    eventlet \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    scapy \
    kaggle \
    requests \
    tqdm \
    imbalanced-learn \
    greenlet

# Create project structure
echo "[8/9] Creating project structure..."
PROJECT_DIR="$HOME/sdn-ai-traffic-classifier"
mkdir -p "$PROJECT_DIR"/{controller,ml_models,traffic_data,topology,logs,utils}

# Set permissions
chown -R $SUDO_USER:$SUDO_USER "$PROJECT_DIR"

# Test installations
echo "[9/9] Testing installations..."
echo "Python version:"
python3 --version

echo "Pip version:"
pip3 --version

echo "Checking installed packages..."
python3 -c "import ryu; print('Ryu: OK')"
python3 -c "import sklearn; print('scikit-learn: OK')"
python3 -c "import numpy; print('NumPy: OK')"
python3 -c "import pandas; print('Pandas: OK')"

echo "Ryu version:"
ryu-manager --version || echo "Ryu installed"

echo "Mininet version:"
mn --version

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo "Project directory: $PROJECT_DIR"
echo "Python version: $(python3 --version)"
echo ""
echo "Next steps:"
echo ""
echo "OPTION A - Quick Start (Synthetic Data):"
echo "  1. cd $PROJECT_DIR"
echo "  2. python3 ml_models/train_classifier.py"
echo "  3. ryu-manager controller/intelligent_controller.py"
echo "  4. sudo python3 topology/simple_topology.py"
echo ""
echo "================================"
