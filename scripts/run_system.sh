#!/bin/bash

# SDN AI Traffic Classifier - Run Script
# This script starts the complete system

PROJECT_DIR="/home/puneeth8055/Desktop/sdn-ai-traffic-classifier"

echo "================================"
echo "Starting SDN AI System"
echo "================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    sudo mn -c
    pkill -f ryu-manager
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if running in project directory
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found at $PROJECT_DIR"
    echo "Please run setup.sh first"
    exit 1
fi

cd "$PROJECT_DIR"

# Train ML model if not exists
if [ ! -f "ml_models/traffic_classifier.pkl" ]; then
    echo "[1/3] Training ML model (first time only)..."
    python3 ml_models/train_classifier.py
    echo ""
fi

# Start Ryu controller in background
echo "[2/3] Starting Ryu SDN Controller..."
ryu-manager controller/intelligent_controller.py --verbose &
CONTROLLER_PID=$!
echo "Controller PID: $CONTROLLER_PID"
sleep 3

# Start Mininet topology
echo "[3/3] Starting Mininet topology..."
echo "Waiting for controller to be ready..."
sleep 2

sudo python3 topology/simple_topology.py

# Cleanup when topology exits
cleanup