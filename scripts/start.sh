#!/bin/bash

# Quick Start Script for SDN AI System

cd "$(dirname "$0")"

echo "=========================================="
echo "SDN AI Traffic Classifier - Quick Start"
echo "=========================================="
echo ""
echo "This will start the Ryu controller."
echo "Open another terminal and run:"
echo "  sudo python3 topology/simple_topology.py"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Check if model exists
if [ ! -f "ml_models/traffic_classifier.pkl" ] && [ ! -f "ml_models/traffic_classifier_real.pkl" ]; then
    echo "⚠ Warning: No trained model found!"
    echo "Training model with synthetic data..."
    python3 ml_models/train_classifier.py
    echo ""
fi

# Start controller
ryu-manager controller/intelligent_controller.py --verbose