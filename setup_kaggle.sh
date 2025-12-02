#!/bin/bash

# Setup Kaggle API for Dataset Downloads
# This script helps configure Kaggle API authentication

set -e

echo "=========================================="
echo "Kaggle API Setup for Dataset Downloads"
echo "=========================================="
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install Python3 and pip3 first."
    exit 1
fi

# Install Kaggle API
echo "[1/4] Installing Kaggle API..."
pip3 install --upgrade kaggle

# Check if installation was successful
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle command not found after installation"
    exit 1
fi

echo "✓ Kaggle API installed successfully"
echo ""

# Create .kaggle directory
echo "[2/4] Setting up Kaggle directory..."
KAGGLE_DIR="$HOME/.kaggle"
mkdir -p "$KAGGLE_DIR"
chmod 700 "$KAGGLE_DIR"
echo "✓ Created directory: $KAGGLE_DIR"
echo ""

# Check if kaggle.json exists
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

if [ -f "$KAGGLE_JSON" ]; then
    echo "[3/4] kaggle.json already exists"
    chmod 600 "$KAGGLE_JSON"
    echo "✓ Updated permissions"
else
    echo "[3/4] kaggle.json not found"
    echo ""
    echo "To get your Kaggle API credentials:"
    echo "1. Go to: https://www.kaggle.com/account"
    echo "2. Scroll down to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. Save the downloaded kaggle.json file"
    echo ""
    
    read -p "Do you have kaggle.json ready? (y/n): " response
    
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo ""
        echo "Please enter the full path to your kaggle.json file:"
        read -p "Path: " json_path
        
        if [ -f "$json_path" ]; then
            cp "$json_path" "$KAGGLE_JSON"
            chmod 600 "$KAGGLE_JSON"
            echo "✓ kaggle.json copied and configured"
        else
            echo "Error: File not found at $json_path"
            exit 1
        fi
    else
        echo ""
        echo "Please follow these steps:"
        echo "1. Visit: https://www.kaggle.com/account"
        echo "2. Create API token"
        echo "3. Move kaggle.json to: $KAGGLE_DIR"
        echo "4. Run this script again"
        exit 1
    fi
fi

echo ""

# Test Kaggle API
echo "[4/4] Testing Kaggle API authentication..."
if kaggle datasets list &> /dev/null; then
    echo "✓ Kaggle API authentication successful!"
else
    echo "✗ Kaggle API authentication failed"
    echo "Please check your kaggle.json file"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now download datasets using:"
echo "  python3 ml_models/dataset_downloader.py"
echo ""
echo "Or manually with Kaggle CLI:"
echo "  kaggle datasets download -d <dataset-name>"
echo ""
echo "Popular datasets for this project:"
echo "  - cicdataset/cicids2017"
echo "  - mrwellsdavid/unsw-nb15"
echo "  - crawford/network-traffic-dataset"
echo ""
echo "=========================================="