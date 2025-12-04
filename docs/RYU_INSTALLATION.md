# Ryu SDN Framework Installation Guide

## Option 1: System Package Installation (Recommended)

```bash
# Install Ryu using system package manager
sudo apt update
sudo apt install python3-ryu

# Install additional dependencies
sudo apt install python3-pip python3-dev libssl-dev libffi-dev
```

## Option 2: Virtual Environment with Compatible Python

```bash
# Create virtual environment with Python 3.10 (compatible with Ryu)
python3.10 -m venv ryu-env
source ryu-env/bin/activate

# Install Ryu
pip install ryu

# Install dependencies
pip install eventlet networkx oslo.config oslo.messaging
```

## Option 3: Install from Source

```bash
# Clone Ryu repository
git clone https://github.com/osrg/ryu.git
cd ryu

# Install dependencies
pip install -r tools/pip-requires

# Install Ryu
python setup.py install
```

## Running the Full Controller

After installing Ryu:

```bash
# Run the intelligent controller
python3 controller/intelligent_controller.py

# Or run with specific parameters
ryu-manager controller/intelligent_controller.py --ofp-tcp-listen-port 6653
```

## Demo Controller (No Dependencies)

If Ryu installation fails, you can use the demo controller:

```bash
# Run the demo (works without Ryu)
python3 demo_controller.py
```

## Troubleshooting

### Common Issues:

1. **Python Version Compatibility**: Ryu works best with Python 3.8-3.10
2. **Missing Dependencies**: Install eventlet, networkx, oslo.config
3. **Permission Issues**: Use virtual environment or sudo for system packages

### Verification:

```bash
# Check if Ryu is installed
python3 -c "import ryu; print('Ryu version:', ryu.__version__)"

# Check OpenFlow support
python3 -c "from ryu.ofproto import ofproto_v1_3; print('OpenFlow 1.3 supported')"
```

## Quick Start with Demo

```bash
# Activate virtual environment
source .venv/bin/activate

# Run demo controller (no Ryu needed)
python3 demo_controller.py

# Run evaluation
python3 evaluate_model.py

# Train new model
python3 ml_models/train_classifier.py
```

The demo controller provides the same core functionality as the full Ryu controller:
- ✅ Traffic classification using ML models
- ✅ Priority-based QoS assignment  
- ✅ Flow feature extraction
- ✅ Real-time processing simulation
- ✅ Performance statistics