====================================================
AI-BASED SDN TRAFFIC CLASSIFICATION SYSTEM
====================================================

A comprehensive Software-Defined Networking (SDN) system that uses Machine Learning 
to classify network traffic in real-time. The system integrates Ryu SDN controller 
with multiple trained ML models for intelligent traffic management.

🚀 **FEATURES**
- Real-time traffic classification using ML models
- Multiple pre-trained classifiers (Basic, Large Scale, Real Traffic)
- Comprehensive testing and evaluation framework
- Performance visualization and analytics
- Support for 6 traffic classes: HTTP, HTTPS, FTP, SSH, Video, VoIP

📊 **MODEL PERFORMANCE**
- Large Scale Model: 99.83% accuracy (6 classes, 14 features)
- Real Traffic Model: Optimized for production environments
- Comprehensive evaluation with confusion matrices and feature analysis

🔧 **QUICK START GUIDE**
========================

1. SETUP (First Time Only)
   cd ~/sdn-ai-traffic-classifier
   sudo ./setup.sh
   
   Wait 5-10 minutes for installation.

2. TRAIN MODELS (First Time Only)
   # Basic model
   python3 ml_models/train_classifier.py
   
   # Large scale model (recommended)
   python3 ml_models/train_large_scale.py
   
   Wait 1-3 minutes for training.

3. START CONTROLLER (Terminal 1)
   ./start.sh
   
   Leave this running.

4. START NETWORK (Terminal 2)
   sudo python3 topology/simple_topology.py
   
   Wait for mininet> prompt.

5. TEST SYSTEM (In Terminal 2 - Mininet CLI)
   mininet> pingall
   mininet> h1 python3 utils/traffic_generator.py 10.0.0.1 10.0.0.3 http &
   mininet> h2 python3 utils/traffic_generator.py 10.0.0.2 10.0.0.4 video &
   
   Watch Terminal 1 for classification results!

6. ACCURACY TESTING (Optional)
   python3 test_large_scale_accuracy_fixed.py
   
   Tests model accuracy with comprehensive evaluation.

7. STOP SYSTEM
   In Terminal 2: mininet> exit
   In Terminal 1: Press Ctrl+C
   sudo mn -c

====================================================
DIRECTORY STRUCTURE
====================================================

sdn-ai-traffic-classifier/
├── 📁 controller/
│   └── intelligent_controller.py      # Main SDN controller with ML integration
├── 📁 ml_models/
│   ├── train_classifier.py            # Basic model training
│   ├── train_large_scale.py           # Large scale model training
│   ├── train_classifier_real.py       # Real traffic model training
│   ├── dataset_processor.py           # Data preprocessing utilities
│   ├── traffic_classifier.pkl          # Basic trained model
│   ├── traffic_classifier_large_scale.pkl  # Large scale model (99.83% acc)
│   └── traffic_classifier_real.pkl    # Real traffic optimized model
├── 📁 data/
│   ├── traffic_dataset.csv             # Full traffic dataset
│   ├── traffic_dataset_50.csv          # Sample dataset (50 records)
│   └── traffic_test_data.csv          # Test dataset for evaluation
├── 📁 results/
│   ├── *.png                          # Performance visualizations
│   ├── *.json                         # Evaluation reports
│   └── *.csv                          # Performance metrics
├── 📁 topology/
│   └── simple_topology.py              # Mininet network topology
├── 📁 utils/
│   └── traffic_generator.py            # Traffic generation utility
├── 📁 docs/
│   ├── PROJECT_SUMMARY.md              # Detailed project overview
│   ├── ARCHITECTURE.md                # System architecture
│   ├── EXECUTION_GUIDE.md             # Step-by-step execution guide
│   └── RYU_INSTALLATION.md            # Ryu installation instructions
├── 📁 scripts/
│   ├── demo_system.py                  # System demonstration
│   ├── evaluate_model.py               # Model evaluation utilities
│   └── various demo scripts
├── 🐍 test_large_scale_accuracy_fixed.py  # Comprehensive accuracy testing
├── 🔧 setup.sh                         # Environment setup script
├── 🔧 setup_kaggle.sh                  # Kaggle dataset setup
├── 🚀 start.sh                        # Controller startup script
├── 🧪 test_traffic.sh                 # Traffic testing script
└── 📖 README.md (this file)

====================================================
COMMON COMMANDS
====================================================

🎯 **SYSTEM OPERATIONS**
Start Controller:
  ryu-manager controller/intelligent_controller.py --verbose

Start Network:
  sudo python3 topology/simple_topology.py

Check Flows:
  sudo ovs-ofctl dump-flows s1 -O OpenFlow13

Clean Up:
  sudo mn -c

Kill Controller:
  pkill -f ryu-manager

🤖 **MODEL OPERATIONS**
Train Basic Model:
  python3 ml_models/train_classifier.py

Train Large Scale Model:
  python3 ml_models/train_large_scale.py

Test Model Accuracy:
  python3 test_large_scale_accuracy_fixed.py

Evaluate Model Performance:
  python3 scripts/evaluate_model.py

📊 **VISUALIZATION & RESULTS**
View Performance Graphs:
  ls results/*.png

View Evaluation Reports:
  cat results/evaluation_report.json

Feature Importance Analysis:
  cat results/feature_importance.csv

====================================================
TROUBLESHOOTING
====================================================

🔧 **COMMON ISSUES**
Problem: Port 6653 already in use
Solution: pkill -f ryu-manager

Problem: Mininet won't start
Solution: sudo mn -c && sudo service openvswitch-switch restart

Problem: Module not found
Solution: pip3 install [module-name] or source .venv/bin/activate

Problem: Permission denied
Solution: chmod +x setup.sh (or use sudo)

🤖 **MODEL ISSUES**
Problem: Model loading failed
Solution: Check if .pkl files exist in ml_models/ directory

Problem: Feature mismatch error
Solution: Ensure test data has correct feature count (14 for large scale model)

Problem: Low accuracy results
Solution: Retrain model with current dataset using train_large_scale.py

📊 **PERFORMANCE ISSUES**
Problem: Slow classification
Solution: Use traffic_classifier_real.pkl for production environments

Problem: Memory issues
Solution: Use basic model (traffic_classifier.pkl) for resource-constrained systems

====================================================
DOCUMENTATION & SUPPORT
====================================================

📖 **DETAILED GUIDES**
- Project Overview: docs/PROJECT_SUMMARY.md
- System Architecture: docs/ARCHITECTURE.md  
- Step-by-Step Guide: docs/EXECUTION_GUIDE.md
- ML Model Guide: docs/ML_MODEL_GUIDE.md
- Ryu Installation: docs/RYU_INSTALLATION.md

📊 **MONITORING**
- Check controller terminal for classification logs
- Statistics update every 10 seconds
- View performance graphs in results/ directory
- Check evaluation reports for detailed metrics

🚀 **PRODUCTION DEPLOYMENT**
- Use results/deploy_production.sh for deployment
- Monitor system with scripts/system_status.py
- Scale with ml_models/traffic_classifier_real.pkl

====================================================
