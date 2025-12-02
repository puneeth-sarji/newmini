====================================================
AI-BASED SDN TRAFFIC CLASSIFICATION SYSTEM
====================================================

QUICK START GUIDE
=================

1. SETUP (First Time Only)
   cd ~/sdn-ai-traffic-classifier
   sudo ./setup.sh
   
   Wait 5-10 minutes for installation.

2. TRAIN MODEL (First Time Only)
   python3 ml_models/train_classifier.py
   
   Wait 1-2 minutes for training.

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

6. STOP SYSTEM
   In Terminal 2: mininet> exit
   In Terminal 1: Press Ctrl+C
   sudo mn -c

====================================================
DIRECTORY STRUCTURE
====================================================

sdn-ai-traffic-classifier/
├── controller/
│   └── intelligent_controller.py
├── ml_models/
│   ├── train_classifier.py
│   ├── train_classifier_real.py
│   ├── dataset_processor.py
│   └── dataset_downloader.py
├── topology/
│   └── simple_topology.py
├── utils/
│   └── traffic_generator.py
├── traffic_data/
├── logs/
├── setup.sh
├── setup_kaggle.sh
├── start.sh
├── test_traffic.sh
├── run_system.sh
└── README.md (this file)

====================================================
COMMON COMMANDS
====================================================

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

====================================================
TROUBLESHOOTING
====================================================

Problem: Port 6653 already in use
Solution: pkill -f ryu-manager

Problem: Mininet won't start
Solution: sudo mn -c && sudo service openvswitch-switch restart

Problem: Module not found
Solution: pip3 install [module-name]

Problem: Permission denied
Solution: chmod +x setup.sh (or use sudo)

====================================================
FOR MORE HELP
====================================================

Check the controller terminal for classification logs.
Statistics update every 10 seconds.
See generated .png files for model performance.

====================================================
