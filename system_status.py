#!/usr/bin/env python3
"""
SDN AI Traffic Classifier - System Status and Demo
Shows the current system capabilities and status
"""

import os
import subprocess
import sys

def check_system_status():
    """Check the status of system components"""
    print("=" * 60)
    print("SDN AI Traffic Classifier - System Status")
    print("=" * 60)
    
    # Check project structure
    print("\n📁 Project Structure:")
    required_dirs = ['controller', 'ml_models', 'topology', 'utils']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
    
    # Check model files
    print("\n🤖 ML Model Status:")
    if os.path.exists('traffic_classifier.pkl'):
        print("  ✅ Trained model available (traffic_classifier.pkl)")
        print(f"  📊 Model size: {os.path.getsize('traffic_classifier.pkl')} bytes")
    else:
        print("  ❌ No trained model found")
    
    # Check Python files
    print("\n📄 Core Files:")
    core_files = {
        'controller/intelligent_controller.py': 'SDN Controller',
        'ml_models/train_classifier.py': 'ML Training Script',
        'topology/simple_topology.py': 'Network Topology',
        'utils/traffic_generator.py': 'Traffic Generator'
    }
    
    for file_path, description in core_files.items():
        if os.path.exists(file_path):
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description} (missing)")
    
    # Check visualization files
    print("\n📈 Visualization Files:")
    viz_files = ['confusion_matrix.png', 'feature_importance.png']
    for file_name in viz_files:
        if os.path.exists(file_name):
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} (run training to generate)")

def show_system_capabilities():
    """Display what the system can do"""
    print("\n" + "=" * 60)
    print("🚀 SYSTEM CAPABILITIES")
    print("=" * 60)
    
    capabilities = [
        {
            "title": "AI-Based Traffic Classification",
            "description": "Uses Random Forest ML model to classify network traffic into 5 types:",
            "features": ["HTTP (web browsing)", "Video (streaming)", "VoIP (voice calls)", "Gaming (online games)", "FTP (file transfers)"]
        },
        {
            "title": "Intelligent SDN Control",
            "description": "Ryu-based SDN controller that installs flow rules based on traffic classification:",
            "features": ["Priority-based routing", "QoS enforcement", "Dynamic flow management", "Real-time statistics"]
        },
        {
            "title": "Network Emulation",
            "description": "Mininet-based network topology for testing:",
            "features": ["6-host topology", "2 switches", "Programmable links", "Realistic traffic patterns"]
        },
        {
            "title": "Traffic Generation",
            "description": "Realistic traffic generators for different applications:",
            "features": ["HTTP requests/responses", "Video streaming packets", "VoIP call simulation", "Gaming traffic patterns", "FTP file transfers"]
        }
    ]
    
    for i, cap in enumerate(capabilities, 1):
        print(f"\n{i}. {cap['title']}")
        print(f"   {cap['description']}")
        for feature in cap['features']:
            print(f"   • {feature}")

def show_usage_instructions():
    """Show how to use the system"""
    print("\n" + "=" * 60)
    print("📖 USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("\n🎯 Quick Start (Demo Mode):")
    print("   python3 demo_system.py")
    print("   → Shows ML classification in action")
    
    print("\n🔧 Full System (requires sudo):")
    print("   Terminal 1: ./start.sh")
    print("   Terminal 2: sudo python3 topology/simple_topology.py")
    print("   Terminal 2 (mininet): pingall")
    print("   Terminal 2 (mininet): h1 python3 utils/traffic_generator.py 10.0.0.1 10.0.0.3 http &")
    
    print("\n📊 Available Scripts:")
    scripts = {
        "demo_system.py": "Run ML classification demo",
        "train_classifier.py": "Train/retrain ML model", 
        "run_system.sh": "Automated system startup",
        "start.sh": "Start SDN controller only",
        "setup.sh": "Install dependencies (requires sudo)"
    }
    
    for script, description in scripts.items():
        if os.path.exists(script):
            print(f"   ✅ {script:20s} - {description}")
        else:
            print(f"   ❌ {script:20s} - {description} (missing)")

def show_next_steps():
    """Show what can be done next"""
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS")
    print("=" * 60)
    
    next_steps = [
        "Install system dependencies (mininet, openvswitch) for full functionality",
        "Set up Kaggle API to download real-world traffic datasets",
        "Experiment with different ML algorithms (SVM, Neural Networks)",
        "Add new traffic types (email, P2P, social media)",
        "Implement load balancing across multiple paths",
        "Create web dashboard for real-time monitoring",
        "Test with hardware SDN switches",
        "Integrate with network monitoring tools"
    ]
    
    for i, step in enumerate(next_steps, 1):
        print(f"{i}. {step}")

def main():
    """Main status function"""
    # Change to project directory
    os.chdir('/home/puneeth8055/Desktop/sdn-ai-traffic-classifier')
    
    check_system_status()
    show_system_capabilities()
    show_usage_instructions()
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("🎉 SDN AI Traffic Classifier System Ready!")
    print("   Run 'python3 demo_system.py' to see it in action")
    print("=" * 60)

if __name__ == "__main__":
    main()