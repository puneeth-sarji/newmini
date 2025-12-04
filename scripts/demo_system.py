#!/usr/bin/env python3
"""
Demo script to showcase the SDN AI Traffic Classifier functionality
This simulates the system behavior without requiring sudo/mininet
"""

import sys
import os
import time
import pickle
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.append('/home/puneeth8055/Desktop/sdn-ai-traffic-classifier')

def load_model():
    """Load the trained ML model"""
    model_paths = [
        '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/ml_models/traffic_classifier_real.pkl',
        '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/ml_models/traffic_classifier.pkl',
        '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/ml_models/traffic_classifier_large_scale.pkl'
    ]
    
    for model_path in model_paths:
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"✅ Loaded model from: {model_path}")
            return model_data
        except Exception as e:
            print(f"Could not load {model_path}: {e}")
            continue
    
    print("❌ No model could be loaded")
    return None

def simulate_traffic_features():
    """Generate simulated traffic features for different traffic types"""
    traffic_types = {
        'HTTP': {'packet_count': 150, 'byte_count': 45000, 'duration': 5.0, 'avg_packet_size': 300},
        'Video': {'packet_count': 500, 'byte_count': 250000, 'duration': 5.0, 'avg_packet_size': 500},
        'VoIP': {'packet_count': 100, 'byte_count': 8000, 'duration': 5.0, 'avg_packet_size': 80},
        'Gaming': {'packet_count': 200, 'byte_count': 30000, 'duration': 5.0, 'avg_packet_size': 150},
        'FTP': {'packet_count': 50, 'byte_count': 50000, 'duration': 5.0, 'avg_packet_size': 1000}
    }
    
    features = []
    labels = []
    
    for traffic_type, params in traffic_types.items():
        # Generate multiple samples per traffic type
        for _ in range(10):
            # Add some randomness to make it realistic
            packet_count = int(params['packet_count'] * np.random.uniform(0.8, 1.2))
            byte_count = int(params['byte_count'] * np.random.uniform(0.8, 1.2))
            duration = params['duration'] * np.random.uniform(0.9, 1.1)
            avg_packet_size = byte_count / packet_count if packet_count > 0 else 0
            
            # Calculate additional features
            packets_per_second = packet_count / duration
            bytes_per_second = byte_count / duration
            
            # Create feature vector (matching the training data format)
            feature_vector = [
                packet_count,
                byte_count,
                duration,
                avg_packet_size,
                packets_per_second,
                bytes_per_second,
                np.random.uniform(0, 1),  # TCP/UDP ratio
                np.random.uniform(0, 1),  # Source port variation
                np.random.uniform(0, 1),  # Dest port variation
                np.random.uniform(0, 1),  # Protocol entropy
                np.random.uniform(0, 1),  # Flow symmetry
                np.random.uniform(0, 1),  # Packet size variance
                np.random.uniform(0, 1),  # Inter-arrival time variance
                np.random.uniform(0, 1)   # Burstiness
            ]
            
            features.append(feature_vector)
            labels.append(traffic_type)
    
    return np.array(features), labels

def main():
    print("=" * 60)
    print("SDN AI Traffic Classifier - Demo Mode")
    print("=" * 60)
    
    # Load the trained model
    print("\n[1/4] Loading trained ML model...")
    model_data = load_model()
    if model_data is None:
        print("❌ Failed to load model. Please train the model first.")
        return
    
    classifier = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    
    print("✅ Model loaded successfully!")
    print(f"   Model type: {type(classifier).__name__}")
    
    # Generate simulated traffic
    print("\n[2/4] Generating simulated traffic flows...")
    features, true_labels = simulate_traffic_features()
    print(f"✅ Generated {len(features)} traffic samples for classification")
    
    # Scale features using the same scaler from training
    features_scaled = scaler.transform(features)
    
    # Classify traffic
    print("\n[3/4] Classifying traffic flows...")
    predicted_indices = classifier.predict(features_scaled)
    predicted_proba = classifier.predict_proba(features_scaled)
    predicted_labels = label_encoder.inverse_transform(predicted_indices)
    
    # Display results
    print("\n[4/4] Classification Results:")
    print("-" * 60)
    
    traffic_priority = {
        'VoIP': 3,      # Highest priority
        'Gaming': 3,
        'Video': 2,
        'HTTP': 1,
        'FTP': 1        # Lowest priority
    }
    
    correct_predictions = 0
    for i in range(len(features)):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        confidence = np.max(predicted_proba[i]) * 100
        priority = traffic_priority.get(predicted_label, 1)
        
        is_correct = true_label == predicted_label
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} Flow {i+1:2d}: {true_label:6s} → {predicted_label:6s} | "
              f"Priority: {priority} | Confidence: {confidence:5.1f}%")
    
    # Summary statistics
    accuracy = (correct_predictions / len(features)) * 100
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total flows processed: {len(features)}")
    print(f"Correct classifications: {correct_predictions}")
    print(f"Classification accuracy: {accuracy:.1f}%")
    
    # Traffic type distribution
    print("\nTraffic Type Distribution:")
    traffic_counts = defaultdict(int)
    for label in predicted_labels:
        traffic_counts[label] += 1
    
    for traffic_type, count in sorted(traffic_counts.items()):
        percentage = (count / len(predicted_labels)) * 100
        priority = traffic_priority.get(traffic_type, 1)
        priority_str = "High" if priority >= 3 else "Medium" if priority == 2 else "Low"
        print(f"  {traffic_type:6s}: {count:2d} flows ({percentage:5.1f}%) | Priority: {priority} ({priority_str})")
    
    print("\n" + "=" * 60)
    print("🎯 SDN AI Traffic Classifier Demo Complete!")
    print("   In a real deployment, these classifications would be used")
    print("   to install OpenFlow rules with appropriate priorities.")
    print("=" * 60)

if __name__ == "__main__":
    main()