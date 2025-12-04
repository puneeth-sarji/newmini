#!/usr/bin/env python3
"""
Generate CSV files for trained models including training data and predictions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json

def generate_model_csvs():
    """Generate CSV files for model analysis"""
    
    print("=" * 60)
    print("GENERATING CSV FILES FOR MODEL ANALYSIS")
    print("=" * 60)
    
    # Load the trained model
    model_path = "ml_models/traffic_classifier_large_scale.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        classes = model_data['classes']
        
        # Load accuracy from training report if not in model file
        if 'accuracy' in model_data:
            accuracy = model_data['accuracy']
        else:
            # Try to load from training report
            try:
                with open('training_report_large_scale.json', 'r') as f:
                    report = json.load(f)
                accuracy = report.get('accuracy', 0.0)
            except:
                accuracy = 0.0
        
        print(f"✓ Loaded model: {model_data['model_name']}")
        print(f"✓ Accuracy: {accuracy:.4f}")
        print(f"✓ Classes: {classes}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate test dataset
    print("\nGenerating test dataset...")
    np.random.seed(123)  # Different seed for test data
    
    # Traffic profiles for test data
    test_profiles = {
        "HTTP": {
            "duration": lambda: np.random.lognormal(2.5, 1.5),
            "packet_count": lambda: np.random.poisson(100) + 20,
            "byte_count": lambda: np.random.exponential(50000) + 10000,
            "avg_packet_size": lambda: np.random.normal(600, 300),
            "std_packet_size": lambda: np.random.gamma(3, 80),
            "avg_inter_arrival_time": lambda: np.random.exponential(0.15),
            "std_inter_arrival_time": lambda: np.random.exponential(0.08),
            "dst_port": lambda: np.random.choice([80, 443, 8080, 8443, 554, 5060], p=[0.3, 0.3, 0.15, 0.15, 0.05, 0.05]),
            "protocol_tcp": 1,
            "protocol_udp": 0,
            "forward_packets": lambda: np.random.poisson(50) + 10,
            "backward_packets": lambda: np.random.poisson(50) + 10,
            "flow_bytes_per_sec": lambda: np.random.lognormal(8.5, 1.5),
        },
        "Video": {
            "duration": lambda: np.random.lognormal(3.8, 1.2),
            "packet_count": lambda: np.random.poisson(1500) + 300,
            "byte_count": lambda: np.random.exponential(1500000) + 300000,
            "avg_packet_size": lambda: np.random.normal(900, 400),
            "std_packet_size": lambda: np.random.gamma(4, 60),
            "avg_inter_arrival_time": lambda: np.random.exponential(0.03),
            "std_inter_arrival_time": lambda: np.random.exponential(0.015),
            "dst_port": lambda: np.random.choice([554, 1935, 8080, 443, 80, 5060], p=[0.25, 0.15, 0.25, 0.15, 0.1, 0.1]),
            "protocol_tcp": 0,
            "protocol_udp": 1,
            "forward_packets": lambda: np.random.poisson(1000) + 200,
            "backward_packets": lambda: np.random.poisson(500) + 100,
            "flow_bytes_per_sec": lambda: np.random.lognormal(9.5, 1.2),
        },
        "VoIP": {
            "duration": lambda: np.random.lognormal(3.2, 1.5),
            "packet_count": lambda: np.random.poisson(800) + 100,
            "byte_count": lambda: np.random.exponential(200000) + 30000,
            "avg_packet_size": lambda: np.random.normal(150, 100),
            "std_packet_size": lambda: np.random.gamma(3, 40),
            "avg_inter_arrival_time": lambda: np.random.exponential(0.025),
            "std_inter_arrival_time": lambda: np.random.exponential(0.012),
            "dst_port": lambda: np.random.choice([5060, 16384, 32768, 80, 443], p=[0.4, 0.2, 0.2, 0.1, 0.1]),
            "protocol_tcp": 0,
            "protocol_udp": 1,
            "forward_packets": lambda: np.random.poisson(400) + 80,
            "backward_packets": lambda: np.random.poisson(400) + 80,
            "flow_bytes_per_sec": lambda: np.random.lognormal(8.0, 1.3),
        },
        "FTP": {
            "duration": lambda: np.random.lognormal(3.5, 1.8),
            "packet_count": lambda: np.random.poisson(800) + 150,
            "byte_count": lambda: np.random.exponential(3000000) + 200000,
            "avg_packet_size": lambda: np.random.normal(1000, 500),
            "std_packet_size": lambda: np.random.gamma(3, 100),
            "avg_inter_arrival_time": lambda: np.random.exponential(0.008),
            "std_inter_arrival_time": lambda: np.random.exponential(0.004),
            "dst_port": lambda: np.random.choice([21, 22, 990, 80, 443], p=[0.4, 0.2, 0.2, 0.1, 0.1]),
            "protocol_tcp": 1,
            "protocol_udp": 0,
            "forward_packets": lambda: np.random.poisson(400) + 80,
            "backward_packets": lambda: np.random.poisson(400) + 80,
            "flow_bytes_per_sec": lambda: np.random.lognormal(8.8, 1.6),
        },
        "Gaming": {
            "duration": lambda: np.random.lognormal(3.8, 1.4),
            "packet_count": lambda: np.random.poisson(3000) + 500,
            "byte_count": lambda: np.random.exponential(800000) + 150000,
            "avg_packet_size": lambda: np.random.normal(200, 150),
            "std_packet_size": lambda: np.random.gamma(3, 50),
            "avg_inter_arrival_time": lambda: np.random.exponential(0.02),
            "std_inter_arrival_time": lambda: np.random.exponential(0.01),
            "dst_port": lambda: np.random.choice([27015, 27016, 7777, 8080, 443], p=[0.25, 0.15, 0.15, 0.25, 0.2]),
            "protocol_tcp": 0,
            "protocol_udp": 1,
            "forward_packets": lambda: np.random.poisson(1500) + 300,
            "backward_packets": lambda: np.random.poisson(1500) + 300,
            "flow_bytes_per_sec": lambda: np.random.lognormal(9.0, 1.4),
        },
        "P2P": {
            "duration": lambda: np.random.lognormal(4.0, 1.6),
            "packet_count": lambda: np.random.poisson(2000) + 300,
            "byte_count": lambda: np.random.exponential(2000000) + 300000,
            "avg_packet_size": lambda: np.random.normal(700, 400),
            "std_packet_size": lambda: np.random.gamma(4, 80),
            "avg_inter_arrival_time": lambda: np.random.exponential(0.04),
            "std_inter_arrival_time": lambda: np.random.exponential(0.02),
            "dst_port": lambda: np.random.choice([6881, 6882, 4444, 8999, 80, 443], p=[0.25, 0.25, 0.15, 0.15, 0.1, 0.1]),
            "protocol_tcp": 1,
            "protocol_udp": 0,
            "forward_packets": lambda: np.random.poisson(1000) + 200,
            "backward_packets": lambda: np.random.poisson(1000) + 200,
            "flow_bytes_per_sec": lambda: np.random.lognormal(8.7, 1.5),
        }
    }
    
    # Generate test data
    test_data = []
    test_labels = []
    samples_per_class = 500
    
    for traffic_type, profile in test_profiles.items():
        print(f"Generating {samples_per_class} {traffic_type} test samples...")
        
        for _ in range(samples_per_class):
            # Add noise
            duration_val = profile["duration"]()
            packet_count_val = profile["packet_count"]()
            byte_count_val = profile["byte_count"]()
            avg_packet_size_val = profile["avg_packet_size"]()
            
            duration = max(0.1, duration_val + np.random.normal(0, abs(duration_val) * 0.1))
            packet_count = max(1, int(packet_count_val + np.random.normal(0, abs(packet_count_val) * 0.05)))
            byte_count = max(1000, byte_count_val + np.random.normal(0, abs(byte_count_val) * 0.05))
            avg_packet_size = max(50, avg_packet_size_val + np.random.normal(0, abs(avg_packet_size_val) * 0.1))
            std_packet_size = max(1, profile["std_packet_size"]())
            avg_inter_arrival_time = max(0.001, profile["avg_inter_arrival_time"]())
            std_inter_arrival_time = max(0.0001, profile["std_inter_arrival_time"]())
            dst_port = profile["dst_port"]()
            forward_packets = max(1, profile["forward_packets"]())
            backward_packets = max(1, profile["backward_packets"]())
            flow_bytes_per_sec = max(1, profile["flow_bytes_per_sec"]())
            
            test_data.append([
                duration,
                packet_count,
                byte_count,
                avg_packet_size,
                std_packet_size,
                avg_inter_arrival_time,
                std_inter_arrival_time,
                np.random.randint(1024, 65535),  # src_port
                dst_port,
                profile["protocol_tcp"],
                profile["protocol_udp"],
                forward_packets,
                backward_packets,
                flow_bytes_per_sec,
            ])
            test_labels.append(traffic_type)
    
    # Convert to numpy arrays
    X_test = np.array(test_data)
    y_test = np.array(test_labels)
    
    print(f"\nGenerated test dataset: {X_test.shape}")
    
    # Make predictions
    print("Making predictions...")
    X_test_scaled = scaler.transform(X_test)
    y_pred_encoded = model.predict(X_test_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Create DataFrames
    
    # 1. Raw test data with true labels
    df_test_data = pd.DataFrame(X_test, columns=feature_names)
    df_test_data['true_label'] = y_test
    df_test_data['src_port'] = df_test_data['src_port'].astype(int)
    df_test_data['dst_port'] = df_test_data['dst_port'].astype(int)
    
    # 2. Predictions with probabilities
    df_predictions = pd.DataFrame({
        'sample_id': range(len(y_pred)),
        'true_label': y_test,
        'predicted_label': y_pred,
        'is_correct': y_test == y_pred
    })
    
    # Add probability columns
    for i, class_name in enumerate(classes):
        df_predictions[f'prob_{class_name}'] = y_pred_proba[:, i]
    
    # 3. Model performance summary
    performance_summary = {
        'model_name': [model_data['model_name']],
        'accuracy': [accuracy],
        'total_samples': [len(y_test)],
        'correct_predictions': [sum(y_test == y_pred)],
        'incorrect_predictions': [sum(y_test != y_pred)],
        'num_classes': [len(classes)],
        'num_features': [len(feature_names)],
        'training_timestamp': [model_data.get('training_timestamp', 'N/A')]
    }
    df_performance = pd.DataFrame(performance_summary)
    
    # 4. Per-class performance
    per_class_stats = []
    for class_name in classes:
        class_mask = y_test == class_name
        class_true = y_test[class_mask]
        class_pred = y_pred[class_mask]
        
        tp = sum((class_true == class_name) & (class_pred == class_name))
        fp = sum((class_true != class_name) & (class_pred == class_name))
        fn = sum((class_true == class_name) & (class_pred != class_name))
        tn = sum((class_true != class_name) & (class_pred != class_name))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_stats.append({
            'class': class_name,
            'samples': len(class_true),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': tp / len(class_true) if len(class_true) > 0 else 0
        })
    
    df_per_class = pd.DataFrame(per_class_stats)
    
    # 5. Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = []
        for i, importance in enumerate(model.feature_importances_):
            feature_importance.append({
                'feature': feature_names[i],
                'importance': importance,
                'rank': i + 1
            })
        
        df_feature_importance = pd.DataFrame(feature_importance)
        df_feature_importance = df_feature_importance.sort_values('importance', ascending=False)
    else:
        df_feature_importance = pd.DataFrame({'feature': [], 'importance': [], 'rank': []})
    
    # Save all CSV files
    output_files = {
        'traffic_test_data.csv': df_test_data,
        'model_predictions.csv': df_predictions,
        'model_performance_summary.csv': df_performance,
        'per_class_performance.csv': df_per_class,
        'feature_importance.csv': df_feature_importance
    }
    
    print("\nSaving CSV files...")
    for filename, dataframe in output_files.items():
        dataframe.to_csv(filename, index=False)
        print(f"✓ Saved {filename} ({len(dataframe)} rows)")
    
    # Save metadata
    metadata = {
        'generated_files': list(output_files.keys()),
        'model_info': {
            'name': model_data['model_name'],
            'accuracy': accuracy,
            'classes': classes,
            'features': feature_names
        },
        'test_dataset_info': {
            'total_samples': len(y_test),
            'samples_per_class': samples_per_class,
            'feature_count': len(feature_names)
        },
        'generation_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open('model_csv_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata to 'model_csv_metadata.json'")
    
    print("\n" + "=" * 60)
    print("CSV GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Generated {len(output_files)} CSV files:")
    for filename in output_files.keys():
        print(f"  - {filename}")
    print("\nFiles ready for analysis and visualization!")
    print("=" * 60)

if __name__ == "__main__":
    generate_model_csvs()