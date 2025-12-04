#!/usr/bin/env python3
"""
Large Scale Model Accuracy Testing
Tests performance of large scale traffic classifier model
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Add project root to path
sys.path.append('/home/puneeth8055/Desktop/sdn-ai-traffic-classifier')

def load_large_scale_model():
    """Load large scale trained model"""
    try:
        model_path = '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/ml_models/traffic_classifier_large_scale.pkl'
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Large scale model loaded successfully")
        print(f"   Model type: {type(model_data.get('model', model_data)).__name__}")
        print(f"   Model name: {model_data.get('model_name', 'Unknown')}")
        print(f"   Training samples: {model_data.get('training_samples', 'Unknown')}")
        print(f"   Features: {model_data.get('n_features', 'Unknown')}")
        print(f"   Classes: {model_data.get('classes', 'Unknown')}")
        
        return model_data
    except Exception as e:
        print(f"❌ Error loading large scale model: {e}")
        return None

def load_test_data():
    """Load test datasets"""
    datasets = []
    
    # Try to load available datasets
    data_files = [
        '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/data/traffic_dataset.csv',
        '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/data/traffic_dataset_50.csv',
        '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/data/traffic_test_data.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Loaded dataset: {file_path} ({len(df)} samples)")
                datasets.append(df)
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
    
    if not datasets:
        print("⚠️ No test datasets found, generating synthetic data...")
        return generate_synthetic_test_data()
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"✅ Combined dataset: {len(combined_df)} samples")
    return combined_df

def generate_synthetic_test_data(n_samples=1000):
    """Generate synthetic test data for evaluation"""
    print(f"🔧 Generating {n_samples} synthetic test samples...")
    
    np.random.seed(42)
    
    # Traffic type patterns
    traffic_patterns = {
        'HTTP': {
            'duration': np.random.uniform(1, 10, n_samples//5),
            'packet_count': np.random.poisson(50, n_samples//5),
            'byte_count': np.random.normal(15000, 5000, n_samples//5),
            'avg_packet_size': np.random.normal(300, 100, n_samples//5),
            'packets_per_sec': np.random.uniform(5, 20, n_samples//5),
            'bytes_per_sec': np.random.uniform(1000, 5000, n_samples//5),
            'src_port': np.random.randint(1024, 65535, n_samples//5),
            'dst_port': np.full(n_samples//5, 80),
            'protocol_tcp': np.ones(n_samples//5),
            'protocol_udp': np.zeros(n_samples//5)
        },
        'Video': {
            'duration': np.random.uniform(10, 30, n_samples//5),
            'packet_count': np.random.poisson(300, n_samples//5),
            'byte_count': np.random.normal(150000, 30000, n_samples//5),
            'avg_packet_size': np.random.normal(500, 100, n_samples//5),
            'packets_per_sec': np.random.uniform(20, 50, n_samples//5),
            'bytes_per_sec': np.random.uniform(5000, 15000, n_samples//5),
            'src_port': np.random.randint(1024, 65535, n_samples//5),
            'dst_port': np.full(n_samples//5, 554),
            'protocol_tcp': np.ones(n_samples//5),
            'protocol_udp': np.zeros(n_samples//5)
        },
        'VoIP': {
            'duration': np.random.uniform(5, 15, n_samples//5),
            'packet_count': np.random.poisson(250, n_samples//5),
            'byte_count': np.random.normal(20000, 5000, n_samples//5),
            'avg_packet_size': np.random.normal(80, 20, n_samples//5),
            'packets_per_sec': np.random.uniform(40, 60, n_samples//5),
            'bytes_per_sec': np.random.uniform(1000, 2000, n_samples//5),
            'src_port': np.random.randint(1024, 65535, n_samples//5),
            'dst_port': np.full(n_samples//5, 5060),
            'protocol_tcp': np.zeros(n_samples//5),
            'protocol_udp': np.ones(n_samples//5)
        },
        'Gaming': {
            'duration': np.random.uniform(10, 20, n_samples//5),
            'packet_count': np.random.poisson(200, n_samples//5),
            'byte_count': np.random.normal(30000, 8000, n_samples//5),
            'avg_packet_size': np.random.normal(150, 50, n_samples//5),
            'packets_per_sec': np.random.uniform(15, 25, n_samples//5),
            'bytes_per_sec': np.random.uniform(2000, 4000, n_samples//5),
            'src_port': np.random.randint(1024, 65535, n_samples//5),
            'dst_port': np.random.randint(27000, 28000, n_samples//5),
            'protocol_tcp': np.zeros(n_samples//5),
            'protocol_udp': np.ones(n_samples//5)
        },
        'FTP': {
            'duration': np.random.uniform(5, 25, n_samples//5),
            'packet_count': np.random.poisson(100, n_samples//5),
            'byte_count': np.random.normal(150000, 40000, n_samples//5),
            'avg_packet_size': np.random.normal(1500, 200, n_samples//5),
            'packets_per_sec': np.random.uniform(4, 15, n_samples//5),
            'bytes_per_sec': np.random.uniform(6000, 20000, n_samples//5),
            'src_port': np.random.randint(1024, 65535, n_samples//5),
            'dst_port': np.full(n_samples//5, 21),
            'protocol_tcp': np.ones(n_samples//5),
            'protocol_udp': np.zeros(n_samples//5)
        }
    }
    
    # Combine all traffic types
    all_data = []
    all_labels = []
    
    for traffic_type, pattern in traffic_patterns.items():
        for i in range(len(pattern['duration'])):
            features = [
                pattern['duration'][i],
                pattern['packet_count'][i],
                pattern['byte_count'][i],
                pattern['avg_packet_size'][i],
                pattern['packets_per_sec'][i],
                pattern['bytes_per_sec'][i],
                np.random.uniform(0, 1),  # std_packet_size
                np.random.uniform(0.01, 0.1),  # avg_inter_arrival_time
                np.random.uniform(0.001, 0.01),  # std_inter_arrival_time
                pattern['src_port'][i],
                pattern['dst_port'][i],
                pattern['protocol_tcp'][i],
                pattern['protocol_udp'][i],
                np.random.randint(10, 1000),  # forward_packets
                np.random.randint(10, 1000),  # backward_packets
                np.random.uniform(100, 10000)  # flow_bytes_per_sec
            ]
            all_data.append(features)
            all_labels.append(traffic_type)
    
    # Create DataFrame
    feature_names = [
        'duration', 'packet_count', 'byte_count', 'avg_packet_size',
        'packets_per_sec', 'bytes_per_sec', 'std_packet_size',
        'avg_inter_arrival_time', 'std_inter_arrival_time', 'src_port',
        'dst_port', 'protocol_tcp', 'protocol_udp', 'forward_packets',
        'backward_packets', 'flow_bytes_per_sec'
    ]
    
    df = pd.DataFrame(all_data, columns=feature_names)
    df['traffic_type'] = all_labels
    
    return df

def test_model_accuracy(model_data, test_data):
    """Test model accuracy on provided data"""
    print("\n🧪 Testing Model Accuracy...")
    
    # Extract model components
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    label_encoder = model_data.get('label_encoder')
    
    if not all([model, scaler, label_encoder]):
        print("❌ Incomplete model data")
        return None
    
    # Prepare test data
    label_col = 'traffic_type' if 'traffic_type' in test_data.columns else 'true_label'
    if label_col not in test_data.columns:
        print("❌ No labels found in test data")
        return None
    
    X = test_data.drop(label_col, axis=1)
    y_true = test_data[label_col]
    
    # Remove any non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"   Features available: {list(X.columns)}")
    print(f"   Feature count: {X.shape[1]}")
    
    # Ensure feature alignment
    expected_features = scaler.n_features_in_
    if X.shape[1] != expected_features:
        print(f"⚠️ Feature mismatch: expected {expected_features}, got {X.shape[1]}")
        # Adjust features
        if X.shape[1] > expected_features:
            X = X.iloc[:, :expected_features]
        else:
            # Pad with zeros if needed
            for i in range(expected_features - X.shape[1]):
                X[f'pad_{i}'] = 0
    
    # Convert to numpy array
    X = X.values
    
    # Scale features
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        print(f"❌ Error scaling features: {e}")
        print(f"   X shape: {X.shape}")
        print(f"   X dtype: {X.dtype}")
        return None
    
    # Make predictions
    try:
        y_pred_encoded = model.predict(X_scaled)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)
        
        print(f"✅ Model testing completed")
        print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'labels': label_encoder.classes_
        }
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

def generate_accuracy_report(results, model_data):
    """Generate comprehensive accuracy report"""
    if not results:
        print("❌ No results to report")
        return
    
    print("\n" + "="*60)
    print("📊 LARGE SCALE MODEL ACCURACY REPORT")
    print("="*60)
    
    accuracy = results['accuracy']
    report = results['classification_report']
    cm = results['confusion_matrix']
    labels = results['labels']
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Model: {model_data.get('model_name', 'Unknown')}")
    print(f"   Training Samples: {model_data.get('training_samples', 'Unknown')}")
    
    print(f"\n📈 Per-Class Performance:")
    for class_name in labels:
        if class_name in report:
            metrics = report[class_name]
            print(f"   {class_name:8s}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    print(f"\n📊 Confusion Matrix:")
    print("   " + "  ".join(f"{label:8s}" for label in labels))
    for i, row in enumerate(cm):
        print(f"   {labels[i]:8s} " + "  ".join(f"{val:8d}" for val in row))
    
    # Save detailed report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'model_name': model_data.get('model_name', 'Unknown'),
            'model_type': type(model_data.get('model', model_data)).__name__,
            'training_samples': model_data.get('training_samples', 'Unknown'),
            'features': model_data.get('n_features', 'Unknown'),
            'classes': model_data.get('classes', 'Unknown')
        },
        'test_results': {
            'overall_accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': {}
        }
    }
    
    # Calculate per-class accuracy
    for i, label in enumerate(labels):
        if i < len(cm):
            tp = cm[i, i]
            total = cm[i, :].sum()
            report_data['test_results']['per_class_accuracy'][label] = tp / total if total > 0 else 0
    
    # Save report
    report_path = '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/results/large_scale_accuracy_report.json'
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    return report_data

def create_visualizations(results):
    """Create accuracy visualization plots"""
    if not results:
        return
    
    try:
        # Set up the plot
        plt.figure(figsize=(15, 10))
        
        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 2, 1)
        cm = results['confusion_matrix']
        labels = results['labels']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 2. Per-Class Performance
        plt.subplot(2, 2, 2)
        report = results['classification_report']
        classes = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_name in labels:
            if class_name in report:
                classes.append(class_name)
                precisions.append(report[class_name]['precision'])
                recalls.append(report[class_name]['recall'])
                f1_scores.append(report[class_name]['f1-score'])
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Traffic Type')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Accuracy Distribution
        plt.subplot(2, 2, 3)
        overall_acc = results['accuracy']
        per_class_acc = []
        
        for i, label in enumerate(labels):
            if i < len(cm):
                tp = cm[i, i]
                total = cm[i, :].sum()
                per_class_acc.append(tp / total if total > 0 else 0)
        
        bars = plt.bar(classes, per_class_acc, alpha=0.7, color='skyblue')
        plt.axhline(y=overall_acc, color='red', linestyle='--', 
                   label=f'Overall Accuracy: {overall_acc:.3f}')
        plt.xlabel('Traffic Type')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, per_class_acc):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 4. Prediction Distribution
        plt.subplot(2, 2, 4)
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Count predictions vs actual
        true_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()
        
        x = np.arange(len(labels))
        width = 0.35
        
        true_values = [true_counts.get(label, 0) for label in labels]
        pred_values = [pred_counts.get(label, 0) for label in labels]
        
        plt.bar(x - width/2, true_values, width, label='Actual', alpha=0.7)
        plt.bar(x + width/2, pred_values, width, label='Predicted', alpha=0.7)
        
        plt.xlabel('Traffic Type')
        plt.ylabel('Count')
        plt.title('Actual vs Predicted Distribution')
        plt.xticks(x, labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = '/home/puneeth8055/Desktop/sdn-ai-traffic-classifier/results/large_scale_accuracy_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

def main():
    """Main testing function"""
    print("="*60)
    print("🎯 LARGE SCALE MODEL ACCURACY TESTING")
    print("="*60)
    
    # Load model
    model_data = load_large_scale_model()
    if not model_data:
        print("❌ Failed to load model")
        return
    
    # Load test data
    test_data = load_test_data()
    if test_data is None or len(test_data) == 0:
        print("❌ No test data available")
        return
    
    # Test accuracy
    results = test_model_accuracy(model_data, test_data)
    if not results:
        print("❌ Accuracy testing failed")
        return
    
    # Generate report
    report = generate_accuracy_report(results, model_data)
    
    # Create visualizations
    create_visualizations(results)
    
    print("\n" + "="*60)
    print("🎉 LARGE SCALE ACCURACY TESTING COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()