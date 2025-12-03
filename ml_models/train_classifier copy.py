#!/usr/bin/env python3
"""
Machine Learning Model Training for Traffic Classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class TrafficClassifier:
    """Train and evaluate traffic classification models"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'duration',
            'packet_count',
            'byte_count',
            'avg_packet_size',
            'std_packet_size',
            'avg_inter_arrival_time',
            'std_inter_arrival_time',
            'src_port',
            'dst_port',
            'protocol_tcp',
            'protocol_udp',
            'forward_packets',
            'backward_packets',
            'flow_bytes_per_sec'
        ]
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data for different traffic types"""
        np.random.seed(42)
        data = []
        labels = []
        
        # HTTP Traffic
        for _ in range(n_samples // 5):
            data.append([
                np.random.uniform(1, 60),        # duration
                np.random.randint(10, 100),      # packet_count
                np.random.randint(5000, 50000),  # byte_count
                np.random.uniform(100, 1500),    # avg_packet_size
                np.random.uniform(100, 500),     # std_packet_size
                np.random.uniform(0.1, 0.5),     # avg_inter_arrival_time
                np.random.uniform(0.05, 0.2),    # std_inter_arrival_time
                np.random.randint(1024, 65535),  # src_port
                80,                               # dst_port (HTTP)
                1,                                # protocol_tcp
                0,                                # protocol_udp
                np.random.randint(5, 50),        # forward_packets
                np.random.randint(5, 50),        # backward_packets
                np.random.uniform(1000, 10000)   # flow_bytes_per_sec
            ])
            labels.append('HTTP')
        
        # Video Streaming
        for _ in range(n_samples // 5):
            data.append([
                np.random.uniform(60, 600),      # duration (longer)
                np.random.randint(500, 5000),    # packet_count (many)
                np.random.randint(500000, 5000000), # byte_count (large)
                np.random.uniform(1000, 1500),   # avg_packet_size (large)
                np.random.uniform(50, 200),      # std_packet_size
                np.random.uniform(0.03, 0.05),   # avg_inter_arrival_time (constant)
                np.random.uniform(0.01, 0.02),   # std_inter_arrival_time (low)
                np.random.randint(1024, 65535),  # src_port
                554,                              # dst_port (RTSP)
                0,                                # protocol_tcp
                1,                                # protocol_udp
                np.random.randint(400, 4500),    # forward_packets
                np.random.randint(50, 500),      # backward_packets
                np.random.uniform(50000, 500000) # flow_bytes_per_sec (high)
            ])
            labels.append('Video')
        
        # VoIP
        for _ in range(n_samples // 5):
            data.append([
                np.random.uniform(30, 300),      # duration
                np.random.randint(500, 3000),    # packet_count
                np.random.randint(50000, 300000), # byte_count
                np.random.uniform(50, 200),      # avg_packet_size (small)
                np.random.uniform(20, 50),       # std_packet_size
                np.random.uniform(0.02, 0.03),   # avg_inter_arrival_time (regular)
                np.random.uniform(0.001, 0.005), # std_inter_arrival_time (very low)
                np.random.randint(1024, 65535),  # src_port
                5060,                             # dst_port (SIP)
                0,                                # protocol_tcp
                1,                                # protocol_udp
                np.random.randint(400, 2500),    # forward_packets
                np.random.randint(400, 2500),    # backward_packets
                np.random.uniform(5000, 20000)   # flow_bytes_per_sec
            ])
            labels.append('VoIP')
        
        # File Transfer (FTP)
        for _ in range(n_samples // 5):
            data.append([
                np.random.uniform(10, 600),      # duration
                np.random.randint(100, 10000),   # packet_count
                np.random.randint(100000, 10000000), # byte_count (very large)
                np.random.uniform(1200, 1500),   # avg_packet_size (max)
                np.random.uniform(10, 100),      # std_packet_size
                np.random.uniform(0.001, 0.01),  # avg_inter_arrival_time (minimal)
                np.random.uniform(0.0001, 0.001), # std_inter_arrival_time
                np.random.randint(1024, 65535),  # src_port
                21,                               # dst_port (FTP)
                1,                                # protocol_tcp
                0,                                # protocol_udp
                np.random.randint(50, 9000),     # forward_packets
                np.random.randint(50, 1000),     # backward_packets
                np.random.uniform(100000, 1000000) # flow_bytes_per_sec (very high)
            ])
            labels.append('FTP')
        
        # Gaming
        for _ in range(n_samples // 5):
            data.append([
                np.random.uniform(60, 600),      # duration
                np.random.randint(1000, 10000),  # packet_count
                np.random.randint(50000, 500000), # byte_count
                np.random.uniform(50, 150),      # avg_packet_size (very small)
                np.random.uniform(20, 50),       # std_packet_size
                np.random.uniform(0.01, 0.05),   # avg_inter_arrival_time
                np.random.uniform(0.005, 0.02),  # std_inter_arrival_time
                np.random.randint(1024, 65535),  # src_port
                np.random.randint(27000, 28000), # dst_port (gaming)
                0,                                # protocol_tcp
                1,                                # protocol_udp
                np.random.randint(800, 9000),    # forward_packets
                np.random.randint(800, 9000),    # backward_packets
                np.random.uniform(10000, 100000) # flow_bytes_per_sec
            ])
            labels.append('Gaming')
        
        return np.array(data), np.array(labels)
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train classification model"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Training {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, self.label_encoder.classes_)
        
        # Feature importance (for tree-based models)
        if model_type in ['random_forest', 'decision_tree']:
            self._plot_feature_importance()
        
        return accuracy
    
    def _plot_confusion_matrix(self, cm, classes):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    def _plot_feature_importance(self):
        """Plot feature importance"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [self.feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved as 'feature_importance.png'")
    
    def save_model(self, filename='traffic_classifier.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModel saved as '{filename}'")
    
    def load_model(self, filename='traffic_classifier.pkl'):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from '{filename}'")

def main():
    """Main training function"""
    classifier = TrafficClassifier()
    
    # Generate synthetic data
    print("Generating synthetic training data...")
    X, y = classifier.generate_synthetic_data(n_samples=5000)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Train model
    classifier.train_model(X, y, model_type='random_forest')
    
    # Save model
    classifier.save_model('traffic_classifier.pkl')

if __name__ == '__main__':
    main()