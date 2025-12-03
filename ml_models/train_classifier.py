#!/usr/bin/env python3
"""
Machine Learning Model Training for Traffic Classification
Python 3.12 Compatible Version
"""

import pickle

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.use("Agg")  # Use non-interactive backend
import sys
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")


class TrafficClassifier:
    """Train and evaluate traffic classification models"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            "duration",
            "packet_count",
            "byte_count",
            "avg_packet_size",
            "std_packet_size",
            "avg_inter_arrival_time",
            "std_inter_arrival_time",
            "src_port",
            "dst_port",
            "protocol_tcp",
            "protocol_udp",
            "forward_packets",
            "backward_packets",
            "flow_bytes_per_sec",
        ]

    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data for different traffic types"""
        print("\n" + "=" * 60)
        print("Generating Synthetic Training Data")
        print("=" * 60)

        np.random.seed(42)
        data = []
        labels = []

        # HTTP Traffic
        print("Generating HTTP traffic samples...")
        for _ in range(n_samples // 5):
            data.append(
                [
                    np.random.uniform(1, 60),  # duration
                    np.random.randint(10, 100),  # packet_count
                    np.random.randint(5000, 50000),  # byte_count
                    np.random.uniform(100, 1500),  # avg_packet_size
                    np.random.uniform(100, 500),  # std_packet_size
                    np.random.uniform(0.1, 0.5),  # avg_inter_arrival_time
                    np.random.uniform(0.05, 0.2),  # std_inter_arrival_time
                    np.random.randint(1024, 65535),  # src_port
                    80,  # dst_port (HTTP)
                    1,  # protocol_tcp
                    0,  # protocol_udp
                    np.random.randint(5, 50),  # forward_packets
                    np.random.randint(5, 50),  # backward_packets
                    np.random.uniform(1000, 10000),  # flow_bytes_per_sec
                ]
            )
            labels.append("HTTP")

        # Video Streaming
        print("Generating Video traffic samples...")
        for _ in range(n_samples // 5):
            data.append(
                [
                    np.random.uniform(60, 600),
                    np.random.randint(500, 5000),
                    np.random.randint(500000, 5000000),
                    np.random.uniform(1000, 1500),
                    np.random.uniform(50, 200),
                    np.random.uniform(0.03, 0.05),
                    np.random.uniform(0.01, 0.02),
                    np.random.randint(1024, 65535),
                    554,
                    0,
                    1,
                    np.random.randint(400, 4500),
                    np.random.randint(50, 500),
                    np.random.uniform(50000, 500000),
                ]
            )
            labels.append("Video")

        # VoIP
        print("Generating VoIP traffic samples...")
        for _ in range(n_samples // 5):
            data.append(
                [
                    np.random.uniform(30, 300),
                    np.random.randint(500, 3000),
                    np.random.randint(50000, 300000),
                    np.random.uniform(50, 200),
                    np.random.uniform(20, 50),
                    np.random.uniform(0.02, 0.03),
                    np.random.uniform(0.001, 0.005),
                    np.random.randint(1024, 65535),
                    5060,
                    0,
                    1,
                    np.random.randint(400, 2500),
                    np.random.randint(400, 2500),
                    np.random.uniform(5000, 20000),
                ]
            )
            labels.append("VoIP")

        # File Transfer (FTP)
        print("Generating FTP traffic samples...")
        for _ in range(n_samples // 5):
            data.append(
                [
                    np.random.uniform(10, 600),
                    np.random.randint(100, 10000),
                    np.random.randint(100000, 10000000),
                    np.random.uniform(1200, 1500),
                    np.random.uniform(10, 100),
                    np.random.uniform(0.001, 0.01),
                    np.random.uniform(0.0001, 0.001),
                    np.random.randint(1024, 65535),
                    21,
                    1,
                    0,
                    np.random.randint(50, 9000),
                    np.random.randint(50, 1000),
                    np.random.uniform(100000, 1000000),
                ]
            )
            labels.append("FTP")

        # Gaming
        print("Generating Gaming traffic samples...")
        for _ in range(n_samples // 5):
            data.append(
                [
                    np.random.uniform(60, 600),
                    np.random.randint(1000, 10000),
                    np.random.randint(50000, 500000),
                    np.random.uniform(50, 150),
                    np.random.uniform(20, 50),
                    np.random.uniform(0.01, 0.05),
                    np.random.uniform(0.005, 0.02),
                    np.random.randint(1024, 65535),
                    np.random.randint(27000, 28000),
                    0,
                    1,
                    np.random.randint(800, 9000),
                    np.random.randint(800, 9000),
                    np.random.uniform(10000, 100000),
                ]
            )
            labels.append("Gaming")

        print(f"\n✓ Generated {len(data)} samples across 5 traffic types")
        return np.array(data), np.array(labels)

    def train_model(self, X, y, model_type="random_forest"):
        """Train classification model"""
        print("\n" + "=" * 60)
        print("Training Model")
        print("=" * 60)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print(f"\nTraining {model_type} model...")
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{'=' * 60}")
        print(f"Model Performance")
        print(f"{'=' * 60}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=self.label_encoder.classes_,
                zero_division=0,
            )
        )

        # Cross-validation
        print("Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, self.label_encoder.classes_)

        # Feature importance
        self._plot_feature_importance()

        return accuracy

    def _plot_confusion_matrix(self, cm, classes):
        """Plot confusion matrix"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
        except Exception as e:
            print(f"\n⚠ Could not create confusion matrix plot: {e}")

    def _plot_feature_importance(self):
        """Plot feature importance"""
        try:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 6))
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(
                range(len(importances)),
                [self.feature_names[i] for i in indices],
                rotation=45,
                ha="right",
            )
            plt.tight_layout()
            plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("✓ Feature importance plot saved as 'feature_importance.png'")
        except Exception as e:
            print(f"⚠ Could not create feature importance plot: {e}")

    def save_model(self, filename="traffic_classifier.pkl"):
        """Save trained model"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "model_name": "Random Forest",
        }
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Model saved as '{filename}'")


def main():
    """Main training function"""
    print("\n" + "=" * 60)
    print("AI-Based Traffic Classifier Training")
    print("Python 3.12 Compatible Version")
    print("=" * 60)

    classifier = TrafficClassifier()

    # Generate synthetic data
    X, y = classifier.generate_synthetic_data(n_samples=5000)

    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Classes: {np.unique(y)}")

    # Train model
    accuracy = classifier.train_model(X, y, model_type="random_forest")

    # Save model
    classifier.save_model("ml_models/traffic_classifier.pkl")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nModel ready for deployment!")
    print("Next step: Start the controller and network")
    print("=" * 60)


if __name__ == "__main__":
    main()
