#!/usr/bin/env python3
"""
RL-Based Traffic Classification on Large Scale Data
Uses Q-learning for multi-class traffic classification
"""

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time


class RLClassifier:
    def __init__(self, n_classes=7, n_features=14):
        self.n_classes = n_classes
        self.n_features = n_features

        # RL parameters
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = 0.01  # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1 # Exploration rate

        # Feature discretization
        self.feature_bins = 10
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        print("🤖 RL Classifier Initialized")
        print(f"   Classes: {n_classes}")
        print(f"   Features: {n_features}")
        print(f"   Feature bins: {self.feature_bins}")

    def discretize_features(self, features):
        """Discretize continuous features for Q-table states"""
        discretized = []
        for i, feature in enumerate(features):
            # Scale feature
            scaled = (feature - self.feature_means[i]) / (self.feature_stds[i] + 1e-8)
            # Discretize
            bin_idx = min(self.feature_bins - 1, max(0, int((scaled + 3) / 6 * self.feature_bins)))
            discretized.append(bin_idx)
        return tuple(discretized)

    def fit(self, X, y, episodes=1000):
        """Train RL classifier"""
        print(f"\n🎯 Training RL Classifier for {episodes} episodes...")

        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)

        # Store feature stats for discretization
        self.feature_means = np.mean(X_scaled, axis=0)
        self.feature_stds = np.std(X_scaled, axis=0)

        # Training loop
        for episode in range(episodes):
            # Sample random instance
            idx = np.random.randint(len(X_scaled))
            features = X_scaled[idx]
            true_label = y_encoded[idx]

            # Discretize state
            state = self.discretize_features(features)

            # Choose action (predicted class)
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.n_classes)
            else:
                action = max(range(self.n_classes), key=lambda a: self.q_table[state][a])

            # Calculate reward
            reward = 10 if action == true_label else -1

            # Update Q-table
            next_state = state  # Classification is single-step
            best_next = max(self.q_table[next_state].values(), default=0)
            self.q_table[state][action] += self.alpha * (reward + self.gamma * best_next - self.q_table[state][action])

            if episode % 100 == 0:
                print(f"   Episode {episode}: Q-table size {len(self.q_table)}")

        print(f"✅ Training complete! Q-table size: {len(self.q_table)}")

    def predict(self, X):
        """Predict using trained RL policy"""
        X_scaled = self.scaler.transform(X)
        predictions = []

        for features in X_scaled:
            state = self.discretize_features(features)
            # Greedy policy
            action = max(range(self.n_classes), key=lambda a: self.q_table[state][a])
            predictions.append(action)

        return self.label_encoder.inverse_transform(predictions)

    def evaluate(self, X_test, y_test):
        """Evaluate RL classifier"""
        print("\n📊 Evaluating RL Classifier...")

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return accuracy


def main():
    print("🚀 RL-Based Traffic Classification on Large Scale Data")
    print("=" * 60)

    # Load large scale data (simulate from training data)
    # Since we don't have the raw data, we'll generate synthetic large scale data
    print("📥 Generating large scale synthetic traffic data...")

    # Traffic profiles (same as training)
    traffic_profiles = {
        "HTTP": {"weight": 0.2, "n_samples": 20000},
        "Video": {"weight": 0.18, "n_samples": 18000},
        "VoIP": {"weight": 0.16, "n_samples": 16000},
        "FTP": {"weight": 0.16, "n_samples": 16000},
        "Gaming": {"weight": 0.18, "n_samples": 18000},
        "P2P": {"weight": 0.12, "n_samples": 12000},
    }

    np.random.seed(42)
    data = []
    labels = []

    for traffic_type, profile in traffic_profiles.items():
        n_samples = profile["n_samples"]
        print(f"   Generating {traffic_type}: {n_samples:,} samples")

        for _ in range(n_samples):
            # Generate features based on traffic type
            if traffic_type == "HTTP":
                features = [
                    np.random.uniform(1, 10),    # duration
                    np.random.randint(100, 1000), # packet_count
                    np.random.randint(1000, 10000), # byte_count
                    np.random.uniform(500, 1500), # avg_packet_size
                    np.random.uniform(100, 500),  # std_packet_size
                    np.random.uniform(0.1, 1.0),  # avg_inter_arrival
                    np.random.uniform(0.05, 0.5), # std_inter_arrival
                    80,  # dst_port
                    1,   # protocol_tcp
                    0,   # protocol_udp
                    np.random.randint(50, 200),   # forward_packets
                    np.random.randint(50, 200),   # backward_packets
                    np.random.uniform(1000, 5000), # flow_bytes_per_sec
                ]
            elif traffic_type == "Video":
                features = [
                    np.random.uniform(60, 7200),
                    np.random.randint(1000, 10000),
                    np.random.randint(100000, 1000000),
                    np.random.uniform(1000, 2000),
                    np.random.uniform(200, 800),
                    np.random.uniform(0.01, 0.1),
                    np.random.uniform(0.005, 0.05),
                    554,
                    1, 0,
                    np.random.randint(500, 2000),
                    np.random.randint(500, 2000),
                    np.random.uniform(50000, 200000),
                ]
            elif traffic_type == "VoIP":
                features = [
                    np.random.uniform(60, 3600),
                    np.random.randint(1000, 5000),
                    np.random.randint(50000, 200000),
                    np.random.uniform(100, 300),
                    np.random.uniform(20, 100),
                    np.random.uniform(0.02, 0.05),
                    np.random.uniform(0.01, 0.02),
                    5060,
                    0, 1,
                    np.random.randint(500, 1000),
                    np.random.randint(500, 1000),
                    np.random.uniform(5000, 20000),
                ]
            elif traffic_type == "FTP":
                features = [
                    np.random.uniform(300, 7200),
                    np.random.randint(500, 5000),
                    np.random.randint(500000, 50000000),
                    np.random.uniform(1200, 2000),
                    np.random.uniform(300, 1000),
                    np.random.uniform(0.1, 2.0),
                    np.random.uniform(0.05, 1.0),
                    21,
                    1, 0,
                    np.random.randint(250, 2500),
                    np.random.randint(250, 2500),
                    np.random.uniform(50000, 500000),
                ]
            elif traffic_type == "Gaming":
                features = [
                    np.random.uniform(60, 3600),
                    np.random.randint(500, 2000),
                    np.random.randint(50000, 500000),
                    np.random.uniform(200, 800),
                    np.random.uniform(50, 300),
                    np.random.uniform(0.05, 0.2),
                    np.random.uniform(0.02, 0.1),
                    27015,
                    0, 1,
                    np.random.randint(250, 1000),
                    np.random.randint(250, 1000),
                    np.random.uniform(10000, 100000),
                ]
            elif traffic_type == "P2P":
                features = [
                    np.random.uniform(600, 7200),
                    np.random.randint(2000, 20000),
                    np.random.randint(1000000, 50000000),
                    np.random.uniform(1000, 1500),
                    np.random.uniform(200, 600),
                    np.random.uniform(0.05, 0.5),
                    np.random.uniform(0.02, 0.2),
                    6881,
                    1, 0,
                    np.random.randint(1000, 10000),
                    np.random.randint(1000, 10000),
                    np.random.uniform(100000, 1000000),
                ]

            data.append(features)
            labels.append(traffic_type)

    X = np.array(data)
    y = np.array(labels)

    print(f"\n✅ Generated {len(X):,} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Data split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")

    # Initialize and train RL classifier
    rl_classifier = RLClassifier(n_classes=len(np.unique(y)), n_features=X.shape[1])

    start_time = time.time()
    rl_classifier.fit(X_train, y_train, episodes=5000)
    training_time = time.time() - start_time

    print(f"\n⏱️  Training time: {training_time:.2f}s")

    # Evaluate
    accuracy = rl_classifier.evaluate(X_test, y_test)

    print(f"\n🎯 RL Classification Results:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"   Q-table states: {len(rl_classifier.q_table)}")
    print(f"   Training episodes: 5000")
    print(f"   Learning rate: {rl_classifier.alpha}")
    print(f"   Discount factor: {rl_classifier.gamma}")
    print(f"   Exploration rate: {rl_classifier.epsilon}")

    # Compare with traditional ML (if available)
    try:
        with open("ml_models/traffic_classifier_large_scale.pkl", "rb") as f:
            ml_model = pickle.load(f)
        ml_predictions = ml_model.predict(rl_classifier.scaler.transform(X_test))
        ml_accuracy = accuracy_score(y_test, ml_predictions)
        print(f"\n📈 Comparison with ML model:")
        print(f"   ML Accuracy: {ml_accuracy:.4f} ({ml_accuracy * 100:.2f}%)")
        print(f"   RL vs ML: {accuracy - ml_accuracy:.4f} difference")
    except:
        print("\n⚠️  ML model not available for comparison")

    print("\n✅ RL-based classification on large scale data complete!")


if __name__ == "__main__":
    main()