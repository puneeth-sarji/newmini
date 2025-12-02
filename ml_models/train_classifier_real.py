#!/usr/bin/env python3
"""
Advanced Traffic Classifier Training with Real Datasets
Supports multiple ML algorithms and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
import time

warnings.filterwarnings('ignore')

class AdvancedTrafficClassifier:
    """Advanced traffic classifier with multiple algorithms"""
    
    def __init__(self, data_dir='../traffic_data/processed'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path('../ml_models')
        self.models_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None
        
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        
        # Define models to try
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20,
                min_samples_split=5,
                random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
    
    def load_dataset(self, file_path='unified_traffic_dataset.csv'):
        """Load processed dataset"""
        dataset_path = self.data_dir / file_path
        
        if not dataset_path.exists():
            print(f"Error: Dataset not found at {dataset_path}")
            print("Please run dataset_processor.py first")
            return None, None
        
        print(f"\nLoading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
        print(f"✓ Loaded dataset: {df.shape}")
        print(f"  Features: {df.shape[1] - 1}")
        print(f"  Samples: {df.shape[0]}")
        print(f"\nClass distribution:")
        for app, count in df['application'].value_counts().items():
            print(f"  {app}: {count} ({count/len(df)*100:.1f}%)")
        
        # Separate features and labels
        X = df.drop(['application'], axis=1)
        y = df['application']
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\nEncoding categorical features: {list(categorical_cols)}")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, use_pca=False, n_components=10):
        """Prepare data for training"""
        print("\nPreparing data...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA if requested
        if use_pca:
            print(f"  Applying PCA (n_components={n_components})...")
            self.pca = PCA(n_components=n_components, random_state=42)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            print(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_single_model(self, model, model_name, X_train, y_train, X_test, y_test):
        """Train and evaluate a single model"""
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print(f"\nResults:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Training time:   {train_time:.2f}s")
        print(f"  Prediction time: {predict_time:.4f}s")
        
        # Cross-validation
        print(f"\nCross-validation (5-fold)...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
        print(f"  CV Scores: {cv_scores}")
        print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        # Update best model
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model
            self.best_model_name = model_name
            print(f"\n🏆 New best model!")
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': train_time,
            'predict_time': predict_time,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare"""
        print("\n" + "="*60)
        print("TRAINING MULTIPLE MODELS")
        print("="*60)
        
        results = []
        
        for model_name, model in self.models.items():
            try:
                result = self.train_single_model(
                    model, model_name, X_train, y_train, X_test, y_test
                )
                results.append(result)
            except Exception as e:
                print(f"\n✗ Error training {model_name}: {e}")
        
        # Compare results
        self.compare_models(results)
        
        return results
    
    def compare_models(self, results):
        """Compare model performance"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('accuracy', ascending=False)
        
        print("\nRanked by Accuracy:")
        print(df_results.to_string(index=False))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        ax = axes[0, 0]
        ax.barh(df_results['model_name'], df_results['accuracy'])
        ax.set_xlabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_xlim([0, 1])
        
        # Precision, Recall, F1
        ax = axes[0, 1]
        x = np.arange(len(df_results))
        width = 0.25
        ax.bar(x - width, df_results['precision'], width, label='Precision')
        ax.bar(x, df_results['recall'], width, label='Recall')
        ax.bar(x + width, df_results['f1'], width, label='F1-Score')
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, F1-Score')
        ax.set_xticks(x)
        ax.set_xticklabels(df_results['model_name'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        
        # Training time
        ax = axes[1, 0]
        ax.barh(df_results['model_name'], df_results['train_time'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Training Time')
        
        # Cross-validation scores
        ax = axes[1, 1]
        ax.barh(df_results['model_name'], df_results['cv_mean'])
        ax.set_xlabel('CV Score')
        ax.set_title('Cross-Validation Score (5-fold)')
        ax.set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'model_comparison.png', dpi=300)
        print(f"\n✓ Comparison plot saved: {self.models_dir / 'model_comparison.png'}")
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.models_dir / 'confusion_matrix.png', dpi=300)
        print(f"✓ Confusion matrix saved: {self.models_dir / 'confusion_matrix.png'}")
    
    def plot_feature_importance(self, X, feature_names):
        """Plot feature importance for tree-based models"""
        if not hasattr(self.best_model, 'feature_importances_'):
            print("Feature importance not available for this model")
            return
        
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importances - {self.best_model_name}")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=90)
        plt.tight_layout()
        plt.savefig(self.models_dir / 'feature_importance.png', dpi=300)
        print(f"✓ Feature importance plot saved: {self.models_dir / 'feature_importance.png'}")
        
        # Print top features
        print("\nTop 10 Most Important Features:")
        for i in range(min(10, len(importances))):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for Random Forest"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING (Random Forest)")
        print("="*60)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        print("Searching best parameters...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model(self, filename='traffic_classifier_real.pkl'):
        """Save the best trained model"""
        model_path = self.models_dir / filename
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'pca': self.pca,
            'accuracy': self.best_accuracy,
            'classes': self.label_encoder.classes_.tolist()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Best model saved: {model_path}")
        print(f"  Model: {self.best_model_name}")
        print(f"  Accuracy: {self.best_accuracy:.4f}")
        
        # Save model info as JSON
        info_path = self.models_dir / 'model_info.json'
        info = {
            'model_name': self.best_model_name,
            'accuracy': float(self.best_accuracy),
            'classes': self.label_encoder.classes_.tolist(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Model info saved: {info_path}")
    
    def full_training_pipeline(self, use_pca=False, tune_hyperparameters=False):
        """Complete training pipeline"""
        print("\n" + "="*70)
        print(" "*15 + "TRAFFIC CLASSIFIER TRAINING")
        print("="*70)
        
        # Load dataset
        X, y = self.load_dataset()
        if X is None:
            return
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            X, y, use_pca=use_pca
        )
        
        # Hyperparameter tuning (optional)
        if tune_hyperparameters:
            tuned_model = self.hyperparameter_tuning(X_train, y_train)
            self.models['Random Forest (Tuned)'] = tuned_model
        
        # Train all models
        results = self.train_all_models(X_train, y_train, X_test, y_test)
        
        # Generate visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        y_pred = self.best_model.predict(X_test)
        self.plot_confusion_matrix(y_test, y_pred)
        
        if not use_pca:
            self.plot_feature_importance(X_train, X.columns)
        
        # Save best model
        self.save_model()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"📊 Accuracy: {self.best_accuracy:.4f}")
        print(f"\n✓ Model ready for deployment!")
        print(f"✓ Use with: controller/intelligent_controller.py")
        print("="*70)

def main():
    """Main training function"""
    classifier = AdvancedTrafficClassifier()
    
    # Run full training pipeline
    classifier.full_training_pipeline(
        use_pca=False,  # Set to True to use PCA
        tune_hyperparameters=False  # Set to True for hyperparameter tuning (slower)
    )

if __name__ == '__main__':
    main()