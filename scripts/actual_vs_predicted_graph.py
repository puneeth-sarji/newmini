#!/usr/bin/env python3
"""
Generate actual vs predicted visualization for the ML traffic classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import json

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_and_data():
    """Load the trained model and test data"""
    print("Loading model and data...")
    
    # Load model
    with open('ml_models/traffic_classifier_large_scale.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    classes = model_data['classes']
    
    # Load test data and predictions
    df_predictions = pd.read_csv('model_predictions.csv')
    df_test_data = pd.read_csv('traffic_test_data.csv')
    
    print(f"✓ Loaded model: {model_data['model_name']}")
    print(f"✓ Classes: {classes}")
    print(f"✓ Test samples: {len(df_predictions)}")
    
    return model, scaler, label_encoder, feature_names, classes, df_predictions, df_test_data

def create_actual_vs_predicted_plot(df_predictions, classes):
    """Create actual vs predicted visualization"""
    print("\nCreating actual vs predicted plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Actual vs Predicted Traffic Classification', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, class_name in enumerate(classes):
        ax = axes[i]
        
        # Filter data for this class
        class_data = df_predictions[df_predictions['true_label'] == class_name]
        
        # Count predictions
        actual_counts = class_data['true_label'].value_counts()
        predicted_counts = class_data['predicted_label'].value_counts()
        
        # Create bar plot
        x_pos = np.arange(len(classes))
        width = 0.35
        
        # Actual counts (all should be this class)
        actual_bars = ax.bar(x_pos - width/2, [actual_counts.get(c, 0) for c in classes], 
                            width, label='Actual', alpha=0.8, color='skyblue')
        
        # Predicted counts
        predicted_bars = ax.bar(x_pos + width/2, [predicted_counts.get(c, 0) for c in classes], 
                               width, label='Predicted', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Traffic Type')
        ax.set_ylabel('Count')
        ax.set_title(f'{class_name} Traffic (n={len(class_data)})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in actual_bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        for bar in predicted_bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved actual_vs_predicted.png")

def create_confusion_matrix_heatmap(df_predictions, classes):
    """Create confusion matrix heatmap"""
    print("\nCreating confusion matrix heatmap...")
    
    # Create confusion matrix
    y_true = df_predictions['true_label']
    y_pred = df_predictions['predicted_label']
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix (Raw Counts)', fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved confusion_matrix_comparison.png")

def create_prediction_confidence_plot(df_predictions, classes):
    """Create prediction confidence visualization"""
    print("\nCreating prediction confidence plot...")
    
    # Calculate confidence scores
    confidence_cols = [f'prob_{c}' for c in classes]
    df_predictions['max_confidence'] = df_predictions[confidence_cols].max(axis=1)
    df_predictions['is_correct'] = df_predictions['true_label'] == df_predictions['predicted_label']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prediction Confidence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall confidence distribution
    ax1 = axes[0, 0]
    ax1.hist(df_predictions['max_confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Confidence Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence by correctness
    ax2 = axes[0, 1]
    correct_conf = df_predictions[df_predictions['is_correct']]['max_confidence']
    incorrect_conf = df_predictions[~df_predictions['is_correct']]['max_confidence']
    
    ax2.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
    if len(incorrect_conf) > 0:
        ax2.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence by Prediction Correctness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence by class
    ax3 = axes[1, 0]
    class_confidence = []
    class_labels = []
    
    for class_name in classes:
        class_data = df_predictions[df_predictions['true_label'] == class_name]
        class_confidence.extend(class_data['max_confidence'])
        class_labels.extend([class_name] * len(class_data))
    
    confidence_df = pd.DataFrame({'confidence': class_confidence, 'class': class_labels})
    
    sns.boxplot(data=confidence_df, x='class', y='confidence', ax=ax3)
    ax3.set_xlabel('Traffic Type')
    ax3.set_ylabel('Prediction Confidence')
    ax3.set_title('Confidence Distribution by Traffic Type')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy vs Confidence threshold
    ax4 = axes[1, 1]
    thresholds = np.arange(0.5, 1.01, 0.05)
    accuracies = []
    
    for threshold in thresholds:
        high_conf = df_predictions[df_predictions['max_confidence'] >= threshold]
        if len(high_conf) > 0:
            accuracy = high_conf['is_correct'].mean()
            accuracies.append(accuracy)
        else:
            accuracies.append(0)
    
    ax4.plot(thresholds, accuracies, 'bo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Confidence Threshold')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy vs Confidence Threshold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.8, 1.01)
    
    plt.tight_layout()
    plt.savefig('prediction_confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved prediction_confidence_analysis.png")

def create_performance_summary(df_predictions, classes):
    """Create performance summary visualization"""
    print("\nCreating performance summary...")
    
    # Calculate metrics
    y_true = df_predictions['true_label']
    y_pred = df_predictions['predicted_label']
    
    # Overall accuracy
    overall_accuracy = (y_true == y_pred).mean()
    
    # Per-class accuracy
    class_accuracies = {}
    for class_name in classes:
        class_data = df_predictions[df_predictions['true_label'] == class_name]
        class_accuracy = (class_data['true_label'] == class_data['predicted_label']).mean()
        class_accuracies[class_name] = class_accuracy
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
    
    # Overall accuracy gauge
    ax1.text(0.5, 0.7, f'{overall_accuracy:.1%}', fontsize=48, ha='center', va='center', 
             fontweight='bold', color='green' if overall_accuracy > 0.9 else 'orange')
    ax1.text(0.5, 0.3, 'Overall Accuracy', fontsize=16, ha='center', va='center')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Add circular progress indicator
    circle = plt.Circle((0.5, 0.5), 0.4, fill=False, edgecolor='lightgray', linewidth=8)
    ax1.add_patch(circle)
    
    # Add progress arc
    theta = np.linspace(0, 2 * np.pi * overall_accuracy, 100)
    x_arc = 0.5 + 0.4 * np.cos(theta)
    y_arc = 0.5 + 0.4 * np.sin(theta)
    ax1.plot(x_arc, y_arc, color='green' if overall_accuracy > 0.9 else 'orange', linewidth=8)
    
    # Per-class accuracy bar chart
    class_names = list(class_accuracies.keys())
    acc_values = list(class_accuracies.values())
    colors = ['green' if acc >= 0.95 else 'orange' if acc >= 0.9 else 'red' for acc in acc_values]
    
    bars = ax2.bar(class_names, acc_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Traffic Type')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Per-Class Accuracy')
    ax2.set_ylim(0.8, 1.01)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, acc_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved performance_summary.png")

def main():
    """Main function to generate all visualizations"""
    print("=" * 60)
    print("GENERATING ACTUAL VS PREDICTED VISUALIZATIONS")
    print("=" * 60)
    
    # Load data
    model, scaler, label_encoder, feature_names, classes, df_predictions, df_test_data = load_model_and_data()
    
    # Generate visualizations
    create_actual_vs_predicted_plot(df_predictions, classes)
    create_confusion_matrix_heatmap(df_predictions, classes)
    create_prediction_confidence_plot(df_predictions, classes)
    create_performance_summary(df_predictions, classes)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("  - actual_vs_predicted.png")
    print("  - confusion_matrix_comparison.png")
    print("  - prediction_confidence_analysis.png")
    print("  - performance_summary.png")
    print("=" * 60)

if __name__ == "__main__":
    main()