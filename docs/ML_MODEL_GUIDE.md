# ML Model Guide (v2.0)

Complete guide to Machine Learning models used in the AI-SDN Traffic Classification System.

## 🧠 Available Models

### 1. 🌟 Large Scale Model (Recommended)
**File**: `ml_models/traffic_classifier_large_scale.pkl`

**Performance Metrics:**
- **Accuracy**: 99.83%
- **Classes**: 6 (HTTP, HTTPS, FTP, SSH, Video, VoIP)
- **Features**: 14 optimized features
- **Training Time**: 2-3 minutes
- **Inference Time**: <1ms per flow
- **Model Size**: ~2.5 MB

**Best For:**
- Production deployment
- Highest accuracy requirements
- Real-time classification
- Comprehensive traffic analysis

**Training Script**: `ml_models/train_large_scale.py`

### 2. 🚀 Real Traffic Model
**File**: `ml_models/traffic_classifier_real.pkl`

**Performance Metrics:**
- **Accuracy**: 95-98%
- **Classes**: 5 (VoIP, Gaming, Video, HTTP, FTP)
- **Features**: 18 comprehensive features
- **Training Time**: 5-10 minutes
- **Inference Time**: <1ms per flow
- **Model Size**: ~3.2 MB

**Best For:**
- Real-world traffic patterns
- Research applications
- Diverse network environments

**Training Script**: `ml_models/train_classifier_real.py`

### 3. ⚡ Basic Model
**File**: `ml_models/traffic_classifier.pkl`

**Performance Metrics:**
- **Accuracy**: 90-95%
- **Classes**: 5 (VoIP, Gaming, Video, HTTP, FTP)
- **Features**: 12 essential features
- **Training Time**: 1-2 minutes
- **Inference Time**: <1ms per flow
- **Model Size**: ~1.8 MB

**Best For:**
- Quick testing and learning
- Resource-constrained environments
- Development and debugging

**Training Script**: `ml_models/train_classifier.py`

## 📊 Feature Sets Comparison

### Large Scale Model (14 Features)
```
1. Flow Duration
2. Protocol Type (TCP/UDP)
3. Source Port
4. Destination Port
5. Forward Packet Count
6. Forward Byte Count
7. Backward Packet Count
8. Backward Byte Count
9. Minimum Packet Size
10. Maximum Packet Size
11. Mean Packet Size
12. Flow Bytes per Second
13. Flow Packets per Second
14. Average Packet Size
```

### Real Traffic Model (18 Features)
```
All Large Scale Features plus:
15. Packet Size Standard Deviation
16. Forward Inter-arrival Time Mean
17. Backward Inter-arrival Time Mean
18. Flow Direction Ratio
```

### Basic Model (12 Features)
```
1. Flow Duration
2. Protocol Type
3. Source Port
4. Destination Port
5. Total Packet Count
6. Total Byte Count
7. Average Packet Size
8. Flow Rate (bytes/sec)
9. Packet Rate (packets/sec)
10. Forward-Backward Ratio
11. Packet Size Variance
12. Idle Time Ratio
```

## 🎯 Traffic Classes

### Large Scale Model Classes (6)
| Class | Priority | Typical Ports | Characteristics |
|-------|----------|---------------|----------------|
| **VoIP** | 5 (Highest) | 5060, 16384-32768 | Small packets, low latency, UDP |
| **Video** | 4 | 554, 1935, 5000 | Medium packets, streaming, TCP/UDP |
| **HTTPS** | 3 | 443, 8443 | Medium packets, encrypted, TCP |
| **HTTP** | 2 | 80, 8080, 8000 | Variable packets, web traffic, TCP |
| **SSH** | 1 | 22, 2222 | Small packets, interactive, TCP |
| **FTP** | 0 (Lowest) | 20, 21 | Large packets, bulk transfer, TCP |

### Real/Basic Model Classes (5)
| Class | Priority | Typical Ports | Characteristics |
|-------|----------|---------------|----------------|
| **VoIP** | 3 | 5060, 16384-32768 | Small packets, low latency, UDP |
| **Gaming** | 3 | 27015, 25565, 3478 | Small packets, real-time, UDP/TCP |
| **Video** | 2 | 554, 1935, 5000 | Medium packets, streaming, TCP/UDP |
| **HTTP** | 1 | 80, 8080, 8000 | Variable packets, web traffic, TCP |
| **FTP** | 0 | 20, 21 | Large packets, bulk transfer, TCP |

## 🔧 Model Training

### Training Large Scale Model
```bash
cd ~/sdn-ai-traffic-classifier

# Train the large scale model
python3 ml_models/train_large_scale.py

# Expected output:
# Large Scale Model Training Started
# Loading dataset: data/traffic_dataset.csv
# Dataset shape: (50000, 15)
# Training Random Forest with 100 estimators...
# Model Accuracy: 0.9983 (99.83%)
# Model saved as: ml_models/traffic_classifier_large_scale.pkl
# Training report saved as: training_report_large_scale.json
```

### Training Real Traffic Model
```bash
# Requires real datasets (CIC-IDS2017, UNSW-NB15)
python3 ml_models/train_classifier_real.py

# Expected output:
# Real Traffic Model Training Started
# Processing datasets...
# Training Random Forest with 200 estimators...
# Model Accuracy: 0.9625 (96.25%)
# Model saved as: ml_models/traffic_classifier_real.pkl
```

### Training Basic Model
```bash
# Quick training with synthetic data
python3 ml_models/train_classifier.py

# Expected output:
# Basic Model Training Started
# Generating synthetic dataset...
# Training Random Forest with 50 estimators...
# Model Accuracy: 0.9234 (92.34%)
# Model saved as: ml_models/traffic_classifier.pkl
```

## 📈 Model Evaluation

### Comprehensive Accuracy Testing
```bash
# Test large scale model accuracy
python3 test_large_scale_accuracy_fixed.py

# Output includes:
# - Overall accuracy metrics
# - Per-class precision/recall/F1 scores
# - Confusion matrix visualization
# - Feature importance analysis
# - Performance comparison charts
```

### Manual Model Evaluation
```bash
# Load and test model manually
python3 -c "
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load model
model = pickle.load(open('ml_models/traffic_classifier_large_scale.pkl', 'rb'))
print(f'Model features: {len(model.feature_names_in_)}')
print(f'Model classes: {model.classes_}')

# Load test data
test_data = pd.read_csv('data/traffic_test_data.csv')
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Predict
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
"
```

## 🔄 Model Selection and Switching

### Switching Models in Controller
Edit `controller/intelligent_controller.py`:

```python
# Change this line to switch models
MODEL_FILE = 'ml_models/traffic_classifier_large_scale.pkl'  # 99.83% accuracy
# MODEL_FILE = 'ml_models/traffic_classifier_real.pkl'      # 95-98% accuracy
# MODEL_FILE = 'ml_models/traffic_classifier.pkl'           # 90-95% accuracy
```

### Model Comparison Script
```python
# Create a comparison script
import pickle
import time
import numpy as np

models = {
    'Large Scale': 'ml_models/traffic_classifier_large_scale.pkl',
    'Real Traffic': 'ml_models/traffic_classifier_real.pkl',
    'Basic': 'ml_models/traffic_classifier.pkl'
}

test_features = np.random.rand(1, 14)  # Adjust feature count per model

for name, path in models.items():
    try:
        model = pickle.load(open(path, 'rb'))
        start_time = time.time()
        prediction = model.predict(test_features[:len(model.feature_names_in_)])
        inference_time = (time.time() - start_time) * 1000
        
        print(f'{name}: {prediction[0]} ({inference_time:.2f}ms)')
    except Exception as e:
        print(f'{name}: Error - {e}')
```

## 🎛️ Model Configuration

### Hyperparameter Tuning

#### Large Scale Model
```python
# In ml_models/train_large_scale.py
model_params = {
    'n_estimators': 100,        # Number of trees
    'max_depth': 20,           # Maximum tree depth
    'min_samples_split': 5,     # Minimum samples to split
    'min_samples_leaf': 2,      # Minimum samples per leaf
    'random_state': 42,         # Reproducibility
    'n_jobs': -1               # Use all CPU cores
}
```

#### Real Traffic Model
```python
# In ml_models/train_classifier_real.py
model_params = {
    'n_estimators': 200,        # More trees for complex data
    'max_depth': 30,           # Deeper trees
    'min_samples_split': 10,    # More conservative splitting
    'min_samples_leaf': 5,      # Larger leaves
    'random_state': 42,
    'n_jobs': -1
}
```

#### Basic Model
```python
# In ml_models/train_classifier.py
model_params = {
    'n_estimators': 50,         # Fewer trees for speed
    'max_depth': 15,           # Shallower trees
    'min_samples_split': 2,     # Aggressive splitting
    'min_samples_leaf': 1,      # Small leaves
    'random_state': 42,
    'n_jobs': -1
}
```

### Feature Engineering

#### Adding New Features
```python
# In training script, add new features to dataset
def extract_advanced_features(df):
    # Add new features
    df['packet_size_variance'] = df['packet_size'].var()
    df['flow_efficiency'] = df['total_bytes'] / df['duration']
    df['burstiness'] = df['packet_rate'].std()
    
    return df
```

#### Feature Selection
```python
# Use feature importance to select best features
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print(f'Selected features: {selected_features.tolist()}')
```

## 📊 Performance Analysis

### Feature Importance Analysis
```bash
# View feature importance for large scale model
python3 -c "
import pickle
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('ml_models/traffic_classifier_large_scale.pkl', 'rb'))
feature_names = model.feature_names_in_
importances = model.feature_importances_

# Create DataFrame
feature_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_df)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_df['feature'], feature_df['importance'])
plt.title('Feature Importance - Large Scale Model')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance_analysis.png')
"
```

### Confusion Matrix Analysis
```bash
# Generate detailed confusion matrix
python3 -c "
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = pickle.load(open('ml_models/traffic_classifier_large_scale.pkl', 'rb'))
test_data = pd.read_csv('data/traffic_test_data.csv')
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Large Scale Model')
plt.tight_layout()
plt.savefig('confusion_matrix_detailed.png')
"
```

## 🐛 Model Troubleshooting

### Common Model Issues

#### Model Loading Errors
```bash
# Check model file integrity
python3 -c "
import pickle
import os

for model_file in ['ml_models/traffic_classifier_large_scale.pkl',
                   'ml_models/traffic_classifier_real.pkl',
                   'ml_models/traffic_classifier.pkl']:
    if os.path.exists(model_file):
        try:
            model = pickle.load(open(model_file, 'rb'))
            print(f'✓ {model_file}: {type(model).__name__}')
            if hasattr(model, 'feature_names_in_'):
                print(f'  Features: {len(model.feature_names_in_)}')
            if hasattr(model, 'classes_'):
                print(f'  Classes: {len(model.classes_)}')
        except Exception as e:
            print(f'✗ {model_file}: {e}')
    else:
        print(f'✗ {model_file}: File not found')
"
```

#### Feature Mismatch Errors
```bash
# Debug feature extraction
python3 -c "
import pickle
import pandas as pd

model = pickle.load(open('ml_models/traffic_classifier_large_scale.pkl', 'rb'))
print(f'Expected features ({len(model.feature_names_in_)}):')
for i, feature in enumerate(model.feature_names_in_):
    print(f'  {i+1:2d}. {feature}')

# Check test data
test_data = pd.read_csv('data/traffic_test_data.csv')
print(f'\\nTest data columns ({len(test_data.columns)}):')
for i, col in enumerate(test_data.columns):
    print(f'  {i+1:2d}. {col}')
"
```

#### Low Accuracy Issues
```python
# Debug low accuracy
def debug_model_performance():
    # Check data quality
    print("Data Quality Check:")
    print(f"Missing values: {X_train.isnull().sum().sum()}")
    print(f"Duplicate rows: {X_train.duplicated().sum()}")
    
    # Check class balance
    print("\nClass Distribution:")
    print(y_train.value_counts())
    
    # Check feature distributions
    print("\nFeature Statistics:")
    print(X_train.describe())
    
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\nCross-validation scores: {scores}")
    print(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## 🚀 Production Deployment

### Model Optimization for Production

#### Model Compression
```python
# Reduce model size for production
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train with optimized parameters
optimized_model = RandomForestClassifier(
    n_estimators=50,          # Reduce trees
    max_depth=15,             # Limit depth
    min_samples_leaf=5,       # Larger leaves
    random_state=42,
    n_jobs=-1
)

# Compress model
joblib.dump(optimized_model, 'ml_models/traffic_classifier_compressed.pkl', compress=3)
```

#### Batch Prediction Optimization
```python
# Optimize for batch processing
def batch_predict(model, features_batch, batch_size=1000):
    predictions = []
    confidences = []
    
    for i in range(0, len(features_batch), batch_size):
        batch = features_batch[i:i+batch_size]
        pred_batch = model.predict(batch)
        prob_batch = model.predict_proba(batch)
        
        predictions.extend(pred_batch)
        confidences.extend(prob_batch.max(axis=1))
    
    return predictions, confidences
```

### Model Monitoring

#### Real-time Performance Monitoring
```python
# Add to controller for monitoring
class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.confidences = []
        self.prediction_times = []
    
    def log_prediction(self, prediction, confidence, pred_time):
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.prediction_times.append(pred_time)
        
        # Alert if performance degrades
        if len(self.prediction_times) > 100:
            avg_time = sum(self.prediction_times[-100:]) / 100
            if avg_time > 0.005:  # 5ms threshold
                print(f"WARNING: High prediction time: {avg_time:.3f}s")
```

## 📚 Advanced Topics

### Ensemble Methods
```python
# Combine multiple models for better accuracy
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Create ensemble
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('lr', LogisticRegression(max_iter=1000)),
        ('svm', SVC(probability=True))
    ],
    voting='soft'  # Use probabilities
)

# Train ensemble
ensemble_model.fit(X_train, y_train)
```

### Online Learning
```python
# Update model with new data
from sklearn.linear_model import SGDClassifier

# Use online learning model
online_model = SGDClassifier(loss='log_loss', random_state=42)

# Train incrementally
for batch in data_stream:
    X_batch, y_batch = batch
    online_model.partial_fit(X_batch, y_batch, classes=all_classes)
```

### Deep Learning Models
```python
# Neural network for traffic classification
from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42
)

nn_model.fit(X_train, y_train)
```

## 📖 References and Resources

### Key Papers
1. **"A Survey on Machine Learning for Network Traffic Classification"** - IEEE Communications Surveys & Tutorials
2. **"Deep Learning for Network Traffic Classification"** - ACM SIGCOMM
3. **"Feature Selection for Network Traffic Classification"** - IEEE Transactions on Network and Service Management

### Useful Libraries
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib/seaborn**: Visualization
- **joblib**: Model serialization

### Datasets for Research
- **CIC-IDS2017**: Comprehensive intrusion detection dataset
- **UNSW-NB15**: Network behavior analysis
- **ISCX VPN**: VPN traffic classification
- **MAWI**: Wide-area traffic traces

---

**🎯 Recommendation**: Use the **Large Scale Model** for production deployment due to its 99.83% accuracy and optimized feature set.

**📈 Performance**: All models achieve <1ms inference time, making them suitable for real-time network traffic classification.

**🔄 Updates**: Models can be retrained with new data to adapt to changing traffic patterns and network conditions.