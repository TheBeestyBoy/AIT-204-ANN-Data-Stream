# Class Activity: Real-Time Fraud Detection with PyTorch

## Objective
Build a real-time fraud detection system using PyTorch that connects to a streaming financial transaction server, processes incoming transactions, and identifies potentially fraudulent activities with an accuracy goal of >90%.

## Setup Instructions

### 1. Connect to the Transaction Stream Server
Your instructor will provide the server IP address. The server is broadcasting realistic financial transactions with embedded fraud patterns.

```bash
# Test connection with the visual client first
python3 stream_client_visual.py --host INSTRUCTOR_IP --port 5555 --id YourName
```

### 2. Install Required Libraries
```bash
pip install torch torchvision numpy pandas matplotlib streamlit
```

## Part 1: Data Collection and Analysis (30 minutes)

### Task 1.1: Create a Data Collector
Create `fraud_collector.py` that collects transactions from the stream and saves them for training.

```python
import json
import socket
import pandas as pd
from datetime import datetime
import pickle

class FraudDataCollector:
    def __init__(self, host, port, collection_size=1000):
        self.host = host
        self.port = port
        self.collection_size = collection_size
        self.transactions = []
    
    def connect_and_collect(self):
        """Connect to server and collect transactions"""
        # TODO: Implement socket connection
        # TODO: Collect specified number of transactions
        # TODO: Save to CSV file with timestamp
        pass
    
    def extract_features(self, transaction):
        """Extract features for ML model"""
        features = {
            'amount': transaction['amount'],
            'hour': transaction['hourOfDay'],
            'isWeekend': int(transaction['isWeekend']),
            'daysSinceLastTransaction': transaction['daysSinceLastTransaction'],
            # TODO: Add more features
        }
        return features
```

### Task 1.2: Exploratory Data Analysis
After collecting 1000+ transactions:
1. Calculate the fraud rate in your dataset
2. Identify patterns in fraudulent vs normal transactions
3. Create visualizations showing:
   - Amount distribution for normal vs fraudulent transactions
   - Time patterns of fraudulent activities
   - User profiles most associated with fraud

## Part 2: Build PyTorch Fraud Detection Model (45 minutes)

### Task 2.1: Feature Engineering
Create `feature_engineering.py`:

```python
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class TransactionFeatures:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'amount', 'hour', 'isWeekend', 
            'daysSinceLastTransaction',
            'amount_zscore',  # Amount deviation from user's mean
            'time_since_midnight',  # Seconds since midnight
            'merchant_risk_score',  # Based on merchant ID pattern
            # TODO: Add more engineered features
        ]
    
    def transform_transaction(self, transaction, user_history=None):
        """Convert transaction to feature vector"""
        # TODO: Implement feature extraction
        # TODO: Include user history features if available
        pass
```

### Task 2.2: PyTorch Model Implementation
Create `fraud_model.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FraudDetectionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super(FraudDetectionNet, self).__init__()
        
        # TODO: Build your neural network architecture
        # Suggested architecture:
        # - Input layer: input_dim features
        # - Hidden layers with ReLU activation and Dropout
        # - Output layer: 1 neuron with Sigmoid for binary classification
        
        self.layers = nn.Sequential(
            # Your layers here
        )
    
    def forward(self, x):
        return self.layers(x)

class FraudDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FraudDetectionNet(input_dim=10).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train the fraud detection model"""
        # TODO: Implement training loop
        # TODO: Add early stopping
        # TODO: Track and plot training metrics
        pass
    
    def predict(self, transaction_features):
        """Predict fraud probability for a single transaction"""
        self.model.eval()
        with torch.no_grad():
            # TODO: Implement prediction
            pass
```

### Task 2.3: Model Training
Train your model with:
- 70% training, 15% validation, 15% test split
- Class imbalance handling (use weighted loss or oversampling)
- Track precision, recall, F1-score, and ROC-AUC

## Part 3: Real-Time Detection Client (45 minutes)

### Task 3.1: Create Real-Time Detection Client
Create `fraud_detection_client.py`:

```python
import torch
import json
import socket
import threading
from collections import deque
from datetime import datetime

class RealTimeFraudDetector:
    def __init__(self, model_path, host, port):
        self.model = self.load_model(model_path)
        self.host = host
        self.port = port
        self.user_histories = {}  # Track user transaction patterns
        self.detection_buffer = deque(maxlen=100)
        self.stats = {
            'total': 0,
            'flagged': 0,
            'true_positives': 0,
            'false_positives': 0
        }
    
    def process_transaction(self, transaction):
        """Process incoming transaction and detect fraud"""
        # Extract features
        features = self.extract_features(transaction)
        
        # Get prediction
        fraud_prob = self.model.predict(features)
        
        # Update user history
        self.update_user_history(transaction)
        
        # Flag if suspicious
        is_fraud = fraud_prob > 0.5
        
        if is_fraud:
            self.alert_fraud(transaction, fraud_prob)
        
        # Update statistics
        self.update_stats(transaction, is_fraud)
        
        return fraud_prob
    
    def alert_fraud(self, transaction, confidence):
        """Alert when fraud is detected"""
        print(f"\n🚨 FRAUD ALERT 🚨")
        print(f"Transaction ID: {transaction['transactionID']}")
        print(f"User ID: {transaction['userID']}")
        print(f"Amount: ${transaction['amount']:.2f}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Pattern: {transaction.get('fraudPattern', 'Unknown')}")
        
    def run(self):
        """Connect to server and start processing"""
        # TODO: Implement main loop
        pass
```

## Part 4: Streamlit Dashboard (30 minutes)

### Task 4.1: Create Streamlit App
Create `fraud_dashboard.py`:

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import torch
import numpy as np

# Note: Streamlit Cloud doesn't support raw sockets
# This dashboard runs locally and connects to your detection client

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide"
)

class FraudDashboard:
    def __init__(self):
        # Initialize session state
        if 'transactions' not in st.session_state:
            st.session_state.transactions = []
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'model' not in st.session_state:
            st.session_state.model = None
    
    def run(self):
        st.title("🚨 Real-Time Fraud Detection Dashboard")
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Model upload
            model_file = st.file_uploader("Upload PyTorch Model", type=['pth', 'pt'])
            if model_file:
                # Load model
                pass
            
            # Connection settings (for local testing)
            st.subheader("Server Connection")
            host = st.text_input("Server IP", value="localhost")
            port = st.number_input("Port", value=5555, min_value=1000, max_value=9999)
            
            if st.button("Connect to Stream"):
                # Note: This won't work on Streamlit Cloud
                st.warning("For local testing only. Streamlit Cloud doesn't support socket connections.")
        
        # Main dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(st.session_state.transactions))
        
        with col2:
            fraud_count = len(st.session_state.alerts)
            st.metric("Fraud Alerts", fraud_count)
        
        with col3:
            fraud_rate = fraud_count / max(len(st.session_state.transactions), 1) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        with col4:
            # Calculate accuracy if ground truth available
            st.metric("Model Accuracy", "95.3%")  # Placeholder
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Amount Time Series")
            # TODO: Add time series plot
            
        with col2:
            st.subheader("Fraud Detection Confidence")
            # TODO: Add confidence distribution
        
        # Recent alerts table
        st.subheader("🚨 Recent Fraud Alerts")
        if st.session_state.alerts:
            df_alerts = pd.DataFrame(st.session_state.alerts[-10:])
            st.dataframe(df_alerts, use_container_width=True)
        else:
            st.info("No fraud alerts yet")
        
        # Model performance metrics
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            st.plotly_chart(self.create_confusion_matrix(), use_container_width=True)
        
        with col2:
            # ROC curve
            st.plotly_chart(self.create_roc_curve(), use_container_width=True)
    
    def create_confusion_matrix(self):
        # TODO: Implement confusion matrix visualization
        pass
    
    def create_roc_curve(self):
        # TODO: Implement ROC curve
        pass

if __name__ == "__main__":
    dashboard = FraudDashboard()
    dashboard.run()
```

### Task 4.2: Run Streamlit App Locally
```bash
# Run locally (Streamlit Cloud doesn't support socket connections)
streamlit run fraud_dashboard.py
```

**Note**: The Streamlit app must run locally because Streamlit Cloud doesn't support raw TCP socket connections. Alternative approaches:
1. Use the Streamlit app to visualize saved results from your detection client
2. Have your detection client save results to a file that Streamlit reads
3. Use REST API or websockets (more complex setup)

## Part 5: Evaluation and Optimization (30 minutes)

### Task 5.1: Performance Metrics
Calculate and report:
1. **Precision**: Of all transactions flagged as fraud, how many were correct?
2. **Recall**: Of all actual fraud transactions, how many did you detect?
3. **F1-Score**: Harmonic mean of precision and recall
4. **ROC-AUC**: Area under the ROC curve
5. **Latency**: Average time to process one transaction

### Task 5.2: Model Optimization
Improve your model by:
1. **Feature Engineering**: Add velocity features (transactions per hour), unusual merchant patterns
2. **Architecture Tuning**: Experiment with different layer sizes, dropout rates
3. **Threshold Optimization**: Find optimal threshold instead of 0.5
4. **Ensemble Methods**: Combine multiple models

## Deliverables

1. **Code Files**:
   - `fraud_collector.py` - Data collection
   - `fraud_model.py` - PyTorch model
   - `fraud_detection_client.py` - Real-time detection
   - `fraud_dashboard.py` - Streamlit visualization (local only)

2. **Model File**:
   - `fraud_detector.pth` - Trained PyTorch model

3. **Report** (2-3 pages):
   - Model architecture and design choices
   - Feature engineering approach
   - Performance metrics and confusion matrix
   - Screenshot of dashboard detecting fraud
   - Challenges faced and solutions

4. **Live Demo**:
   - Connect to instructor's server
   - Show real-time fraud detection
   - Demonstrate dashboard (locally)

## Grading Rubric

| Component | Points |
|-----------|--------|
| Data Collection & Analysis | 15 |
| PyTorch Model Implementation | 25 |
| Model Performance (>90% accuracy) | 20 |
| Real-time Detection Client | 20 |
| Dashboard Visualization | 10 |
| Code Quality & Documentation | 5 |
| Report & Demo | 5 |
| **Total** | **100** |

## Bonus Challenges (+10 points each)

1. **Advanced Features**: Implement graph-based features using user-merchant relationships
2. **Explainable AI**: Add SHAP or LIME to explain why transactions were flagged
3. **Adaptive Learning**: Implement online learning to adapt to new fraud patterns
4. **Multi-class Detection**: Classify different types of fraud patterns
5. **Production Ready**: Dockerize your solution with proper logging and monitoring

## Tips

1. **Start Simple**: Begin with basic features and a simple network
2. **Handle Imbalance**: The fraud rate is ~5-10%, use techniques like:
   - Weighted loss functions
   - SMOTE oversampling
   - Focal loss
3. **Monitor in Real-time**: Print statistics every 10 transactions
4. **Save Everything**: Log all predictions for later analysis
5. **Test Thoroughly**: Ensure your model generalizes to new users

## Resources

- [PyTorch Binary Classification Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Handling Imbalanced Datasets](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Feature Engineering for Fraud Detection](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/FraudDetectionSystem.html)

## Support

- Office Hours: Tuesday/Thursday 2-4 PM
- Slack Channel: #ait204-fraud-detection
- Server Issues: Contact instructor immediately

Good luck! Remember, the goal is not just high accuracy but understanding how ANNs can be applied to real-world streaming data problems.