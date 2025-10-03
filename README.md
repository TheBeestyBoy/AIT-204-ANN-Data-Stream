# Real-Time Fraud Detection System with PyTorch

## Overview
This is a complete real-time fraud detection system using PyTorch neural networks to identify potentially fraudulent financial transactions with >90% accuracy.

## Files Created
1. **fraud_collector.py** - Collects transaction data from the streaming server
2. **feature_engineering.py** - Feature extraction and transformation module
3. **fraud_model.py** - PyTorch neural network model for fraud detection
4. **fraud_detection_client.py** - Real-time fraud detection client
5. **fraud_dashboard.py** - Streamlit visualization dashboard

## Setup Instructions

### 1. Install Dependencies
```bash
pip install torch pandas numpy scikit-learn matplotlib streamlit plotly
```

### 2. Start the Fraud Stream Server
First, make sure the fraud stream server is running:
```bash
# Terminal 1: Start YOUR server (Individual work)
python fraud_stream_server.py --host localhost --port 5555

# OR for group work (one partner runs server)
python fraud_stream_server.py --host 0.0.0.0 --port 5555
```

### 3. Collect Training Data
Run the data collector to gather transactions for training:
```bash
# Terminal 2: Collect data
python fraud_collector.py
```
This will collect 1000 transactions and save them to CSV and pickle files.

### 4. Train the Fraud Detection Model
Train the PyTorch neural network on the collected data:
```bash
python fraud_model.py
```
This will:
- Load the most recent data file
- Train a neural network with 4 hidden layers
- Use weighted sampling to handle class imbalance
- Find the optimal decision threshold
- Save the trained model as `fraud_detector.pth`
- Display training curves and evaluation metrics

### 5. Run Real-Time Detection
Start the real-time fraud detection client:
```bash
python fraud_detection_client.py --host localhost --port 5555
```
Options:
- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 5555)
- `--model`: Model file path (default: fraud_detector.pth)
- `--id`: Client ID for identification

The detector will:
- Connect to the streaming server
- Process transactions in real-time
- Alert when fraud is detected
- Track user and merchant statistics
- Display performance metrics
- Save results to CSV

### 6. Visualization Dashboard (Optional)
You have two options for visualization:

#### Option A: Use the provided visual client (Recommended)
```bash
python3 stream_client_visual.py --host localhost --port 5555
```

#### Option B: Run Streamlit Dashboard
```bash
streamlit run fraud_dashboard.py
```
The dashboard provides:
- Real-time metrics and statistics
- Transaction time series charts
- Fraud probability distributions
- Confusion matrix and ROC curves
- Recent fraud alerts table
- Export functionality

## Model Architecture
The fraud detection model uses:
- **Input Layer**: 14 engineered features
- **Hidden Layers**: [128, 64, 32, 16] neurons with BatchNorm, ReLU, and Dropout
- **Output Layer**: Sigmoid activation for binary classification
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam with weight decay
- **Class Balancing**: Weighted random sampling

## Feature Engineering
The system extracts 14 features including:
- Transaction amount and velocity
- Time-based risk scores
- User transaction history
- Merchant risk profiles
- Statistical features (z-scores, percentiles)

## Performance Goals
- **Accuracy**: >90%
- **Precision**: High to minimize false positives
- **Recall**: High to catch actual fraud
- **Latency**: <50ms per transaction
- **Throughput**: >20 transactions/second

## Network Configurations

### Individual Work (Default)
```bash
# Everything runs on localhost
Server: python3 fraud_stream_server.py --host localhost --port 5555
Client: python fraud_detection_client.py --host localhost --port 5555
```

### Group Work
```bash
# Partner 1 (Server)
python3 fraud_stream_server.py --host 0.0.0.0 --port 5555

# Other Partners (Clients)
python fraud_detection_client.py --host PARTNER_IP --port 5555
```

### Class-Wide Activity
```bash
# Connect to instructor's server
python fraud_detection_client.py --host INSTRUCTOR_IP --port 5555
```

## Troubleshooting

### No data files found
Make sure to run `fraud_collector.py` first to collect training data.

### Model not found
Train a model first using `fraud_model.py`.

### Connection refused
1. Check that the server is running
2. Verify the host and port are correct
3. Check firewall settings for group work

### Low accuracy
1. Collect more training data (increase collection_size)
2. Adjust model hyperparameters
3. Engineer additional features
4. Tune the decision threshold

## Expected Results
- Fraud detection rate: 5-10% of transactions
- Model accuracy: >90%
- Processing latency: <50ms per transaction
- Real-time alerts with confidence scores

## Files Generated
- `fraud_transactions_*.csv` - Collected transaction data
- `fraud_transactions_*.pkl` - Pickled transaction data
- `fraud_detector.pth` - Trained PyTorch model
- `best_fraud_detector.pth` - Best model during training
- `training_history.png` - Training curves
- `fraud_detection_results_*.csv` - Detection results

## Notes
- The fraud rate in the stream is approximately 5-10%
- The model uses class balancing to handle imbalanced data
- Real-time detection includes user and merchant profiling
- The dashboard updates automatically if auto-refresh is enabled
