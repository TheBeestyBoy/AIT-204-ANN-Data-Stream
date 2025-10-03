# Real-Time Fraud Detection System - Streamlit App

## Overview
This Streamlit application provides a real-time visualization of fraud detection on 50,000 transactions using a pretrained neural network model.

## Features
- 🚨 Real-time fraud detection with visual alerts
- 📊 Live performance metrics (Accuracy, Precision, Recall, F1-Score)
- 📈 Fraud probability trend visualization
- 🎯 Confusion matrix tracking
- ⚙️ Adjustable detection speed (0.1 - 2.0 seconds per transaction)
- 💾 Uses pretrained model from training

## Prerequisites

1. **Trained Model**: You must have already trained the model using `fraud_model.py`
   - This creates `best_fraud_detector.pth`

2. **Transaction Data**: You must have collected transactions using `fraud_collector.py`
   - This creates `fraud_transactions_YYYYMMDD_HHMMSS.pkl` or `.csv`

3. **Python Dependencies**: Install required packages

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy torch scikit-learn plotly matplotlib
```

## Running the App

### Option 1: Command Line
```bash
streamlit run streamlit_app.py
```

### Option 2: With Custom Port
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Option 3: Network Access
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

## How to Use

1. **Start the App**: Run the command above
2. **Wait for Loading**: The app loads the model and 50,000 transactions
3. **Control Panel** (Left Sidebar):
   - Adjust detection speed (default: 0.5 seconds)
   - Click "▶️ Start" to begin processing transactions
   - Click "⏸️ Stop" to pause
   - Click "🔄 Reset" to start over

4. **Dashboard Features**:
   - **Top Metrics**: Transactions processed, fraud detected, accuracy
   - **Current Transaction**: Shows fraud probability gauge and details
   - **Transaction Details**: User ID, amount, merchant, time, etc.
   - **Fraud Trend Chart**: Last 100 transactions' fraud probability
   - **Confusion Matrix**: Real-time TP, FP, TN, FN tracking
   - **Performance Metrics**: Precision, Recall, F1-Score

5. **Display Options** (Sidebar):
   - Toggle transaction details
   - Toggle fraud trend chart

## File Structure
```
AIT-204-ANN-Data-Stream/
├── streamlit_app.py              # Main Streamlit application
├── fraud_model.py                # Model training and loading
├── fraud_collector.py            # Data collection script
├── fraud_stream_server.py        # Data streaming server
├── feature_engineering.py        # Feature transformation
├── best_fraud_detector.pth       # Pretrained model (required)
├── fraud_transactions_*.pkl      # Transaction data (required)
├── requirements.txt              # Python dependencies
└── README_STREAMLIT.md          # This file
```

## Troubleshooting

### Model Not Found Error
```
Error: Model file 'best_fraud_detector.pth' not found!
```
**Solution**: Run `python fraud_model.py` first to train the model

### Transaction Data Not Found Error
```
Error: No transaction data found!
```
**Solution**: Run `python fraud_collector.py` first to collect 50,000 transactions

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Use a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Slow Performance
- Reduce the speed slider to process faster
- Close other browser tabs
- Check if GPU is being utilized (CUDA)

## Performance

- **Processes**: 50,000 transactions
- **Speed**: Adjustable from 0.1 to 2.0 seconds per transaction
- **Time to Complete**: 
  - At 0.5 sec/transaction: ~7 hours
  - At 0.1 sec/transaction: ~1.4 hours
- **Model**: Neural network with ~13,000 parameters
- **Expected Accuracy**: ~98-99%

## Tips

1. **Use GPU**: The app automatically uses CUDA if available for faster predictions
2. **Faster Processing**: Set speed to 0.1 seconds for quicker demonstration
3. **Presentation Mode**: Use 0.5-1.0 seconds for better visibility during presentations
4. **Stop/Start**: You can pause at any time and resume processing
5. **Reset**: Click reset to reprocess from the beginning with different settings

## Dashboard Metrics Explained

- **Precision**: Of all transactions flagged as fraud, how many were actually fraud?
- **Recall**: Of all actual fraud transactions, how many did we detect?
- **F1-Score**: Harmonic mean of precision and recall
- **True Positives (TP)**: Correctly identified fraud
- **False Positives (FP)**: Normal transactions incorrectly flagged as fraud
- **True Negatives (TN)**: Correctly identified normal transactions
- **False Negatives (FN)**: Fraud transactions missed by the model

## Support

For issues or questions, check:
1. All required files are present
2. Dependencies are installed
3. Model has been trained
4. Data has been collected

## License

Educational use for CST-405 class project.
