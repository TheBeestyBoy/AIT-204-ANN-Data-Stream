# 🚨 Real-Time Fraud Detection System

An AI-powered fraud detection system that uses neural networks to identify fraudulent transactions in real-time. This project includes data collection, model training, and an interactive Streamlit dashboard for visualization.

## 🎯 Project Overview

This system processes **50,000 transactions** and detects fraud using a trained neural network with **98-99% accuracy**. The Streamlit app provides real-time visualization of fraud detection with interactive charts and metrics.

## 📊 Features

- **Real-time fraud detection** with visual alerts
- **Neural network model** with 13,000+ parameters
- **Interactive dashboard** built with Streamlit
- **Live performance metrics** (Accuracy, Precision, Recall, F1-Score)
- **Fraud trend visualization** with Plotly charts
- **Confusion matrix tracking** for model evaluation
- **Adjustable processing speed** (0.1 - 2.0 seconds per transaction)

## 🏗️ Project Structure

```
ait-204-ann-data-stream/
├── streamlit_app.py              # Main Streamlit dashboard
├── fraud_model.py                # Neural network training
├── fraud_collector.py            # Data collection from stream
├── fraud_stream_server.py        # Transaction streaming server
├── feature_engineering.py        # Feature transformation
├── best_fraud_detector.pth       # Trained model (Git LFS)
├── fraud_transactions_*.pkl      # Transaction data (Git LFS)
├── requirements.txt              # Python dependencies
├── DEPLOYMENT_GUIDE.md          # Streamlit Cloud deployment guide
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ait-204-ann-data-stream.git
cd ait-204-ann-data-stream
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Collect Training Data (Optional - if starting fresh)

Start the server:
```bash
python fraud_stream_server.py
```

In another terminal, collect data:
```bash
python fraud_collector.py
```

### 4. Train the Model (Optional - if starting fresh)

```bash
python fraud_model.py
```

### 5. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions on deploying to Streamlit Cloud.

**Quick steps:**
1. Push this repo to GitHub (with Git LFS for large files)
2. Go to https://share.streamlit.io/
3. Connect your GitHub repo
4. Deploy!

## 📦 Dependencies

- **Python 3.8+**
- **PyTorch** - Neural network framework
- **Streamlit** - Interactive web app
- **Pandas & NumPy** - Data processing
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning utilities

## 🎮 How to Use the App

1. **Start the app** with `streamlit run streamlit_app.py`
2. **Adjust speed** using the slider in the sidebar (default: 0.5 seconds)
3. **Click "▶️ Start"** to begin processing transactions
4. **Watch real-time detection** with visual fraud alerts
5. **Monitor performance** with live accuracy metrics
6. **Click "⏸️ Stop"** to pause or **"🔄 Reset"** to restart

## 📈 Model Performance

- **Accuracy**: ~98-99%
- **Precision**: High (minimizes false positives)
- **Recall**: Good (catches most fraud)
- **F1-Score**: Balanced performance
- **Training Data**: 50,000 transactions
- **Fraud Rate**: ~1.6%

## 🧠 Neural Network Architecture

```
Input Layer (14 features)
    ↓
Hidden Layer 1 (128 neurons) + BatchNorm + ReLU + Dropout
    ↓
Hidden Layer 2 (64 neurons) + BatchNorm + ReLU + Dropout
    ↓
Hidden Layer 3 (32 neurons) + BatchNorm + ReLU + Dropout
    ↓
Hidden Layer 4 (16 neurons) + BatchNorm + ReLU + Dropout
    ↓
Output Layer (1 neuron) + Sigmoid
```

## 📊 Features Used for Detection

1. Transaction amount
2. Hour of day
3. Weekend indicator
4. Days since last transaction
5. Amount Z-score (deviation from user's typical spending)
6. Time since midnight
7. Merchant risk score
8. User velocity
9. Amount percentile
10. Hour risk score
11. Round amount indicator
12. Day of week
13. Merchant category
14. User trust score

## 🎓 Use Cases

- **Educational**: Machine learning and fraud detection demonstration
- **Research**: Testing fraud detection algorithms
- **Presentation**: Live demo of real-time AI system
- **Training**: Understanding neural networks and classification

## 🔧 Customization

### Adjust Detection Speed
Change the slider in the Streamlit sidebar (0.1 - 2.0 seconds)

### Modify Model Architecture
Edit `fraud_model.py` and change `hidden_dims` parameter:
```python
self.model = FraudDetectionNet(
    input_dim=input_dim,
    hidden_dims=[256, 128, 64, 32],  # Modify this
    dropout_rate=0.3
)
```

### Change Data Collection Size
Edit `fraud_collector.py`:
```python
collector = FraudDataCollector(
    collection_size=100000  # Change this
)
```

## 📝 License

Educational use for CST-405 class project.

## 👨‍💻 Author

Created for CST-405: Applied Intelligent Technologies

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit for the amazing web app framework
- scikit-learn for ML utilities

## 📧 Support

For issues or questions:
1. Check the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. Review requirements and dependencies
3. Ensure model and data files are present

---

**⭐ If you find this project useful, please star it on GitHub!**
