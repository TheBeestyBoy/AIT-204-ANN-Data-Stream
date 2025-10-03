import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fraud_model import FraudDetector
import glob
import os

# Page configuration
st.set_page_config(
    page_title="Real-Time Fraud Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .normal-alert {
        background-color: #00cc66;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pretrained fraud detection model"""
    model_path = 'best_fraud_detector.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please train the model first.")
        st.stop()
    
    detector = FraudDetector(model_path=model_path)
    return detector

@st.cache_data
def load_transactions():
    """Load the transaction dataset"""
    # Find the most recent data file
    pkl_files = glob.glob("fraud_transactions_*.pkl")
    csv_files = glob.glob("fraud_transactions_*.csv")
    
    if pkl_files:
        data_file = max(pkl_files, key=os.path.getctime)
        with open(data_file, 'rb') as f:
            transactions = pickle.load(f)
            df = pd.DataFrame(transactions)
    elif csv_files:
        data_file = max(csv_files, key=os.path.getctime)
        df = pd.read_csv(data_file)
    else:
        st.error("No transaction data found! Please run fraud_collector.py first.")
        st.stop()
    
    return df, data_file

def prepare_features(detector, transaction_row):
    """Prepare features for a single transaction"""
    transaction_dict = transaction_row.to_dict()
    features = detector.feature_transformer.transform_transaction(transaction_dict)
    return features

def create_gauge_chart(value, title, threshold=0.5):
    """Create a gauge chart for fraud probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': threshold * 100},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#00cc66'},
                {'range': [30, 70], 'color': '#ffcc00'},
                {'range': [70, 100], 'color': '#ff4b4b'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_fraud_trend_chart(fraud_history):
    """Create a line chart showing fraud detection trend"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(fraud_history))),
        y=fraud_history,
        mode='lines',
        name='Fraud Probability',
        line=dict(color='royalblue', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Fraud Probability Trend",
        xaxis_title="Transaction Number",
        yaxis_title="Fraud Probability",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_confusion_matrix_chart(tp, fp, tn, fn):
    """Create a confusion matrix visualization"""
    z = [[tn, fp], [fn, tp]]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=['Predicted Normal', 'Predicted Fraud'],
        y=['Actual Normal', 'Actual Fraud'],
        text=z,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='RdYlGn_r',
        showscale=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    # Title and description
    st.title("🚨 Real-Time Fraud Detection System")
    st.markdown("### Monitoring 50,000 Transactions with AI-Powered Detection")
    
    # Load model and data
    with st.spinner("Loading model and transaction data..."):
        detector = load_model()
        df, data_file = load_transactions()
    
    st.success(f"✅ Loaded {len(df):,} transactions from {data_file}")
    st.info(f"🧠 Model: Neural Network with {sum(p.numel() for p in detector.model.parameters()):,} parameters")
    
    # Sidebar controls
    st.sidebar.title("⚙️ Control Panel")
    
    # Speed control
    speed = st.sidebar.slider("Detection Speed (seconds per transaction)", 0.1, 2.0, 0.5, 0.1)
    
    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)
    start_button = col1.button("▶️ Start", use_container_width=True)
    stop_button = col2.button("⏸️ Stop", use_container_width=True)
    
    reset_button = st.sidebar.button("🔄 Reset", use_container_width=True)
    
    # Display options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Options")
    show_details = st.sidebar.checkbox("Show Transaction Details", value=True)
    show_trend = st.sidebar.checkbox("Show Fraud Trend", value=True)
    
    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'fraud_count' not in st.session_state:
        st.session_state.fraud_count = 0
    if 'normal_count' not in st.session_state:
        st.session_state.normal_count = 0
    if 'fraud_history' not in st.session_state:
        st.session_state.fraud_history = []
    if 'tp' not in st.session_state:
        st.session_state.tp = 0
    if 'fp' not in st.session_state:
        st.session_state.fp = 0
    if 'tn' not in st.session_state:
        st.session_state.tn = 0
    if 'fn' not in st.session_state:
        st.session_state.fn = 0
    
    # Handle button clicks
    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False
    if reset_button:
        st.session_state.current_idx = 0
        st.session_state.fraud_count = 0
        st.session_state.normal_count = 0
        st.session_state.fraud_history = []
        st.session_state.tp = 0
        st.session_state.fp = 0
        st.session_state.tn = 0
        st.session_state.fn = 0
        st.session_state.running = False
        st.rerun()
    
    # Main dashboard
    if st.session_state.current_idx < len(df):
        # Get current transaction
        current_transaction = df.iloc[st.session_state.current_idx]
        
        # Prepare features and get prediction
        features = prepare_features(detector, current_transaction)
        fraud_prob = detector.predict(features)
        predicted_fraud = fraud_prob > detector.best_threshold
        actual_fraud = current_transaction['isFraud']
        
        # Update confusion matrix
        if predicted_fraud and actual_fraud:
            st.session_state.tp += 1
        elif predicted_fraud and not actual_fraud:
            st.session_state.fp += 1
        elif not predicted_fraud and not actual_fraud:
            st.session_state.tn += 1
        else:
            st.session_state.fn += 1
        
        # Update counts
        if predicted_fraud:
            st.session_state.fraud_count += 1
        else:
            st.session_state.normal_count += 1
        
        st.session_state.fraud_history.append(fraud_prob)
        
        # Progress bar
        progress = st.session_state.current_idx / len(df)
        st.progress(progress, text=f"Processing: {st.session_state.current_idx:,} / {len(df):,} transactions ({progress*100:.1f}%)")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Transactions Processed", f"{st.session_state.current_idx:,}")
        
        with col2:
            st.metric("Fraud Detected", st.session_state.fraud_count, 
                     delta=f"{100*st.session_state.fraud_count/max(st.session_state.current_idx, 1):.2f}%")
        
        with col3:
            st.metric("Normal Transactions", st.session_state.normal_count)
        
        with col4:
            # Calculate accuracy
            total_predictions = st.session_state.tp + st.session_state.fp + st.session_state.tn + st.session_state.fn
            if total_predictions > 0:
                accuracy = (st.session_state.tp + st.session_state.tn) / total_predictions * 100
            else:
                accuracy = 0
            st.metric("Accuracy", f"{accuracy:.2f}%")
        
        st.markdown("---")
        
        # Current transaction section
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("Current Transaction")
            
            # Alert box
            if predicted_fraud:
                st.markdown(f'<div class="fraud-alert">🚨 FRAUD DETECTED - Probability: {fraud_prob*100:.1f}%</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="normal-alert">✅ NORMAL - Probability: {(1-fraud_prob)*100:.1f}%</div>', 
                           unsafe_allow_html=True)
            
            # Gauge chart
            gauge_fig = create_gauge_chart(fraud_prob, "Fraud Probability", detector.best_threshold)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Actual vs Predicted
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Actual Label:**")
                if actual_fraud:
                    st.error("Fraud")
                else:
                    st.success("Normal")
            
            with col_b:
                st.markdown("**Predicted Label:**")
                if predicted_fraud:
                    st.error("Fraud")
                else:
                    st.success("Normal")
        
        with col2:
            if show_details:
                st.subheader("Transaction Details")
                
                # Transaction info in a nice table
                details_df = pd.DataFrame({
                    'Field': ['Transaction ID', 'User ID', 'Amount', 'Merchant ID', 'Hour of Day', 
                             'Is Weekend', 'User Profile', 'Item ID'],
                    'Value': [
                        current_transaction.get('transactionID', 'N/A'),
                        current_transaction.get('userID', 'N/A'),
                        f"${current_transaction.get('amount', 0):.2f}",
                        current_transaction.get('merchantID', 'N/A'),
                        current_transaction.get('hour', 0),
                        'Yes' if current_transaction.get('isWeekend', False) else 'No',
                        current_transaction.get('userProfile', 'N/A') if 'userProfile' in current_transaction else 'N/A',
                        current_transaction.get('itemID', 'N/A') if 'itemID' in current_transaction else 'N/A'
                    ]
                })
                st.dataframe(details_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            if show_trend and len(st.session_state.fraud_history) > 1:
                trend_fig = create_fraud_trend_chart(st.session_state.fraud_history[-100:])  # Last 100 transactions
                st.plotly_chart(trend_fig, use_container_width=True)
        
        with col2:
            confusion_fig = create_confusion_matrix_chart(
                st.session_state.tp, st.session_state.fp, 
                st.session_state.tn, st.session_state.fn
            )
            st.plotly_chart(confusion_fig, use_container_width=True)
        
        # Performance metrics
        st.markdown("---")
        st.subheader("📊 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total = st.session_state.tp + st.session_state.fp + st.session_state.tn + st.session_state.fn
        if total > 0:
            precision = st.session_state.tp / max(st.session_state.tp + st.session_state.fp, 1)
            recall = st.session_state.tp / max(st.session_state.tp + st.session_state.fn, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 0.0001)
            
            with col1:
                st.metric("Precision", f"{precision*100:.2f}%")
            with col2:
                st.metric("Recall", f"{recall*100:.2f}%")
            with col3:
                st.metric("F1-Score", f"{f1*100:.2f}%")
            with col4:
                st.metric("True Positives", st.session_state.tp)
        
        # Auto-advance if running
        if st.session_state.running and st.session_state.current_idx < len(df) - 1:
            st.session_state.current_idx += 1
            time.sleep(speed)
            st.rerun()
        elif st.session_state.current_idx >= len(df) - 1:
            st.session_state.running = False
            st.success("🎉 All transactions processed!")
    
    else:
        st.success("🎉 All transactions processed!")
        st.info("Click 'Reset' to start over.")
        
        # Final summary
        st.subheader("Final Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Processed", f"{len(df):,}")
        with col2:
            st.metric("Total Fraud Detected", st.session_state.fraud_count)
        with col3:
            total = st.session_state.tp + st.session_state.fp + st.session_state.tn + st.session_state.fn
            if total > 0:
                accuracy = (st.session_state.tp + st.session_state.tn) / total * 100
                st.metric("Final Accuracy", f"{accuracy:.2f}%")

if __name__ == "__main__":
    main()
