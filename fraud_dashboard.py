import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import torch
import numpy as np
import time
import glob
import os
from fraud_model import FraudDetector

# Page configuration
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
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
    
    def load_model(self, model_path):
        """Load the fraud detection model"""
        try:
            detector = FraudDetector(model_path=model_path)
            st.session_state.model = detector
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def load_results(self, results_file):
        """Load detection results from CSV"""
        try:
            df = pd.read_csv(results_file)
            # Convert to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Store transactions
            st.session_state.transactions = df.to_dict('records')
            
            # Extract alerts
            alerts = df[df['predicted_fraud'] == True].to_dict('records')
            st.session_state.alerts = alerts
            
            return df
        except Exception as e:
            st.error(f"Error loading results: {e}")
            return None
    
    def create_metrics_cards(self, df):
        """Create metric cards for dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Transactions", 
                f"{len(df):,}",
                delta=f"+{len(df[df['timestamp'] > datetime.now() - timedelta(hours=1)]):,} last hour"
            )
        
        with col2:
            fraud_count = df['predicted_fraud'].sum()
            fraud_rate = fraud_count / len(df) * 100 if len(df) > 0 else 0
            st.metric(
                "Fraud Alerts", 
                f"{fraud_count:,}",
                delta=f"{fraud_rate:.1f}% rate"
            )
        
        with col3:
            if 'actual_fraud' in df.columns:
                tp = ((df['predicted_fraud'] == True) & (df['actual_fraud'] == True)).sum()
                tn = ((df['predicted_fraud'] == False) & (df['actual_fraud'] == False)).sum()
                accuracy = (tp + tn) / len(df) * 100 if len(df) > 0 else 0
                st.metric("Model Accuracy", f"{accuracy:.1f}%")
            else:
                st.metric("Avg Confidence", f"{df['fraud_probability'].mean():.2%}")
        
        with col4:
            total_amount = df['amount'].sum()
            fraud_amount = df[df['predicted_fraud'] == True]['amount'].sum()
            st.metric(
                "Fraud Amount", 
                f"${fraud_amount:,.2f}",
                delta=f"{fraud_amount/total_amount*100:.1f}% of total" if total_amount > 0 else "0%"
            )
    
    def create_time_series_chart(self, df):
        """Create time series chart of transactions"""
        # Group by hour
        df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
        
        hourly_stats = df.groupby('hour').agg({
            'amount': 'sum',
            'predicted_fraud': 'sum',
            'transactionID': 'count'
        }).reset_index()
        
        hourly_stats.columns = ['hour', 'total_amount', 'fraud_count', 'transaction_count']
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Transaction Volume', 'Fraud Detection Rate'),
            vertical_spacing=0.1
        )
        
        # Transaction volume
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['transaction_count'],
                mode='lines+markers',
                name='Transactions',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Fraud alerts
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['fraud_count'],
                mode='lines+markers',
                name='Fraud Alerts',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.1)'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Fraud Alerts", row=2, col=1)
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_text="Transaction Activity Over Time"
        )
        
        return fig
    
    def create_confidence_distribution(self, df):
        """Create histogram of fraud confidence scores"""
        fig = go.Figure()
        
        # All transactions
        fig.add_trace(go.Histogram(
            x=df['fraud_probability'],
            name='All Transactions',
            opacity=0.7,
            nbinsx=50,
            marker_color='blue'
        ))
        
        # Actual fraud (if available)
        if 'actual_fraud' in df.columns:
            fraud_df = df[df['actual_fraud'] == True]
            fig.add_trace(go.Histogram(
                x=fraud_df['fraud_probability'],
                name='Actual Fraud',
                opacity=0.7,
                nbinsx=50,
                marker_color='red'
            ))
        
        fig.update_layout(
            title="Fraud Probability Distribution",
            xaxis_title="Fraud Probability",
            yaxis_title="Count",
            barmode='overlay',
            height=400
        )
        
        # Add threshold line
        if st.session_state.model:
            threshold = st.session_state.model.best_threshold
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Threshold: {threshold:.2f}"
            )
        
        return fig
    
    def create_confusion_matrix(self, df):
        """Create confusion matrix visualization"""
        if 'actual_fraud' not in df.columns:
            return None
        
        # Calculate confusion matrix
        tp = ((df['predicted_fraud'] == True) & (df['actual_fraud'] == True)).sum()
        fp = ((df['predicted_fraud'] == True) & (df['actual_fraud'] == False)).sum()
        tn = ((df['predicted_fraud'] == False) & (df['actual_fraud'] == False)).sum()
        fn = ((df['predicted_fraud'] == False) & (df['actual_fraud'] == True)).sum()
        
        # Create matrix
        matrix = [[tn, fp], [fn, tp]]
        labels = ['Normal', 'Fraud']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=[[f'TN: {tn}', f'FP: {fp}'], [f'FN: {fn}', f'TP: {tp}']],
            texttemplate='%{text}',
            colorscale='RdBu_r',
            showscale=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            xaxis=dict(side='bottom')
        )
        
        return fig
    
    def create_roc_curve(self, df):
        """Create ROC curve"""
        if 'actual_fraud' not in df.columns:
            return None
        
        # Calculate ROC points
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            predicted = df['fraud_probability'] > threshold
            tp = ((predicted == True) & (df['actual_fraud'] == True)).sum()
            fp = ((predicted == True) & (df['actual_fraud'] == False)).sum()
            tn = ((predicted == False) & (df['actual_fraud'] == False)).sum()
            fn = ((predicted == False) & (df['actual_fraud'] == True)).sum()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Calculate AUC
        auc = np.trapz(tpr_list, fpr_list)
        
        # Create plot
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr_list,
            y=tpr_list,
            mode='lines',
            name=f'ROC Curve (AUC = {auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def run(self):
        """Main dashboard application"""
        st.title("🚨 Real-Time Fraud Detection Dashboard")
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Model selection
            st.subheader("Model")
            model_files = glob.glob("*.pth")
            if model_files:
                selected_model = st.selectbox("Select Model", model_files)
                if st.button("Load Model"):
                    if self.load_model(selected_model):
                        st.success("Model loaded successfully!")
            else:
                st.warning("No model files found. Train a model first.")
            
            # Results file selection
            st.subheader("Load Results")
            results_files = glob.glob("fraud_detection_results_*.csv")
            if results_files:
                # Sort by modification time
                results_files.sort(key=os.path.getmtime, reverse=True)
                selected_results = st.selectbox("Select Results File", results_files)
                if st.button("Load Results"):
                    df = self.load_results(selected_results)
                    if df is not None:
                        st.success(f"Loaded {len(df)} transactions")
            
            # Auto-refresh option
            st.subheader("Auto Refresh")
            auto_refresh = st.checkbox("Enable Auto Refresh")
            refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
            
            if auto_refresh:
                time.sleep(refresh_interval)
                st.rerun()
        
        # Main dashboard content
        if st.session_state.transactions:
            df = pd.DataFrame(st.session_state.transactions)
            
            # Metrics cards
            self.create_metrics_cards(df)
            
            # Charts row 1
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Transaction Activity")
                fig = self.create_time_series_chart(df)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Fraud Confidence Distribution")
                fig = self.create_confidence_distribution(df)
                st.plotly_chart(fig, use_container_width=True)
            
            # Charts row 2
            col1, col2 = st.columns(2)
            
            with col1:
                if 'actual_fraud' in df.columns:
                    st.subheader("Confusion Matrix")
                    fig = self.create_confusion_matrix(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.subheader("Amount Distribution")
                    fig = px.box(
                        df, 
                        x='predicted_fraud', 
                        y='amount',
                        title="Transaction Amounts by Prediction",
                        labels={'predicted_fraud': 'Predicted as Fraud', 'amount': 'Amount ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'actual_fraud' in df.columns:
                    st.subheader("ROC Curve")
                    fig = self.create_roc_curve(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.subheader("Top Risk Merchants")
                    merchant_risk = df.groupby('merchantID').agg({
                        'predicted_fraud': 'sum',
                        'transactionID': 'count'
                    }).reset_index()
                    merchant_risk['fraud_rate'] = merchant_risk['predicted_fraud'] / merchant_risk['transactionID']
                    merchant_risk = merchant_risk.sort_values('fraud_rate', ascending=False).head(10)
                    
                    fig = px.bar(
                        merchant_risk,
                        x='fraud_rate',
                        y='merchantID',
                        orientation='h',
                        title="Merchants with Highest Fraud Rate",
                        labels={'fraud_rate': 'Fraud Rate', 'merchantID': 'Merchant ID'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recent alerts table
            st.subheader("🚨 Recent Fraud Alerts")
            if st.session_state.alerts:
                alerts_df = pd.DataFrame(st.session_state.alerts[-20:])
                
                # Format for display
                display_cols = ['timestamp', 'transactionID', 'userID', 'merchantID', 
                               'amount', 'fraud_probability']
                if all(col in alerts_df.columns for col in display_cols):
                    alerts_df = alerts_df[display_cols]
                    alerts_df['fraud_probability'] = alerts_df['fraud_probability'].apply(lambda x: f"{x:.2%}")
                    alerts_df['amount'] = alerts_df['amount'].apply(lambda x: f"${x:.2f}")
                    
                    st.dataframe(
                        alerts_df,
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("No fraud alerts yet")
            
            # Performance metrics
            if 'actual_fraud' in df.columns:
                st.subheader("Model Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate metrics
                tp = ((df['predicted_fraud'] == True) & (df['actual_fraud'] == True)).sum()
                fp = ((df['predicted_fraud'] == True) & (df['actual_fraud'] == False)).sum()
                tn = ((df['predicted_fraud'] == False) & (df['actual_fraud'] == False)).sum()
                fn = ((df['predicted_fraud'] == False) & (df['actual_fraud'] == True)).sum()
                
                accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("Precision", f"{precision:.4f}")
                with col3:
                    st.metric("Recall", f"{recall:.4f}")
                with col4:
                    st.metric("F1-Score", f"{f1:.4f}")
            
            # Export options
            st.subheader("Export Data")
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV",
                    csv,
                    "fraud_detection_export.csv",
                    "text/csv",
                    key='download-csv'
                )
            with col2:
                if st.session_state.alerts:
                    alerts_csv = pd.DataFrame(st.session_state.alerts).to_csv(index=False)
                    st.download_button(
                        "Download Alerts as CSV",
                        alerts_csv,
                        "fraud_alerts_export.csv",
                        "text/csv",
                        key='download-alerts'
                    )
        else:
            # No data loaded
            st.info("No data loaded. Please load results from the sidebar.")
            
            # Instructions
            with st.expander("How to use this dashboard"):
                st.markdown("""
                1. **Train a model**: Run `python fraud_model.py` to train a fraud detection model
                2. **Collect data**: Run `python fraud_collector.py` to collect transaction data
                3. **Run detection**: Run `python fraud_detection_client.py` to detect fraud in real-time
                4. **Load results**: Use the sidebar to load the detection results CSV file
                5. **Analyze**: View metrics, charts, and alerts in this dashboard
                
                The dashboard will automatically update if auto-refresh is enabled.
                """)

if __name__ == "__main__":
    dashboard = FraudDashboard()
    dashboard.run()
