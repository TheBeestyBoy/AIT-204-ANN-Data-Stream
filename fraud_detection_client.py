import torch
import json
import socket
import threading
from collections import deque, defaultdict
from datetime import datetime
import time
import numpy as np
from fraud_model import FraudDetector
import pandas as pd

class RealTimeFraudDetector:
    def __init__(self, model_path, host='localhost', port=5555, client_id='FraudDetector'):
        """Initialize real-time fraud detector
        
        Args:
            model_path: Path to trained PyTorch model
            host: Server host
            port: Server port
            client_id: Client identifier
        """
        # Load trained model
        self.detector = FraudDetector(model_path=model_path)
        
        # Connection settings
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # User history tracking
        self.user_histories = defaultdict(lambda: {
            'transactions': deque(maxlen=100),
            'total_amount': 0,
            'count': 0,
            'last_transaction_time': None,
            'fraud_count': 0
        })
        
        # Merchant statistics
        self.merchant_stats = defaultdict(lambda: {
            'transactions': deque(maxlen=50),
            'fraud_alerts': 0,
            'total_transactions': 0
        })
        
        # Detection buffer for recent predictions
        self.detection_buffer = deque(maxlen=100)
        
        # Real-time statistics
        self.stats = {
            'total': 0,
            'flagged': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'total_amount': 0,
            'fraud_amount': 0,
            'start_time': datetime.now()
        }
        
        # Alert thresholds
        self.alert_threshold = self.detector.best_threshold
        self.high_confidence_threshold = 0.8
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        
        # Results storage for analysis
        self.results = []
    
    def process_transaction(self, transaction):
        """Process incoming transaction and detect fraud"""
        start_time = time.time()
        
        # Extract user and merchant IDs
        user_id = transaction.get('userID', '')
        merchant_id = transaction.get('merchantID', '')
        
        # Update user history before prediction
        self.update_user_history(transaction)
        
        # Get fraud probability
        fraud_prob = self.detector.predict(transaction)
        
        # Determine if fraud based on threshold
        is_fraud_predicted = fraud_prob > self.alert_threshold
        is_high_confidence = fraud_prob > self.high_confidence_threshold
        
        # Get actual fraud label if available
        actual_fraud = transaction.get('isFraud', False)
        
        # Store result
        result = {
            'timestamp': datetime.now().isoformat(),
            'transactionID': transaction.get('transactionID', ''),
            'userID': user_id,
            'merchantID': merchant_id,
            'amount': transaction.get('amount', 0),
            'fraud_probability': fraud_prob,
            'predicted_fraud': is_fraud_predicted,
            'actual_fraud': actual_fraud,
            'high_confidence': is_high_confidence
        }
        
        self.results.append(result)
        self.detection_buffer.append(result)
        
        # Alert if fraud detected
        if is_fraud_predicted:
            self.alert_fraud(transaction, fraud_prob, is_high_confidence)
            self.stats['flagged'] += 1
            self.stats['fraud_amount'] += transaction.get('amount', 0)
            
            # Update merchant stats
            self.merchant_stats[merchant_id]['fraud_alerts'] += 1
        
        # Update statistics
        self.update_stats(transaction, is_fraud_predicted, actual_fraud)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Update merchant stats
        self.merchant_stats[merchant_id]['total_transactions'] += 1
        
        return fraud_prob
    
    def update_user_history(self, transaction):
        """Update user transaction history"""
        user_id = transaction.get('userID', '')
        if not user_id:
            return
        
        user_hist = self.user_histories[user_id]
        
        # Add transaction to history
        user_hist['transactions'].append({
            'amount': transaction.get('amount', 0),
            'hour': transaction.get('hourOfDay', 0),
            'merchant': transaction.get('merchantID', ''),
            'timestamp': datetime.now()
        })
        
        # Update statistics
        user_hist['total_amount'] += transaction.get('amount', 0)
        user_hist['count'] += 1
        user_hist['last_transaction_time'] = datetime.now()
        
        # Calculate velocity features for next prediction
        if len(user_hist['transactions']) > 1:
            recent_transactions = list(user_hist['transactions'])[-10:]
            time_diffs = []
            for i in range(1, len(recent_transactions)):
                diff = (recent_transactions[i]['timestamp'] - 
                       recent_transactions[i-1]['timestamp']).total_seconds()
                time_diffs.append(diff)
            
            if time_diffs:
                user_hist['avg_time_between'] = np.mean(time_diffs)
    
    def alert_fraud(self, transaction, confidence, is_high_confidence):
        """Alert when fraud is detected"""
        alert_level = "🚨 HIGH CONFIDENCE" if is_high_confidence else "⚠️  SUSPICIOUS"
        
        print(f"\n{alert_level} FRAUD ALERT")
        print(f"{'='*50}")
        print(f"Transaction ID: {transaction.get('transactionID', 'Unknown')}")
        print(f"User ID: {transaction.get('userID', 'Unknown')}")
        print(f"Merchant ID: {transaction.get('merchantID', 'Unknown')}")
        print(f"Amount: ${transaction.get('amount', 0):.2f}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Time: {transaction.get('hourOfDay', 0)}:00")
        print(f"Pattern: {transaction.get('fraudPattern', 'Unknown')}")
        
        # Check user history for patterns
        user_id = transaction.get('userID', '')
        if user_id in self.user_histories:
            user_hist = self.user_histories[user_id]
            print(f"User History: {user_hist['count']} transactions, " 
                  f"{user_hist['fraud_count']} previous fraud alerts")
        
        # Check merchant risk
        merchant_id = transaction.get('merchantID', '')
        if merchant_id in self.merchant_stats:
            merchant = self.merchant_stats[merchant_id]
            fraud_rate = merchant['fraud_alerts'] / max(merchant['total_transactions'], 1)
            print(f"Merchant Risk: {fraud_rate:.2%} fraud rate")
        
        print(f"{'='*50}\n")
    
    def update_stats(self, transaction, predicted_fraud, actual_fraud):
        """Update detection statistics"""
        self.stats['total'] += 1
        self.stats['total_amount'] += transaction.get('amount', 0)
        
        # Update confusion matrix stats
        if predicted_fraud and actual_fraud:
            self.stats['true_positives'] += 1
        elif predicted_fraud and not actual_fraud:
            self.stats['false_positives'] += 1
        elif not predicted_fraud and actual_fraud:
            self.stats['false_negatives'] += 1
        else:
            self.stats['true_negatives'] += 1
        
        # Update user fraud count if detected
        if actual_fraud:
            user_id = transaction.get('userID', '')
            if user_id:
                self.user_histories[user_id]['fraud_count'] += 1
    
    def print_statistics(self):
        """Print current detection statistics"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        print("\n" + "="*60)
        print("REAL-TIME DETECTION STATISTICS")
        print("="*60)
        
        print(f"Runtime: {runtime:.0f} seconds")
        print(f"Total Transactions: {self.stats['total']}")
        print(f"Fraud Alerts: {self.stats['flagged']}")
        print(f"Alert Rate: {self.stats['flagged'] / max(self.stats['total'], 1) * 100:.2f}%")
        print(f"Total Amount: ${self.stats['total_amount']:.2f}")
        print(f"Fraud Amount: ${self.stats['fraud_amount']:.2f}")
        
        # Calculate accuracy metrics if we have ground truth
        tp = self.stats['true_positives']
        fp = self.stats['false_positives']
        tn = self.stats['true_negatives']
        fn = self.stats['false_negatives']
        
        if (tp + fp + tn + fn) > 0:
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 0.001)
            
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(f"  True Positives: {tp}")
            print(f"  False Positives: {fp}")
            print(f"  True Negatives: {tn}")
            print(f"  False Negatives: {fn}")
        
        # Processing performance
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000  # Convert to ms
            max_time = np.max(self.processing_times) * 1000
            print(f"\nProcessing Performance:")
            print(f"  Avg Latency: {avg_time:.2f} ms")
            print(f"  Max Latency: {max_time:.2f} ms")
            print(f"  Throughput: {self.stats['total'] / max(runtime, 1):.1f} tx/sec")
        
        print("="*60 + "\n")
    
    def run(self):
        """Connect to server and start processing transactions"""
        print(f"Connecting to fraud stream server at {self.host}:{self.port}...")
        
        try:
            # Create socket connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.host, self.port))
            print(f"Connected successfully!")
            
            # Send handshake
            handshake = {'clientID': self.client_id, 'clientType': 'detector'}
            message = json.dumps(handshake) + '\n'
            client_socket.send(message.encode())
            
            print(f"Real-time fraud detection started...")
            print(f"Alert threshold: {self.alert_threshold:.2f}")
            print(f"Press Ctrl+C to stop\n")
            
            # Start statistics thread
            stats_thread = threading.Thread(target=self.periodic_stats)
            stats_thread.daemon = True
            stats_thread.start()
            
            # Process transactions
            buffer = ""
            
            while True:
                # Receive data
                data = client_socket.recv(4096).decode()
                if not data:
                    break
                
                buffer += data
                
                # Process complete JSON objects
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line:
                        try:
                            transaction = json.loads(line)
                            
                            # Process transaction
                            fraud_prob = self.process_transaction(transaction)
                            
                            # Print summary every 10 transactions
                            if self.stats['total'] % 10 == 0:
                                print(f"Processed {self.stats['total']} transactions, " 
                                      f"{self.stats['flagged']} fraud alerts")
                            
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"Error processing transaction: {e}")
        
        except KeyboardInterrupt:
            print("\n\nStopping fraud detection...")
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            # Print final statistics
            self.print_statistics()
            
            # Save results
            self.save_results()
            
            # Close connection
            try:
                client_socket.close()
            except:
                pass
            
            print("Fraud detection stopped.")
    
    def periodic_stats(self):
        """Periodically print statistics"""
        while True:
            time.sleep(30)  # Print stats every 30 seconds
            self.print_statistics()
    
    def save_results(self):
        """Save detection results to CSV"""
        if not self.results:
            print("No results to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fraud_detection_results_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

# Main execution
if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Real-time Fraud Detection Client')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--model', default='fraud_detector.pth', help='Model file path')
    parser.add_argument('--id', default='FraudDetector', help='Client ID')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file '{args.model}' not found!")
        print("Please train a model first using fraud_model.py")
        exit(1)
    
    # Create and run detector
    detector = RealTimeFraudDetector(
        model_path=args.model,
        host=args.host,
        port=args.port,
        client_id=args.id
    )
    
    # Run real-time detection
    detector.run()
