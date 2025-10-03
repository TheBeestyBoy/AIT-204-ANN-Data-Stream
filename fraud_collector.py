import json
import socket
import pandas as pd
from datetime import datetime
import pickle
import time
import numpy as np

class FraudDataCollector:
    def __init__(self, host='localhost', port=5555, collection_size=1000):
        """
        Args:
            host: 'localhost' for your own server
                  or IP address like '192.168.1.10' for shared server
            port: Usually 5555
            collection_size: Number of transactions to collect
        """
        self.host = host
        self.port = port
        self.collection_size = collection_size
        self.transactions = []
    
    def connect_and_collect(self):
        """Connect to server and collect transactions"""
        print(f"Connecting to {self.host}:{self.port}...")
        
        try:
            # Create socket connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.host, self.port))
            print(f"Connected! Collecting {self.collection_size} transactions...")
            
            # Send initial handshake
            client_id = {'clientID': 'FraudCollector', 'clientType': 'collector'}
            message = json.dumps(client_id) + '\n'
            client_socket.send(message.encode())
            
            # Collect transactions
            buffer = ""
            collected = 0
            
            while collected < self.collection_size:
                # Receive data
                data = client_socket.recv(4096).decode()
                buffer += data
                
                # Process complete JSON objects
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line:
                        try:
                            transaction = json.loads(line)
                            # Extract features and store
                            features = self.extract_features(transaction)
                            features['isFraud'] = transaction.get('isFraud', False)
                            features['fraudPattern'] = transaction.get('fraudPattern', 'none')
                            features['transactionID'] = transaction.get('transactionID', '')
                            features['userID'] = transaction.get('userID', '')
                            features['merchantID'] = transaction.get('merchantID', '')
                            self.transactions.append(features)
                            collected += 1
                            
                            if collected % 100 == 0:
                                print(f"Collected {collected}/{self.collection_size} transactions")
                        except json.JSONDecodeError:
                            continue
            
            # Save to CSV
            self.save_to_csv()
            
            # Close connection
            client_socket.close()
            print(f"Collection complete! Saved {collected} transactions.")
            
        except Exception as e:
            print(f"Error during collection: {e}")
            
    def extract_features(self, transaction):
        """Extract features for ML model"""
        features = {
            'amount': transaction.get('amount', 0),
            'hour': transaction.get('hourOfDay', 0),
            'isWeekend': int(transaction.get('isWeekend', False)),
            'daysSinceLastTransaction': transaction.get('daysSinceLastTransaction', 0),
            'merchantCategory': self.encode_merchant_category(transaction.get('merchantID', '')),
            'userActivityLevel': self.calculate_user_activity(transaction.get('userID', '')),
            'amountVelocity': transaction.get('amount', 0) / max(transaction.get('daysSinceLastTransaction', 1), 0.1),
            'timeOfDayRisk': self.calculate_time_risk(transaction.get('hourOfDay', 0)),
            'merchantRiskScore': self.calculate_merchant_risk(transaction.get('merchantID', '')),
            'isRoundAmount': int(transaction.get('amount', 0) % 10 == 0),
        }
        return features
    
    def encode_merchant_category(self, merchant_id):
        """Encode merchant ID to category (simplified)"""
        if not merchant_id:
            return 0
        # Simple hash-based categorization
        return hash(merchant_id) % 10
    
    def calculate_user_activity(self, user_id):
        """Calculate user activity level (simplified)"""
        if not user_id:
            return 1
        # Simulate activity level based on user ID
        return (hash(user_id) % 5) + 1
    
    def calculate_time_risk(self, hour):
        """Calculate risk based on time of day"""
        # Higher risk during late night hours
        if 2 <= hour <= 6:
            return 3  # High risk
        elif 22 <= hour or hour < 2:
            return 2  # Medium risk
        else:
            return 1  # Low risk
    
    def calculate_merchant_risk(self, merchant_id):
        """Calculate merchant risk score"""
        if not merchant_id:
            return 1
        # Simulate risk based on merchant patterns
        risk_hash = hash(merchant_id) % 100
        if risk_hash < 5:
            return 3  # High risk merchant
        elif risk_hash < 20:
            return 2  # Medium risk
        else:
            return 1  # Low risk
    
    def save_to_csv(self):
        """Save collected transactions to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fraud_transactions_{timestamp}.csv"
        
        df = pd.DataFrame(self.transactions)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
        # Also save as pickle for faster loading
        with open(f"fraud_transactions_{timestamp}.pkl", 'wb') as f:
            pickle.dump(self.transactions, f)
        
        return filename
    
    def analyze_collected_data(self):
        """Perform exploratory data analysis"""
        df = pd.DataFrame(self.transactions)
        
        print("\n=== Data Analysis ===")
        print(f"Total transactions: {len(df)}")
        print(f"Fraud rate: {df['isFraud'].mean():.2%}")
        
        print("\n=== Fraud vs Normal Statistics ===")
        fraud_df = df[df['isFraud'] == True]
        normal_df = df[df['isFraud'] == False]
        
        print(f"Average amount (Fraud): ${fraud_df['amount'].mean():.2f}")
        print(f"Average amount (Normal): ${normal_df['amount'].mean():.2f}")
        
        print(f"Most common fraud hour: {fraud_df['hour'].mode().values[0] if len(fraud_df) > 0 else 'N/A'}")
        print(f"Weekend fraud rate: {fraud_df['isWeekend'].mean():.2%}" if len(fraud_df) > 0 else "N/A")
        
        return df

# Main execution
if __name__ == "__main__":
    # Create collector
    collector = FraudDataCollector(
        host='localhost',  # Change to instructor IP for class-wide
        port=5555,
        collection_size=1000
    )
    
    # Collect data
    collector.connect_and_collect()
    
    # Analyze collected data
    df = collector.analyze_collected_data()
    
    print("\n✅ Data collection complete!")
