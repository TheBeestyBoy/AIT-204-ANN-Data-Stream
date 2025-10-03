import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import hashlib

class TransactionFeatures:
    def __init__(self):
        self.scaler = StandardScaler()
        self.user_stats = defaultdict(lambda: {'count': 0, 'total': 0, 'mean': 0, 'std': 1})
        self.merchant_stats = defaultdict(lambda: {'fraud_count': 0, 'total_count': 0})
        self.fitted = False
        
        self.feature_names = [
            'amount',
            'hour', 
            'isWeekend',
            'daysSinceLastTransaction',
            'amount_zscore',  # Amount deviation from user's mean
            'time_since_midnight',  # Seconds since midnight
            'merchant_risk_score',  # Based on merchant ID pattern
            'user_velocity',  # Transaction frequency
            'amount_percentile',  # Amount relative to all transactions
            'hour_risk',  # Risk score based on hour
            'is_round_amount',  # Is amount a round number
            'day_of_week',  # 0-6
            'merchant_category',  # Encoded merchant category
            'user_trust_score'  # User's historical fraud rate
        ]
    
    def fit(self, transactions_df):
        """Fit the scaler and collect statistics from training data"""
        # Fit scaler on amount
        self.scaler.fit(transactions_df[['amount']])
        
        # Calculate user statistics
        for _, row in transactions_df.iterrows():
            user_id = row.get('userID', '')
            merchant_id = row.get('merchantID', '')
            
            if user_id:
                self.user_stats[user_id]['count'] += 1
                self.user_stats[user_id]['total'] += row['amount']
                self.user_stats[user_id]['mean'] = self.user_stats[user_id]['total'] / self.user_stats[user_id]['count']
                
                if row.get('isFraud', False):
                    self.user_stats[user_id].setdefault('fraud_count', 0)
                    self.user_stats[user_id]['fraud_count'] += 1
            
            if merchant_id:
                self.merchant_stats[merchant_id]['total_count'] += 1
                if row.get('isFraud', False):
                    self.merchant_stats[merchant_id]['fraud_count'] += 1
        
        # Calculate amount percentiles
        self.amount_percentiles = np.percentile(transactions_df['amount'], [25, 50, 75, 90, 95])
        
        self.fitted = True
    
    def transform_transaction(self, transaction, user_history=None):
        """Convert transaction to feature vector"""
        features = []
        
        # Basic features
        amount = transaction.get('amount', 0)
        hour = transaction.get('hourOfDay', 0)
        is_weekend = int(transaction.get('isWeekend', False))
        days_since_last = transaction.get('daysSinceLastTransaction', 0)
        
        features.append(amount)
        features.append(hour)
        features.append(is_weekend)
        features.append(days_since_last)
        
        # Amount Z-score (deviation from user's typical amount)
        user_id = transaction.get('userID', '')
        if user_id in self.user_stats and self.user_stats[user_id]['count'] > 0:
            user_mean = self.user_stats[user_id]['mean']
            amount_zscore = (amount - user_mean) / max(user_mean * 0.5, 1)  # Normalize
        else:
            amount_zscore = 0
        features.append(amount_zscore)
        
        # Time since midnight in hours
        time_since_midnight = hour
        features.append(time_since_midnight)
        
        # Merchant risk score
        merchant_id = transaction.get('merchantID', '')
        merchant_risk = self.calculate_merchant_risk_score(merchant_id)
        features.append(merchant_risk)
        
        # User velocity (transactions per day)
        if days_since_last > 0:
            user_velocity = 1.0 / days_since_last
        else:
            user_velocity = 1.0  # Same day transaction
        features.append(user_velocity)
        
        # Amount percentile
        amount_percentile = self.get_amount_percentile(amount)
        features.append(amount_percentile)
        
        # Hour risk (late night/early morning are higher risk)
        hour_risk = self.calculate_hour_risk(hour)
        features.append(hour_risk)
        
        # Is round amount (potential indicator of fraud)
        is_round_amount = int(amount % 10 == 0 or amount % 100 == 0)
        features.append(is_round_amount)
        
        # Day of week (0-6, Monday=0)
        # Simulated since we don't have actual date
        day_of_week = hash(transaction.get('transactionID', '')) % 7
        features.append(day_of_week)
        
        # Merchant category (encoded)
        merchant_category = self.encode_merchant_category(merchant_id)
        features.append(merchant_category)
        
        # User trust score (inverse of historical fraud rate)
        user_trust = self.calculate_user_trust_score(user_id)
        features.append(user_trust)
        
        return np.array(features, dtype=np.float32)
    
    def calculate_merchant_risk_score(self, merchant_id):
        """Calculate risk score for merchant based on historical fraud"""
        if not merchant_id or merchant_id not in self.merchant_stats:
            return 0.5  # Unknown merchant, medium risk
        
        stats = self.merchant_stats[merchant_id]
        if stats['total_count'] == 0:
            return 0.5
        
        fraud_rate = stats['fraud_count'] / stats['total_count']
        return min(fraud_rate * 5, 1.0)  # Scale to 0-1
    
    def get_amount_percentile(self, amount):
        """Get percentile rank of amount"""
        if not self.fitted:
            return 0.5
        
        if amount <= self.amount_percentiles[0]:
            return 0.25
        elif amount <= self.amount_percentiles[1]:
            return 0.5
        elif amount <= self.amount_percentiles[2]:
            return 0.75
        elif amount <= self.amount_percentiles[3]:
            return 0.9
        else:
            return 0.95
    
    def calculate_hour_risk(self, hour):
        """Calculate risk based on hour of day"""
        # Peak fraud hours: 2-6 AM
        if 2 <= hour <= 6:
            return 0.9
        # Elevated risk: 10 PM - 2 AM
        elif hour >= 22 or hour < 2:
            return 0.7
        # Medium risk: early morning and late evening
        elif 6 <= hour <= 8 or 20 <= hour <= 22:
            return 0.5
        # Lower risk: business hours
        else:
            return 0.3
    
    def encode_merchant_category(self, merchant_id):
        """Encode merchant ID to category number"""
        if not merchant_id:
            return 0
        
        # Use hash to create consistent category assignment
        hash_obj = hashlib.md5(merchant_id.encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        return (hash_int % 20) / 20.0  # Normalize to 0-1
    
    def calculate_user_trust_score(self, user_id):
        """Calculate user trust score based on historical behavior"""
        if not user_id or user_id not in self.user_stats:
            return 0.5  # Unknown user
        
        stats = self.user_stats[user_id]
        if stats['count'] == 0:
            return 0.5
        
        # Trust score is inverse of fraud rate
        fraud_rate = stats.get('fraud_count', 0) / stats['count']
        trust_score = 1.0 - fraud_rate
        
        # Adjust for user activity (more active users are slightly more trusted)
        activity_bonus = min(stats['count'] / 100, 0.1)
        
        return min(trust_score + activity_bonus, 1.0)
    
    def transform_batch(self, transactions_list):
        """Transform a batch of transactions"""
        features = []
        for transaction in transactions_list:
            features.append(self.transform_transaction(transaction))
        return np.array(features, dtype=np.float32)
    
    def get_feature_importance(self, model=None):
        """Get feature importance if model provides it"""
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            # Default importance based on domain knowledge
            if 'amount' in name:
                importance_dict[name] = 0.9
            elif 'merchant' in name:
                importance_dict[name] = 0.8
            elif 'hour' in name or 'time' in name:
                importance_dict[name] = 0.7
            elif 'user' in name:
                importance_dict[name] = 0.6
            else:
                importance_dict[name] = 0.5
        
        return importance_dict
