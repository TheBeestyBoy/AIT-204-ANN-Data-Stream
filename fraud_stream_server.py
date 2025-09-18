import json
import random
import socket
import threading
import time
from datetime import datetime, timedelta
import sys
import math
import numpy as np

class UserProfile:
    def __init__(self, user_id, profile_type):
        self.user_id = user_id
        self.profile_type = profile_type
        self.last_transaction_time = datetime.now()
        self.transaction_count = 0
        self.daily_count = 0
        self.last_reset = datetime.now().date()
        self.favorite_items = []
        self.typical_hours = []
        self.setup_profile()
    
    def setup_profile(self):
        profiles = {
            "high_freq_big_spender": {
                "frequency": 0.9,
                "amount_range": (150, 500),
                "amount_std": 50,
                "peak_hours": [9, 10, 11, 14, 15, 16, 19, 20],
                "item_categories": ["E", "D", "F"],
                "daily_limit": 20,
                "fraud_probability": 0.01
            },
            "low_freq_big_spender": {
                "frequency": 0.3,
                "amount_range": (200, 800),
                "amount_std": 100,
                "peak_hours": [11, 15, 19],
                "item_categories": ["E", "F"],
                "daily_limit": 5,
                "fraud_probability": 0.02
            },
            "high_freq_low_spender": {
                "frequency": 0.8,
                "amount_range": (5, 50),
                "amount_std": 10,
                "peak_hours": [7, 8, 12, 13, 18, 19],
                "item_categories": ["C", "B"],
                "daily_limit": 15,
                "fraud_probability": 0.005
            },
            "periodic_big_spender": {
                "frequency": 0.6,
                "amount_range": (10, 60),
                "amount_std": 15,
                "peak_hours": [12, 18, 19, 20],
                "item_categories": ["B", "C", "D"],
                "daily_limit": 10,
                "fraud_probability": 0.01,
                "big_purchase_interval": 7,
                "big_purchase_range": (300, 600)
            },
            "morning_shopper": {
                "frequency": 0.7,
                "amount_range": (30, 150),
                "amount_std": 25,
                "peak_hours": [6, 7, 8, 9, 10],
                "item_categories": ["C", "D"],
                "daily_limit": 8,
                "fraud_probability": 0.01
            },
            "night_owl": {
                "frequency": 0.6,
                "amount_range": (40, 200),
                "amount_std": 30,
                "peak_hours": [20, 21, 22, 23, 0, 1],
                "item_categories": ["E", "B"],
                "daily_limit": 7,
                "fraud_probability": 0.015
            },
            "weekend_warrior": {
                "frequency": 0.4,
                "amount_range": (100, 400),
                "amount_std": 60,
                "peak_hours": [10, 11, 14, 15, 16],
                "item_categories": ["D", "E", "F"],
                "daily_limit": 12,
                "fraud_probability": 0.02,
                "weekend_multiplier": 3.0
            },
            "budget_conscious": {
                "frequency": 0.5,
                "amount_range": (5, 30),
                "amount_std": 5,
                "peak_hours": [11, 17, 18],
                "item_categories": ["B", "C"],
                "daily_limit": 6,
                "fraud_probability": 0.003
            },
            "impulse_buyer": {
                "frequency": 0.7,
                "amount_range": (20, 250),
                "amount_std": 80,
                "peak_hours": list(range(10, 22)),
                "item_categories": ["B", "C", "D", "E", "F"],
                "daily_limit": 15,
                "fraud_probability": 0.025
            },
            "corporate_card": {
                "frequency": 0.6,
                "amount_range": (50, 300),
                "amount_std": 40,
                "peak_hours": [9, 10, 11, 14, 15, 16],
                "item_categories": ["D", "E", "F"],
                "daily_limit": 10,
                "fraud_probability": 0.02,
                "weekday_only": True
            },
            "student": {
                "frequency": 0.6,
                "amount_range": (5, 40),
                "amount_std": 8,
                "peak_hours": [8, 12, 13, 18, 19, 22],
                "item_categories": ["B", "C"],
                "daily_limit": 8,
                "fraud_probability": 0.01
            },
            "retiree": {
                "frequency": 0.4,
                "amount_range": (30, 120),
                "amount_std": 20,
                "peak_hours": [10, 11, 14, 15],
                "item_categories": ["C", "D"],
                "daily_limit": 5,
                "fraud_probability": 0.008
            },
            "fraud_prone_1": {
                "frequency": 0.5,
                "amount_range": (20, 100),
                "amount_std": 25,
                "peak_hours": list(range(8, 20)),
                "item_categories": ["B", "C", "D"],
                "daily_limit": 8,
                "fraud_probability": 0.15,
                "fraud_patterns": ["sudden_spike", "unusual_time", "rapid_succession"]
            },
            "fraud_prone_2": {
                "frequency": 0.6,
                "amount_range": (40, 180),
                "amount_std": 35,
                "peak_hours": list(range(10, 21)),
                "item_categories": ["C", "D", "E"],
                "daily_limit": 10,
                "fraud_probability": 0.12,
                "fraud_patterns": ["location_jump", "duplicate_amount", "unusual_item"]
            },
            "fraud_prone_3": {
                "frequency": 0.4,
                "amount_range": (30, 150),
                "amount_std": 30,
                "peak_hours": list(range(9, 19)),
                "item_categories": ["B", "C", "D", "E"],
                "daily_limit": 7,
                "fraud_probability": 0.10,
                "fraud_patterns": ["midnight_purchase", "round_numbers", "category_switch"]
            },
            "consistent_regular": {
                "frequency": 0.5,
                "amount_range": (45, 55),
                "amount_std": 3,
                "peak_hours": [12, 18],
                "item_categories": ["C"],
                "daily_limit": 2,
                "fraud_probability": 0.005
            },
            "seasonal_shopper": {
                "frequency": 0.3,
                "amount_range": (80, 300),
                "amount_std": 50,
                "peak_hours": [14, 15, 16, 17],
                "item_categories": ["D", "E", "F"],
                "daily_limit": 6,
                "fraud_probability": 0.02,
                "seasonal_multiplier": 2.5
            },
            "micro_transactor": {
                "frequency": 0.9,
                "amount_range": (1, 10),
                "amount_std": 2,
                "peak_hours": list(range(6, 23)),
                "item_categories": ["B"],
                "daily_limit": 30,
                "fraud_probability": 0.008
            },
            "luxury_buyer": {
                "frequency": 0.2,
                "amount_range": (500, 2000),
                "amount_std": 300,
                "peak_hours": [14, 15, 16],
                "item_categories": ["F"],
                "daily_limit": 2,
                "fraud_probability": 0.03
            },
            "erratic_spender": {
                "frequency": 0.5,
                "amount_range": (5, 500),
                "amount_std": 150,
                "peak_hours": list(range(0, 24)),
                "item_categories": ["B", "C", "D", "E", "F"],
                "daily_limit": 10,
                "fraud_probability": 0.04
            }
        }
        
        self.params = profiles[self.profile_type]
        self.favorite_items = [f"{cat}{random.randint(1000, 9999)}" 
                              for cat in self.params["item_categories"] 
                              for _ in range(random.randint(2, 5))]
    
    def should_transact(self, current_hour, current_minute):
        if datetime.now().date() != self.last_reset:
            self.daily_count = 0
            self.last_reset = datetime.now().date()
        
        if self.daily_count >= self.params["daily_limit"]:
            return False
        
        if "weekday_only" in self.params and self.params["weekday_only"]:
            if datetime.now().weekday() >= 5:
                return False
        
        base_prob = self.params["frequency"]
        
        if current_hour in self.params["peak_hours"]:
            base_prob *= 1.5
        else:
            base_prob *= 0.3
        
        if "weekend_multiplier" in self.params and datetime.now().weekday() >= 5:
            base_prob *= self.params["weekend_multiplier"]
        
        time_since_last = (datetime.now() - self.last_transaction_time).seconds
        if time_since_last < 60:
            base_prob *= 0.1
        elif time_since_last < 300:
            base_prob *= 0.5
        
        return random.random() < (base_prob / 60)
    
    def generate_transaction(self, transaction_id):
        self.transaction_count += 1
        self.daily_count += 1
        self.last_transaction_time = datetime.now()
        
        is_fraud = random.random() < self.params["fraud_probability"]
        
        if is_fraud and "fraud_patterns" in self.params:
            return self.generate_fraud_transaction(transaction_id)
        
        if "big_purchase_interval" in self.params:
            if self.transaction_count % self.params["big_purchase_interval"] == 0:
                amount = random.uniform(*self.params["big_purchase_range"])
            else:
                amount = max(5, random.gauss(
                    sum(self.params["amount_range"]) / 2,
                    self.params["amount_std"]
                ))
        else:
            mean = sum(self.params["amount_range"]) / 2
            amount = max(self.params["amount_range"][0], 
                        min(self.params["amount_range"][1],
                            random.gauss(mean, self.params["amount_std"])))
        
        if random.random() < 0.7:
            item_id = random.choice(self.favorite_items)
        else:
            cat = random.choice(self.params["item_categories"])
            item_id = f"{cat}{random.randint(1000, 9999)}"
        
        return {
            "transactionID": transaction_id,
            "userID": self.user_id,
            "amount": round(amount, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "itemID": item_id,
            "merchantID": f"M{random.randint(100, 999)}",
            "isWeekend": datetime.now().weekday() >= 5,
            "hourOfDay": datetime.now().hour,
            "daysSinceLastTransaction": (datetime.now() - self.last_transaction_time).days,
            "userProfile": self.profile_type,
            "isFraud": False
        }
    
    def generate_fraud_transaction(self, transaction_id):
        pattern = random.choice(self.params["fraud_patterns"])
        
        if pattern == "sudden_spike":
            amount = random.uniform(
                self.params["amount_range"][1] * 3,
                self.params["amount_range"][1] * 5
            )
        elif pattern == "unusual_time":
            amount = random.uniform(*self.params["amount_range"])
        elif pattern == "rapid_succession":
            amount = random.uniform(*self.params["amount_range"]) * 1.5
        elif pattern == "duplicate_amount":
            amount = round(random.uniform(*self.params["amount_range"]), 0)
        elif pattern == "round_numbers":
            amount = round(random.uniform(100, 500), -2)
        elif pattern == "midnight_purchase":
            amount = random.uniform(*self.params["amount_range"]) * 2
        else:
            amount = random.uniform(
                self.params["amount_range"][1] * 1.5,
                self.params["amount_range"][1] * 3
            )
        
        unusual_categories = [c for c in ["B", "C", "D", "E", "F"] 
                            if c not in self.params["item_categories"]]
        if unusual_categories and pattern in ["unusual_item", "category_switch"]:
            item_id = f"{random.choice(unusual_categories)}{random.randint(1000, 9999)}"
        else:
            item_id = f"{random.choice(['E', 'F'])}{random.randint(8000, 9999)}"
        
        hour = datetime.now().hour
        if pattern == "unusual_time":
            unusual_hours = [h for h in range(24) if h not in self.params["peak_hours"]]
            if unusual_hours:
                hour = random.choice(unusual_hours)
        elif pattern == "midnight_purchase":
            hour = random.choice([0, 1, 2, 3, 4])
        
        return {
            "transactionID": transaction_id,
            "userID": self.user_id,
            "amount": round(amount, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "itemID": item_id,
            "merchantID": f"M{random.randint(900, 999)}",
            "isWeekend": datetime.now().weekday() >= 5,
            "hourOfDay": hour,
            "daysSinceLastTransaction": 0 if pattern == "rapid_succession" else (datetime.now() - self.last_transaction_time).days,
            "userProfile": self.profile_type,
            "isFraud": True,
            "fraudPattern": pattern
        }

class FraudStreamServer:
    def __init__(self, host='localhost', port=5555, interval=1.0):
        self.host = host
        self.port = port
        self.interval = interval
        self.clients = []
        self.lock = threading.Lock()
        self.running = False
        self.transaction_id_counter = 0
        
        profile_types = [
            "high_freq_big_spender", "low_freq_big_spender", "high_freq_low_spender",
            "periodic_big_spender", "morning_shopper", "night_owl", "weekend_warrior",
            "budget_conscious", "impulse_buyer", "corporate_card", "student", "retiree",
            "fraud_prone_1", "fraud_prone_2", "fraud_prone_3", "consistent_regular",
            "seasonal_shopper", "micro_transactor", "luxury_buyer", "erratic_spender"
        ]
        
        self.users = {}
        for i in range(1, 26):
            profile = profile_types[(i-1) % len(profile_types)]
            self.users[i] = UserProfile(i, profile)
            print(f"User {i}: {profile}")
    
    def generate_transaction(self):
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        
        active_users = [user for user in self.users.values() 
                       if user.should_transact(current_hour, current_minute)]
        
        if not active_users:
            return None
        
        user = random.choice(active_users)
        self.transaction_id_counter += 1
        return user.generate_transaction(self.transaction_id_counter)
    
    def broadcast_to_clients(self, message):
        if message is None:
            return
            
        disconnected_clients = []
        data = (json.dumps(message) + '\n').encode('utf-8')
        
        with self.lock:
            for client_socket, client_addr in self.clients:
                try:
                    client_socket.send(data)
                except (socket.error, BrokenPipeError):
                    disconnected_clients.append((client_socket, client_addr))
            
            for client in disconnected_clients:
                self.clients.remove(client)
                client[0].close()
                print(f"Client {client[1]} disconnected")
    
    def data_generator_thread(self):
        print(f"Fraud-aware data generator started - broadcasting every {self.interval} seconds")
        fraud_count = 0
        total_count = 0
        
        while self.running:
            transaction = self.generate_transaction()
            if transaction:
                total_count += 1
                if transaction.get("isFraud", False):
                    fraud_count += 1
                    fraud_indicator = "🚨 FRAUD" if transaction["isFraud"] else ""
                    print(f"Transaction #{transaction['transactionID']}: User {transaction['userID']} ({transaction['userProfile']}) - ${transaction['amount']:.2f} {fraud_indicator}")
                elif total_count % 10 == 0:
                    print(f"Processed {total_count} transactions ({fraud_count} fraudulent)")
                
                self.broadcast_to_clients(transaction)
            time.sleep(self.interval)
    
    def handle_client_connection(self, client_socket, client_addr):
        print(f"New client connected: {client_addr}")
        with self.lock:
            self.clients.append((client_socket, client_addr))
        
        welcome_message = {
            "type": "connection",
            "message": "Connected to fraud detection training stream",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "totalUsers": len(self.users),
            "profileTypes": list(set(u.profile_type for u in self.users.values()))
        }
        try:
            client_socket.send((json.dumps(welcome_message) + '\n').encode('utf-8'))
        except:
            pass
    
    def start(self):
        self.running = True
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            print(f"Fraud Detection Stream Server listening on {self.host}:{self.port}")
            print(f"Total user profiles: {len(self.users)}")
            print(f"Fraud-prone users: 3")
            print(f"Waiting for client connections...")
            
            generator_thread = threading.Thread(target=self.data_generator_thread, daemon=True)
            generator_thread.start()
            
            while self.running:
                try:
                    client_socket, client_addr = server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client_connection,
                        args=(client_socket, client_addr),
                        daemon=True
                    )
                    client_thread.start()
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error accepting connection: {e}")
                    
        except KeyboardInterrupt:
            print("\nShutting down server...")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.running = False
            with self.lock:
                for client_socket, _ in self.clients:
                    client_socket.close()
            server_socket.close()
            print("Server stopped")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fraud Detection Training Stream Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=5555, help='Port to bind to (default: 5555)')
    parser.add_argument('--interval', type=float, default=0.5, help='Base interval between transactions in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    server = FraudStreamServer(host=args.host, port=args.port, interval=args.interval)
    
    print("=" * 60)
    print("FRAUD DETECTION TRAINING STREAM SERVER")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Base Interval: {args.interval} seconds")
    print("=" * 60)
    print("\nUser Profiles Distribution:")
    print("- 17 Normal users with various spending patterns")
    print("- 3 Fraud-prone users (15%, 12%, 10% fraud probability)")
    print("- Each user has consistent behavioral patterns")
    print("=" * 60)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nServer shutdown complete")

if __name__ == "__main__":
    main()