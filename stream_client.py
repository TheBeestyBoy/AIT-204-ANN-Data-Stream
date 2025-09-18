import json
import socket
import sys
from collections import deque
from datetime import datetime
import statistics
import threading
import time

class StreamStatistics:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.amounts = deque(maxlen=window_size)
        self.user_ids = deque(maxlen=window_size)
        self.transaction_count = 0
        self.total_amount = 0.0
        self.unique_users = set()
        self.unique_items = set()
        self.lock = threading.Lock()
        
    def update(self, transaction):
        with self.lock:
            self.transaction_count += 1
            amount = transaction['amount']
            self.amounts.append(amount)
            self.user_ids.append(transaction['userID'])
            self.total_amount += amount
            self.unique_users.add(transaction['userID'])
            self.unique_items.add(transaction['itemID'])
        
    def get_statistics(self):
        with self.lock:
            if not self.amounts:
                return None
                
            stats = {
                'total_transactions': self.transaction_count,
                'total_amount': round(self.total_amount, 2),
                'avg_amount_overall': round(self.total_amount / self.transaction_count, 2) if self.transaction_count > 0 else 0,
                'window_statistics': {
                    'size': len(self.amounts),
                    'mean': round(statistics.mean(self.amounts), 2),
                    'median': round(statistics.median(self.amounts), 2),
                    'min': round(min(self.amounts), 2),
                    'max': round(max(self.amounts), 2),
                    'std_dev': round(statistics.stdev(self.amounts), 2) if len(self.amounts) > 1 else 0
                },
                'unique_users': len(self.unique_users),
                'unique_items': len(self.unique_items),
                'active_users_in_window': len(set(self.user_ids))
            }
            return stats

class DataStreamClient:
    def __init__(self, host='localhost', port=5555, client_id=None):
        self.host = host
        self.port = port
        self.client_id = client_id or f"Client-{datetime.now().strftime('%H%M%S')}"
        self.stats = StreamStatistics(window_size=10)
        self.running = False
        
    def print_separator(self):
        print("-" * 80)
    
    def format_transaction(self, transaction):
        return (f"Transaction #{transaction['transactionID']} | "
                f"User: {transaction['userID']} | "
                f"Amount: ${transaction['amount']:.2f} | "
                f"Item: {transaction['itemID']} | "
                f"Time: {transaction['timestamp']}")
    
    def format_statistics(self, stats):
        if not stats:
            return "No statistics available yet"
        
        lines = [
            "\n📊 STATISTICS:",
            f"Total Transactions: {stats['total_transactions']}",
            f"Total Amount: ${stats['total_amount']:.2f}",
            f"Overall Average: ${stats['avg_amount_overall']:.2f}",
            f"Unique Users: {stats['unique_users']}",
            f"Unique Items: {stats['unique_items']}",
            f"\n📈 Last {stats['window_statistics']['size']} Transactions:",
            f"  Mean: ${stats['window_statistics']['mean']:.2f}",
            f"  Median: ${stats['window_statistics']['median']:.2f}",
            f"  Min: ${stats['window_statistics']['min']:.2f}",
            f"  Max: ${stats['window_statistics']['max']:.2f}",
            f"  Std Dev: ${stats['window_statistics']['std_dev']:.2f}",
            f"  Active Users: {stats['active_users_in_window']}"
        ]
        return "\n".join(lines)
    
    def connect_and_stream(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            print(f"Connecting to {self.host}:{self.port}...")
            client_socket.connect((self.host, self.port))
            print(f"Connected as {self.client_id}")
            print("Receiving data stream. Press Ctrl+C to stop\n")
            
            self.running = True
            buffer = ""
            
            while self.running:
                try:
                    data = client_socket.recv(4096).decode('utf-8')
                    if not data:
                        print("Connection closed by server")
                        break
                    
                    buffer += data
                    lines = buffer.split('\n')
                    buffer = lines[-1]
                    
                    for line in lines[:-1]:
                        if line.strip():
                            try:
                                message = json.loads(line)
                                
                                if message.get('type') == 'connection':
                                    print(f"[{self.client_id}] {message.get('message', 'Connected')}")
                                    print()
                                else:
                                    self.print_separator()
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {self.client_id} - New Transaction:")
                                    print(self.format_transaction(message))
                                    
                                    self.stats.update(message)
                                    
                                    statistics_data = self.stats.get_statistics()
                                    print(self.format_statistics(statistics_data))
                                    print()
                                    
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON: {e}", file=sys.stderr)
                            except KeyError as e:
                                if 'type' not in str(e):
                                    print(f"Missing required field in transaction: {e}", file=sys.stderr)
                
                except socket.timeout:
                    continue
                except socket.error as e:
                    print(f"Socket error: {e}")
                    break
                    
        except ConnectionRefusedError:
            print(f"Error: Could not connect to server at {self.host}:{self.port}")
            print("Make sure the server is running.")
        except KeyboardInterrupt:
            print("\n\nDisconnecting from stream...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.running = False
            client_socket.close()
            
            print(f"\n{self.client_id} disconnected.")
            final_stats = self.stats.get_statistics()
            if final_stats:
                print("\n🏁 FINAL STATISTICS:")
                print(self.format_statistics(final_stats))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Data Stream Client')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5555, help='Server port (default: 5555)')
    parser.add_argument('--id', dest='client_id', help='Client ID (default: auto-generated)')
    
    args = parser.parse_args()
    
    client = DataStreamClient(host=args.host, port=args.port, client_id=args.client_id)
    
    print("=" * 50)
    print("DATA STREAM CLIENT")
    print("=" * 50)
    print(f"Client ID: {client.client_id}")
    print(f"Server: {args.host}:{args.port}")
    print("=" * 50)
    
    try:
        client.connect_and_stream()
    except KeyboardInterrupt:
        print("\nClient shutdown complete")

if __name__ == "__main__":
    main()