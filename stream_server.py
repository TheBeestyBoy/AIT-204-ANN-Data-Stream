import json
import random
import socket
import threading
import time
from datetime import datetime, timedelta
import sys

class DataStreamServer:
    def __init__(self, host='localhost', port=5555, interval=1.0):
        self.host = host
        self.port = port
        self.interval = interval
        self.clients = []
        self.lock = threading.Lock()
        self.running = False
        self.transaction_id_counter = 0
        
    def generate_transaction(self):
        self.transaction_id_counter += 1
        transaction = {
            "transactionID": self.transaction_id_counter,
            "userID": random.randint(1, 100),
            "amount": round(random.uniform(5.0, 500.0), 2),
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "itemID": f"{random.choice(['E', 'C', 'B', 'D', 'F'])}{random.randint(1000, 9999)}"
        }
        return transaction
    
    def broadcast_to_clients(self, message):
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
        print(f"Data generator started - broadcasting every {self.interval} seconds")
        while self.running:
            transaction = self.generate_transaction()
            print(f"Broadcasting: Transaction #{transaction['transactionID']} - ${transaction['amount']:.2f}")
            self.broadcast_to_clients(transaction)
            time.sleep(self.interval)
    
    def handle_client_connection(self, client_socket, client_addr):
        print(f"New client connected: {client_addr}")
        with self.lock:
            self.clients.append((client_socket, client_addr))
        
        welcome_message = {
            "type": "connection",
            "message": "Connected to data stream server",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
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
            print(f"Server listening on {self.host}:{self.port}")
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
    parser = argparse.ArgumentParser(description='Data Stream Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=5555, help='Port to bind to (default: 5555)')
    parser.add_argument('--interval', type=float, default=1.0, help='Interval between transactions in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    server = DataStreamServer(host=args.host, port=args.port, interval=args.interval)
    
    print("=" * 50)
    print("DATA STREAM SERVER")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Broadcast Interval: {args.interval} seconds")
    print("=" * 50)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nServer shutdown complete")

if __name__ == "__main__":
    main()