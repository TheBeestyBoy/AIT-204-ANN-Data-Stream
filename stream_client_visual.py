import json
import socket
import threading
from collections import deque
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

class DataStreamVisualClient:
    def __init__(self, host='localhost', port=5555, client_id=None, max_records=100):
        self.host = host
        self.port = port
        self.client_id = client_id or f"Client-{datetime.now().strftime('%H%M%S')}"
        self.max_records = max_records
        
        self.transactions = deque(maxlen=max_records)
        self.lock = threading.Lock()
        self.running = False
        self.socket = None
        
        self.timestamps = deque(maxlen=max_records)
        self.amounts = deque(maxlen=max_records)
        self.user_counts = {}
        self.item_counts = {}
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'Data Stream Monitor - {self.client_id}', fontsize=16, fontweight='bold')
        
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        self.ax_table = self.fig.add_subplot(gs[0:2, :])
        self.ax_timeseries = self.fig.add_subplot(gs[2, 0])
        self.ax_histogram = self.fig.add_subplot(gs[2, 1])
        self.ax_stats = self.fig.add_subplot(gs[2, 2])
        
        self.ax_table.axis('tight')
        self.ax_table.axis('off')
        
        self.table = None
        self.stats_text = None
        
    def connect_to_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            print(f"Connected as {self.client_id}")
            self.running = True
            return True
        except ConnectionRefusedError:
            print(f"Error: Could not connect to server at {self.host}:{self.port}")
            return False
    
    def receive_data(self):
        buffer = ""
        while self.running:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]
                
                for line in lines[:-1]:
                    if line.strip():
                        try:
                            message = json.loads(line)
                            if message.get('type') != 'connection':
                                self.process_transaction(message)
                        except json.JSONDecodeError:
                            pass
            except socket.error:
                break
        self.running = False
    
    def process_transaction(self, transaction):
        with self.lock:
            self.transactions.append({
                'ID': transaction['transactionID'],
                'User': transaction['userID'],
                'Amount': f"${transaction['amount']:.2f}",
                'Item': transaction['itemID'],
                'Time': transaction['timestamp'].split('T')[1]
            })
            
            self.timestamps.append(datetime.strptime(transaction['timestamp'], '%Y-%m-%dT%H:%M:%S'))
            self.amounts.append(transaction['amount'])
            
            user_id = transaction['userID']
            self.user_counts[user_id] = self.user_counts.get(user_id, 0) + 1
            
            item_id = transaction['itemID']
            self.item_counts[item_id] = self.item_counts.get(item_id, 0) + 1
    
    def update_visualizations(self, frame):
        with self.lock:
            if not self.transactions:
                return
            
            self.ax_table.clear()
            self.ax_table.axis('tight')
            self.ax_table.axis('off')
            
            display_data = list(self.transactions)[-20:]
            
            if display_data:
                table_data = []
                colors = []
                for i, trans in enumerate(display_data):
                    table_data.append([
                        trans['ID'],
                        trans['User'],
                        trans['Amount'],
                        trans['Item'],
                        trans['Time']
                    ])
                    if i == len(display_data) - 1:
                        colors.append(['#00ff00'] * 5)
                    else:
                        colors.append(['#ffffff'] * 5)
                
                self.table = self.ax_table.table(
                    cellText=table_data,
                    colLabels=['Transaction ID', 'User ID', 'Amount', 'Item ID', 'Timestamp'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.25],
                    cellColours=[[c if c == '#00ff00' else '#1a1a1a' for c in row] for row in colors]
                )
                self.table.auto_set_font_size(False)
                self.table.set_fontsize(9)
                self.ax_table.set_title(f'Transaction Stream (Last 20 of {len(self.transactions)} transactions)', 
                                       fontweight='bold', pad=10)
            
            if len(self.amounts) > 1:
                self.ax_timeseries.clear()
                self.ax_timeseries.plot(range(len(self.amounts)), list(self.amounts), 
                                       color='cyan', linewidth=2, marker='o', markersize=3)
                self.ax_timeseries.fill_between(range(len(self.amounts)), list(self.amounts), 
                                               alpha=0.3, color='cyan')
                self.ax_timeseries.set_xlabel('Transaction Index')
                self.ax_timeseries.set_ylabel('Amount ($)')
                self.ax_timeseries.set_title('Amount Time Series', fontweight='bold')
                self.ax_timeseries.grid(True, alpha=0.3)
                
                if len(self.amounts) > 0:
                    self.ax_timeseries.axhline(y=np.mean(list(self.amounts)), 
                                              color='red', linestyle='--', alpha=0.7, 
                                              label=f'Mean: ${np.mean(list(self.amounts)):.2f}')
                    self.ax_timeseries.legend(loc='upper right')
            
            if len(self.amounts) > 0:
                self.ax_histogram.clear()
                n, bins, patches = self.ax_histogram.hist(list(self.amounts), bins=15, 
                                                          color='magenta', alpha=0.7, edgecolor='white')
                
                for i, patch in enumerate(patches):
                    height = patch.get_height()
                    if height > 0:
                        patch.set_facecolor(plt.cm.plasma(i / len(patches)))
                
                self.ax_histogram.set_xlabel('Amount ($)')
                self.ax_histogram.set_ylabel('Frequency')
                self.ax_histogram.set_title('Amount Distribution', fontweight='bold')
                self.ax_histogram.grid(True, alpha=0.3, axis='y')
            
            self.ax_stats.clear()
            self.ax_stats.axis('off')
            
            if len(self.amounts) > 0:
                stats_info = [
                    f"📊 STATISTICS",
                    f"",
                    f"Total Transactions: {len(self.transactions)}",
                    f"Total Amount: ${sum(self.amounts):.2f}",
                    f"",
                    f"Mean Amount: ${np.mean(list(self.amounts)):.2f}",
                    f"Median Amount: ${np.median(list(self.amounts)):.2f}",
                    f"Std Dev: ${np.std(list(self.amounts)):.2f}",
                    f"Min Amount: ${min(self.amounts):.2f}",
                    f"Max Amount: ${max(self.amounts):.2f}",
                    f"",
                    f"Unique Users: {len(self.user_counts)}",
                    f"Unique Items: {len(self.item_counts)}",
                    f"",
                    f"Top User: {max(self.user_counts, key=self.user_counts.get) if self.user_counts else 'N/A'}",
                    f"Top Item: {max(self.item_counts, key=self.item_counts.get) if self.item_counts else 'N/A'}"
                ]
                
                y_pos = 0.95
                for line in stats_info:
                    if line.startswith("📊"):
                        self.ax_stats.text(0.5, y_pos, line, fontsize=12, fontweight='bold',
                                         ha='center', transform=self.ax_stats.transAxes)
                    elif line == "":
                        y_pos -= 0.03
                    else:
                        self.ax_stats.text(0.1, y_pos, line, fontsize=10,
                                         ha='left', transform=self.ax_stats.transAxes)
                    y_pos -= 0.055
                
                rect = Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=2, 
                               edgecolor='cyan', facecolor='none', 
                               transform=self.ax_stats.transAxes)
                self.ax_stats.add_patch(rect)
    
    def start(self):
        if not self.connect_to_server():
            return
        
        receive_thread = threading.Thread(target=self.receive_data, daemon=True)
        receive_thread.start()
        
        ani = FuncAnimation(self.fig, self.update_visualizations, interval=500, cache_frame_data=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if self.socket:
                self.socket.close()
            print(f"\n{self.client_id} disconnected.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visual Data Stream Client')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5555, help='Server port (default: 5555)')
    parser.add_argument('--id', dest='client_id', help='Client ID (default: auto-generated)')
    parser.add_argument('--max-records', type=int, default=100, help='Maximum records to keep (default: 100)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("VISUAL DATA STREAM CLIENT")
    print("=" * 50)
    
    client = DataStreamVisualClient(
        host=args.host, 
        port=args.port, 
        client_id=args.client_id,
        max_records=args.max_records
    )
    
    print(f"Client ID: {client.client_id}")
    print(f"Server: {args.host}:{args.port}")
    print(f"Max Records: {args.max_records}")
    print("=" * 50)
    print("Close the window to disconnect")
    print("=" * 50)
    
    try:
        client.start()
    except KeyboardInterrupt:
        print("\nClient shutdown complete")

if __name__ == "__main__":
    main()