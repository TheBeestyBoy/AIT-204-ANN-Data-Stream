# Real-Time Financial Data Stream System for ANN Training

## Overview
This project implements a client-server architecture for streaming financial transaction data in real-time, designed specifically for training Artificial Neural Networks (ANNs) in fraud detection. The system simulates realistic banking transactions with embedded fraudulent patterns, providing students with hands-on experience in building real-time ML applications.

## Table of Contents
- [Architecture](#architecture)
- [Components](#components)
- [User Profiles and Data Generation](#user-profiles-and-data-generation)
- [Network Concepts](#network-concepts)
- [Installation](#installation)
- [Usage](#usage)
- [Educational Applications](#educational-applications)

## Architecture

### Client-Server Model
The system uses a **TCP/IP socket-based architecture** where:
- **Server**: Generates and broadcasts transaction data to all connected clients
- **Clients**: Connect to the server, receive data streams, and perform real-time analysis

### Key Technologies
- **Python Sockets**: Low-level networking for reliable data transmission
- **Threading**: Concurrent handling of multiple client connections
- **JSON**: Lightweight data serialization format
- **Matplotlib**: Real-time data visualization
- **TCP Protocol**: Ensures reliable, ordered delivery of data packets

### Network Flow
```
[Data Generator] → [TCP Server:5555] → [Network] → [Multiple Clients]
                         ↓
                   [Broadcast Loop]
                         ↓
                [Client 1] [Client 2] ... [Client N]
```

## Components

### 1. Basic Streaming System

#### `stream_server.py`
- Simple transaction generator
- Broadcasts random transactions to all connected clients
- Configurable generation interval
- Multi-client support through threading

#### `stream_client.py`
- Terminal-based client
- Displays transactions and basic statistics
- Rolling window analysis

#### `stream_client_visual.py`
- GUI client with real-time visualizations
- Live updating transaction table
- Time series plots and histograms
- Statistical dashboard

### 2. Fraud Detection Training System

#### `fraud_stream_server.py`
**Advanced server with realistic user behavior simulation:**

- **25 Unique User Profiles** including:
  - High/low frequency spenders
  - Time-based patterns (morning shoppers, night owls)
  - Corporate cards (weekday only)
  - Students, retirees, luxury buyers
  - Weekend warriors, budget conscious users
  - 3 fraud-prone profiles with specific attack patterns

- **Intelligent Transaction Generation:**
  - Users maintain consistent spending patterns
  - Time-of-day preferences
  - Favorite items and merchants
  - Daily transaction limits
  - Seasonal and periodic behaviors

- **Fraud Pattern Injection:**
  - ~5-10% fraud rate from specific users
  - Multiple fraud patterns:
    - Sudden amount spikes (3-5x normal)
    - Unusual purchase times
    - Rapid succession transactions
    - Round number amounts ($100, $200)
    - Category switches
    - Midnight purchases
    - Duplicate amounts

## User Profiles and Data Generation

### Profile Types (20 Normal + 3 Fraud-Prone)

1. **High-Frequency Big Spender**: 
   - $150-500 per transaction
   - Peak hours: business hours
   - Up to 20 transactions/day

2. **Periodic Big Spender**:
   - Normal: $10-60
   - Every 7th transaction: $300-600
   - Models subscription renewals

3. **Corporate Card**:
   - Weekdays only
   - Business hours
   - $50-300 range

4. **Student**:
   - Small amounts: $5-40
   - Lunch and evening peaks
   - Budget-conscious

5. **Fraud-Prone Users** (IDs 13, 14, 15):
   - 10-15% fraud probability
   - Specific attack patterns
   - Used for training detection models

### Transaction Data Structure
```json
{
  "transactionID": 1234,
  "userID": 13,
  "amount": 459.99,
  "timestamp": "2024-01-15T14:30:45",
  "itemID": "E5678",
  "merchantID": "M234",
  "isWeekend": false,
  "hourOfDay": 14,
  "daysSinceLastTransaction": 1,
  "userProfile": "fraud_prone_1",
  "isFraud": true,
  "fraudPattern": "sudden_spike"
}
```

## Network Concepts

### TCP/IP Communication
- **TCP (Transmission Control Protocol)**: Ensures reliable, ordered data delivery
- **Connection-oriented**: Establishes connection before data transfer
- **Error checking**: Automatic retransmission of lost packets
- **Flow control**: Prevents overwhelming slower clients

### Socket Programming
```python
# Server binds to address and listens
server_socket.bind((host, port))
server_socket.listen(max_clients)

# Client connects to server
client_socket.connect((server_host, server_port))
```

### Threading Model
- **Main Thread**: Accepts new connections
- **Generator Thread**: Creates transactions at intervals
- **Client Threads**: Handle individual client connections
- **Lock Synchronization**: Prevents race conditions

### Broadcasting Pattern
Server maintains a list of connected clients and sends each transaction to all clients simultaneously:
```python
for client in connected_clients:
    client.send(transaction_data)
```

## Installation

### Requirements
```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Required packages:
# - matplotlib>=3.5.0 (for visualizations)
# - pandas>=1.3.0 (for data handling)
# - numpy>=1.21.0 (for numerical operations)
```

### File Structure
```
Data-Stream/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── stream_server.py            # Basic streaming server
├── stream_client.py            # Terminal client
├── stream_client_visual.py     # Visual client with graphs
├── fraud_stream_server.py      # Advanced server with fraud patterns
└── test_multiple_clients.sh    # Testing script
```

## Usage

### Classroom Setup (Recommended)

#### Teacher's Computer (Server)

1. **Create WiFi Hotspot** using phone
2. **Connect computer** to hotspot
3. **Find IP address**:
   ```bash
   # Mac/Linux
   ifconfig | grep inet
   # Look for: inet 192.168.1.5
   
   # Windows
   ipconfig
   # Look for: IPv4 Address
   ```
4. **Start fraud detection server**:
   ```bash
   python3 fraud_stream_server.py --host 0.0.0.0 --port 5555 --interval 0.5
   ```

#### Students' Computers (Clients)

1. **Connect to same hotspot**
2. **Run visual client** with teacher's IP:
   ```bash
   python3 stream_client_visual.py --host 192.168.1.5 --port 5555 --id YourName
   ```

### Local Testing
```bash
# Terminal 1: Start server
python3 fraud_stream_server.py

# Terminal 2: Start visual client
python3 stream_client_visual.py --id Student1

# Terminal 3: Another client
python3 stream_client_visual.py --id Student2
```

### Command-Line Options

#### Server Options
```bash
python3 fraud_stream_server.py \
  --host 0.0.0.0 \      # Bind to all interfaces
  --port 5555 \         # TCP port
  --interval 0.5        # Seconds between transactions
```

#### Client Options
```bash
python3 stream_client_visual.py \
  --host 192.168.1.5 \  # Server IP address
  --port 5555 \         # Server port
  --id StudentName \    # Client identifier
  --max-records 100     # Buffer size for visualizations
```

## Educational Applications

### 1. Real-Time Fraud Detection
Students can build ANNs that:
- Process streaming transactions
- Detect anomalies in real-time
- Update model predictions continuously
- Calculate precision, recall, F1-score

### 2. Time Series Prediction
- Predict next transaction amount
- Forecast daily transaction volumes
- Identify spending trends

### 3. User Behavior Clustering
- Group users by spending patterns
- Identify customer segments
- Detect behavior changes

### 4. Network Programming Concepts
Learn about:
- Client-server architecture
- Socket programming
- Network protocols (TCP/IP)
- Concurrent programming
- Data serialization

### Sample Student Implementation
```python
import tensorflow as tf
from stream_client import DataStreamClient

class FraudDetector(DataStreamClient):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = tf.keras.models.load_model(model_path)
    
    def process_transaction(self, transaction):
        # Extract features
        features = self.extract_features(transaction)
        
        # Predict fraud probability
        fraud_prob = self.model.predict(features)[0][0]
        
        # Flag if suspicious
        if fraud_prob > 0.5:
            print(f"🚨 FRAUD ALERT: Transaction {transaction['transactionID']}")
            print(f"   Confidence: {fraud_prob:.2%}")
```

## Troubleshooting

### Common Issues

1. **"Address already in use"**
   - Another server is running on the port
   - Solution: `pkill -f stream_server.py` or use different port

2. **"Connection refused"**
   - Server not running or wrong IP/port
   - Check firewall settings
   - Verify both devices on same network

3. **"nodename nor servname provided"**
   - Invalid IP address format
   - Use actual IP, not placeholder

4. **Firewall Blocking**
   - Mac: System Preferences → Security → Firewall Options
   - Windows: Allow Python through Windows Defender
   - Linux: Check iptables/ufw settings

### Network Diagnostics
```bash
# Test connectivity
ping server_ip

# Check if port is open
telnet server_ip 5555

# View active connections
netstat -an | grep 5555
```

## Performance Considerations

- **Bandwidth**: ~1-2 KB per transaction
- **Latency**: < 100ms on local network
- **Scalability**: Supports 20+ simultaneous clients
- **CPU Usage**: Minimal (< 5% per client)

## Security Notes

This is an **educational system** designed for classroom use:
- No encryption (data sent in plaintext)
- No authentication (any client can connect)
- No data persistence (transactions are ephemeral)

For production systems, implement:
- TLS/SSL encryption
- Client authentication
- Rate limiting
- Data validation

## Future Enhancements

Potential additions for advanced courses:
- WebSocket support for browser-based clients
- Apache Kafka integration for scalability
- PostgreSQL for transaction persistence
- REST API for historical data access
- Docker containerization
- Kubernetes orchestration

## Credits

Developed for AIT-204: Artificial Neural Networks course
- Real-time data streaming
- Fraud detection patterns
- Network programming concepts
- Machine learning applications

## License

Educational use only. Not for production deployment.