# Advanced Enhancement: WebSocket Implementation

## Why Upgrade to WebSockets?

The current system uses TCP sockets, which work perfectly for local networks. However, if you want to deploy your streaming server to the cloud and access it from anywhere via a web browser, you'll need WebSockets.

### TCP Sockets vs WebSockets

| Feature | TCP Sockets (Current) | WebSockets (Advanced) |
|---------|----------------------|----------------------|
| **Browser Support** | ❌ No | ✅ Yes |
| **Cloud Deployment** | ❌ Limited | ✅ Full support |
| **Firewall Friendly** | ❌ Often blocked | ✅ Uses HTTP/HTTPS ports |
| **Real-time** | ✅ Yes | ✅ Yes |
| **Complexity** | Simple | Moderate |
| **Protocol** | Raw TCP | HTTP upgrade to WS |

## Prerequisites

Before attempting this enhancement, you should understand:
- Asynchronous programming (`async`/`await`)
- HTTP protocol basics
- JavaScript for browser clients
- Cloud deployment concepts

## Implementation Guide

### Step 1: Create WebSocket Server

Create `websocket_fraud_server.py`:

```python
import asyncio
import json
import random
from datetime import datetime
from aiohttp import web
import aiohttp_cors

class WebSocketFraudServer:
    def __init__(self):
        self.websockets = set()
        self.transaction_id = 0
        self.users = self.create_user_profiles()
        
    def create_user_profiles(self):
        """Create the same 25 user profiles as TCP version"""
        # Port your user profile logic here
        pass
        
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        try:
            # Send welcome message
            await ws.send_json({
                "type": "connection",
                "message": "Connected to WebSocket fraud stream",
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep connection alive
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle client messages if needed
                    pass
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f'WebSocket error: {ws.exception()}')
                    
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.websockets.discard(ws)
            
        return ws
    
    async def generate_transactions(self):
        """Generate and broadcast transactions"""
        while True:
            transaction = self.create_transaction()
            
            # Broadcast to all connected clients
            disconnected = set()
            for ws in self.websockets:
                try:
                    await ws.send_json(transaction)
                except ConnectionResetError:
                    disconnected.add(ws)
            
            # Remove disconnected clients
            self.websockets -= disconnected
            
            await asyncio.sleep(0.5)  # Generate every 0.5 seconds
    
    def create_transaction(self):
        """Generate a transaction (port from TCP version)"""
        self.transaction_id += 1
        # Your transaction generation logic here
        return {
            "transactionID": self.transaction_id,
            "userID": random.randint(1, 25),
            "amount": round(random.uniform(5.0, 500.0), 2),
            "timestamp": datetime.now().isoformat(),
            # Add all other fields
        }
    
    async def health_check(self, request):
        """Health check endpoint for cloud providers"""
        return web.json_response({
            "status": "healthy",
            "clients": len(self.websockets),
            "uptime": datetime.now().isoformat()
        })

def create_app():
    server = WebSocketFraudServer()
    app = web.Application()
    
    # Setup CORS for browser access
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Routes
    app.router.add_get('/ws', server.websocket_handler)
    app.router.add_get('/health', server.health_check)
    
    # Static file serving (for HTML client)
    app.router.add_static('/', path='static', name='static')
    
    # Start background transaction generator
    async def start_background_tasks(app):
        app['transaction_generator'] = asyncio.create_task(
            server.generate_transactions()
        )
    
    app.on_startup.append(start_background_tasks)
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 8080))  # Use PORT env var for cloud
    web.run_app(app, host='0.0.0.0', port=port)
```

### Step 2: Create Browser Client

Create `static/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection Stream</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
        }
        #transactions {
            background: white;
            padding: 20px;
            border-radius: 8px;
            max-height: 400px;
            overflow-y: auto;
        }
        .transaction {
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .fraud {
            background: #ffebee;
            border-left: 4px solid #f44336;
        }
    </style>
</head>
<body>
    <h1>🔍 Real-Time Fraud Detection Stream</h1>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value" id="total">0</div>
            <div>Total Transactions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="fraud-count">0</div>
            <div>Fraud Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="fraud-rate">0%</div>
            <div>Fraud Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="amount">$0</div>
            <div>Total Amount</div>
        </div>
    </div>
    
    <div id="chart"></div>
    <div id="transactions">
        <h3>Recent Transactions</h3>
    </div>
    
    <script>
        // Determine WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        let ws;
        let transactions = [];
        let fraudCount = 0;
        let totalAmount = 0;
        
        // Initialize Plotly chart
        Plotly.newPlot('chart', [{
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Transaction Amount'
        }], {
            title: 'Transaction Amount Over Time',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Amount ($)' }
        });
        
        function connect() {
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('Connected to WebSocket server');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'connection') {
                    console.log(data.message);
                    return;
                }
                
                // Process transaction
                transactions.push(data);
                totalAmount += data.amount;
                
                if (data.isFraud) {
                    fraudCount++;
                }
                
                // Update stats
                document.getElementById('total').textContent = transactions.length;
                document.getElementById('fraud-count').textContent = fraudCount;
                document.getElementById('fraud-rate').textContent = 
                    ((fraudCount / transactions.length) * 100).toFixed(1) + '%';
                document.getElementById('amount').textContent = 
                    '$' + totalAmount.toFixed(2);
                
                // Update chart
                Plotly.extendTraces('chart', {
                    x: [[new Date(data.timestamp)]],
                    y: [[data.amount]]
                }, [0]);
                
                // Keep only last 50 points
                if (transactions.length > 50) {
                    Plotly.relayout('chart', {
                        'xaxis.range': [
                            transactions[transactions.length - 50].timestamp,
                            data.timestamp
                        ]
                    });
                }
                
                // Add to transaction list
                const transDiv = document.createElement('div');
                transDiv.className = 'transaction' + (data.isFraud ? ' fraud' : '');
                transDiv.innerHTML = `
                    <strong>#${data.transactionID}</strong> - 
                    User ${data.userID} - 
                    $${data.amount.toFixed(2)} - 
                    ${new Date(data.timestamp).toLocaleTimeString()}
                    ${data.isFraud ? ' ⚠️ FRAUD' : ''}
                `;
                
                const container = document.getElementById('transactions');
                container.insertBefore(transDiv, container.children[1]);
                
                // Keep only last 10 transactions
                while (container.children.length > 11) {
                    container.removeChild(container.lastChild);
                }
            };
            
            ws.onclose = () => {
                console.log('Disconnected. Reconnecting...');
                setTimeout(connect, 3000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        // Start connection
        connect();
    </script>
</body>
</html>
```

### Step 3: Python Client for WebSocket

Create `websocket_client.py`:

```python
import asyncio
import json
import websockets
from datetime import datetime

class WebSocketFraudClient:
    def __init__(self, uri="ws://localhost:8080/ws"):
        self.uri = uri
        self.transactions = []
        
    async def collect_data(self, num_transactions=1000):
        """Collect transactions for training"""
        async with websockets.connect(self.uri) as websocket:
            print(f"Connected to {self.uri}")
            
            while len(self.transactions) < num_transactions:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get('type') != 'connection':
                    self.transactions.append(data)
                    if len(self.transactions) % 100 == 0:
                        print(f"Collected {len(self.transactions)} transactions")
            
            print(f"Collection complete: {len(self.transactions)} transactions")
            return self.transactions
    
    async def stream_predictions(self, model):
        """Stream predictions in real-time"""
        async with websockets.connect(self.uri) as websocket:
            print(f"Connected for real-time prediction")
            
            async for message in websocket:
                data = json.loads(message)
                
                if data.get('type') != 'connection':
                    # Make prediction
                    features = self.extract_features(data)
                    prediction = model.predict(features)
                    
                    if prediction > 0.5:
                        print(f"🚨 FRAUD DETECTED: Transaction {data['transactionID']}")
                        print(f"   Confidence: {prediction:.2%}")

# Usage
async def main():
    client = WebSocketFraudClient("ws://localhost:8080/ws")
    
    # Collect training data
    data = await client.collect_data(1000)
    
    # Save to file
    with open('training_data.json', 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    asyncio.run(main())
```

## Cloud Deployment Options

### 1. Heroku (Free tier discontinued, paid only)

Create `Procfile`:
```
web: python websocket_fraud_server.py
```

Create `runtime.txt`:
```
python-3.11.0
```

Deploy:
```bash
heroku create your-fraud-stream
git push heroku main
```

### 2. Railway.app (Recommended for beginners)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Railway automatically detects Python and runs your server. You get a URL like:
`https://your-app.railway.app`

### 3. Google Cloud Run

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "websocket_fraud_server.py"]
```

Deploy:
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/fraud-stream

# Deploy to Cloud Run
gcloud run deploy --image gcr.io/PROJECT-ID/fraud-stream --platform managed
```

### 4. AWS Elastic Beanstalk

Create `.ebextensions/python.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: websocket_fraud_server.py
  aws:elasticbeanstalk:application:environment:
    PORT: 8080
```

Deploy:
```bash
eb init -p python-3.11 fraud-stream
eb create fraud-stream-env
eb deploy
```

### 5. Azure App Service

```bash
# Create app service
az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name fraud-stream --runtime "PYTHON:3.11"

# Deploy
az webapp deployment source config-local-git --name fraud-stream --resource-group myResourceGroup
git remote add azure <deployment-url>
git push azure main
```

## Environment Requirements

### Local Development
```txt
# requirements.txt
aiohttp==3.9.0
aiohttp-cors==0.7.0
websockets==12.0
python-dotenv==1.0.0
```

### Production Requirements
```txt
# Additional for production
gunicorn==21.2.0
uvloop==0.19.0  # Faster event loop
redis==5.0.1    # For scaling across multiple instances
```

## Scaling Considerations

### Multiple Server Instances

When you need to scale beyond a single server, use Redis Pub/Sub:

```python
import aioredis

class ScalableWebSocketServer:
    def __init__(self):
        self.redis = None
        self.websockets = set()
    
    async def setup_redis(self):
        self.redis = await aioredis.create_redis_pool('redis://localhost')
        
    async def broadcast(self, message):
        """Broadcast via Redis to all server instances"""
        await self.redis.publish('transactions', json.dumps(message))
    
    async def redis_listener(self):
        """Listen for messages from other instances"""
        channel = (await self.redis.subscribe('transactions'))[0]
        
        while await channel.wait_message():
            message = await channel.get()
            # Send to local WebSocket clients
            for ws in self.websockets:
                await ws.send_str(message.decode())
```

## Security Considerations

### 1. Authentication

Add token-based authentication:

```python
async def websocket_handler(self, request):
    # Check for auth token
    token = request.headers.get('Authorization')
    if not self.verify_token(token):
        return web.Response(status=401, text='Unauthorized')
    
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    # ... rest of handler
```

### 2. Rate Limiting

Prevent abuse:

```python
from aiohttp_ratelimit import rate_limit

@rate_limit(rate=10, per=60)  # 10 requests per minute
async def websocket_handler(self, request):
    # ... handler code
```

### 3. HTTPS/WSS

Always use WSS (WebSocket Secure) in production:

```python
# In production, use reverse proxy (nginx) for SSL termination
# Or use cloud provider's built-in SSL
```

## Testing WebSocket Server

Create `test_websocket.py`:

```python
import asyncio
import websockets
import json

async def test_connection():
    uri = "ws://localhost:8080/ws"
    
    async with websockets.connect(uri) as websocket:
        print("Connected!")
        
        # Receive 10 transactions
        for i in range(10):
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received: {data}")
        
        print("Test complete!")

asyncio.run(test_connection())
```

## Performance Optimization

### 1. Message Compression

Enable compression for large-scale deployments:

```python
ws = web.WebSocketResponse(compress=True)
```

### 2. Connection Pooling

Reuse connections:

```python
class ConnectionPool:
    def __init__(self, max_connections=1000):
        self.connections = asyncio.Queue(maxsize=max_connections)
        self.all_connections = set()
```

### 3. Binary Protocol

For maximum efficiency, use MessagePack instead of JSON:

```python
import msgpack

# Serialize
data = msgpack.packb(transaction)
await ws.send_bytes(data)

# Deserialize
transaction = msgpack.unpackb(message, raw=False)
```

## Monitoring and Logging

### Add Prometheus Metrics

```python
from aiohttp_prometheus import setup_prometheus

app = web.Application()
setup_prometheus(app)
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

logger.info("transaction_generated", 
           transaction_id=transaction_id,
           amount=amount,
           is_fraud=is_fraud)
```

## Advantages of WebSocket Implementation

1. **Browser Compatibility**: Students can view streams directly in browser
2. **Cloud Ready**: Deploy to any cloud provider
3. **Firewall Friendly**: Uses standard HTTP/HTTPS ports
4. **Mobile Support**: Can create mobile apps that connect
5. **Global Access**: Stream accessible from anywhere
6. **Load Balancing**: Can scale horizontally with multiple instances

## Challenges to Consider

1. **Complexity**: More complex than TCP sockets
2. **Async Programming**: Requires understanding of async/await
3. **Cloud Costs**: Running 24/7 can incur charges
4. **Security**: Need to implement authentication for public deployment
5. **Debugging**: Harder to debug than simple TCP connections

## Project Extensions

Once you have WebSockets working, consider:

1. **Real-time Dashboard**: Build a React/Vue dashboard
2. **Mobile App**: Create React Native or Flutter app
3. **Multi-region Deployment**: Deploy to multiple cloud regions
4. **GraphQL Subscriptions**: Add GraphQL API for more flexibility
5. **Machine Learning Pipeline**: Integrate with MLflow or Kubeflow
6. **Event Sourcing**: Store all events for replay and analysis

## Resources

- [WebSocket Protocol RFC](https://datatracker.ietf.org/doc/html/rfc6455)
- [aiohttp Documentation](https://docs.aiohttp.org/)
- [WebSocket Client/Server Tutorial](https://websockets.readthedocs.io/)
- [Railway Deployment Guide](https://docs.railway.app/deploy/deployments)
- [Google Cloud Run WebSockets](https://cloud.google.com/run/docs/triggering/websockets)

## Conclusion

Upgrading to WebSockets opens up many possibilities for your fraud detection system. While more complex than TCP sockets, the benefits of cloud deployment and browser accessibility make it worthwhile for production systems. Start with the basic implementation, deploy to Railway or Heroku, and gradually add features as you learn.

Remember: The TCP socket version is perfect for learning and local development. Only pursue WebSockets if you need cloud deployment or browser access. Focus on mastering the neural network concepts first!