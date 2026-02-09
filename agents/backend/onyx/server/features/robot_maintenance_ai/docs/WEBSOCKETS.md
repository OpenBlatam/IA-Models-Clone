# WebSocket API Documentation

## Overview

The Robot Maintenance AI system supports WebSocket connections for real-time updates. This allows clients to receive instant notifications when new messages are added to conversations or when maintenance predictions are updated.

## Connection

### Endpoint

```
ws://localhost:8000/ws/conversation/{conversation_id}
```

### Example (JavaScript)

```javascript
const conversationId = "conv_123";
const ws = new WebSocket(`ws://localhost:8000/ws/conversation/${conversationId}`);

ws.onopen = () => {
    console.log("WebSocket connected");
    // Subscribe to updates
    ws.send(JSON.stringify({
        type: "subscribe",
        conversation_id: conversationId
    }));
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log("Received:", message);
    
    if (message.type === "new_message") {
        // Handle new message
        console.log("New message:", message.data);
    } else if (message.type === "prediction_update") {
        // Handle prediction update
        console.log("Prediction updated:", message.data);
    }
};

ws.onerror = (error) => {
    console.error("WebSocket error:", error);
};

ws.onclose = () => {
    console.log("WebSocket disconnected");
};
```

### Example (Python)

```python
import asyncio
import websockets
import json

async def connect_websocket(conversation_id: str):
    uri = f"ws://localhost:8000/ws/conversation/{conversation_id}"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe
        await websocket.send(json.dumps({
            "type": "subscribe",
            "conversation_id": conversation_id
        }))
        
        # Listen for messages
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received: {data}")

# Run
asyncio.run(connect_websocket("conv_123"))
```

## Message Types

### Client to Server

#### Subscribe
```json
{
    "type": "subscribe",
    "conversation_id": "conv_123"
}
```

#### Ping
```json
{
    "type": "ping"
}
```

### Server to Client

#### Subscribed
```json
{
    "type": "subscribed",
    "conversation_id": "conv_123"
}
```

#### Pong
```json
{
    "type": "pong"
}
```

#### New Message
```json
{
    "type": "new_message",
    "data": {
        "role": "assistant",
        "content": "Message content",
        "timestamp": "2024-01-01T12:00:00"
    }
}
```

#### Prediction Update
```json
{
    "type": "prediction_update",
    "data": {
        "robot_type": "robots_industriales",
        "prediction": {
            "next_maintenance_days": 30,
            "priority": "normal"
        }
    }
}
```

## Use Cases

1. **Real-time Chat**: Receive new messages as they are generated
2. **Live Updates**: Get instant notifications when maintenance predictions change
3. **Status Monitoring**: Monitor conversation status in real-time
4. **Collaborative Sessions**: Multiple users can watch the same conversation

## Best Practices

1. **Reconnection**: Implement automatic reconnection logic
2. **Heartbeat**: Send ping messages periodically to keep connection alive
3. **Error Handling**: Handle connection errors gracefully
4. **Message Queuing**: Queue messages if connection is temporarily lost

## Limitations

- Maximum concurrent connections per conversation: 100 (configurable)
- Message size limit: 64KB
- Connection timeout: 60 seconds of inactivity






