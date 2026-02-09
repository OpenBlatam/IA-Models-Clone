"""WebSocket utilities for real-time communication."""

from typing import Any, Callable, Dict, Optional, Set
import asyncio
import json
from datetime import datetime

try:
    from fastapi import WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState
except ImportError:
    WebSocket = None
    WebSocketDisconnect = None
    WebSocketState = None


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, metadata: Optional[Dict[str, Any]] = None):
        """Accept and store connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = metadata or {}
        self.connection_metadata[websocket]['connected_at'] = datetime.utcnow().isoformat()
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        self.active_connections.discard(websocket)
        self.connection_metadata.pop(websocket, None)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific connection."""
        if websocket in self.active_connections:
            await websocket.send_text(message)
    
    async def send_json(self, data: Dict[str, Any], websocket: WebSocket):
        """Send JSON to specific connection."""
        await self.send_personal_message(json.dumps(data), websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.add(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast JSON to all connections."""
        await self.broadcast(json.dumps(data))
    
    def get_connection_count(self) -> int:
        """Get active connection count."""
        return len(self.active_connections)


class WebSocketHandler:
    """WebSocket message handler."""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.handlers: Dict[str, Callable] = {}
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register handler for message type."""
        self.handlers[message_type] = handler
    
    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming message."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type in self.handlers:
                await self.handlers[message_type](websocket, data)
            else:
                await self.send_error(websocket, f"Unknown message type: {message_type}")
        except json.JSONDecodeError:
            await self.send_error(websocket, "Invalid JSON")
        except Exception as e:
            await self.send_error(websocket, str(e))
    
    async def send_error(self, websocket: WebSocket, error: str):
        """Send error message."""
        await self.manager.send_json({
            'type': 'error',
            'message': error,
            'timestamp': datetime.utcnow().isoformat()
        }, websocket)
    
    async def send_success(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send success message."""
        await self.manager.send_json({
            'type': 'success',
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }, websocket)


async def websocket_endpoint(
    websocket: WebSocket,
    manager: ConnectionManager,
    handler: Optional[WebSocketHandler] = None
):
    """Generic WebSocket endpoint."""
    await manager.connect(websocket)
    
    try:
        while True:
            message = await websocket.receive_text()
            
            if handler:
                await handler.handle_message(websocket, message)
            else:
                await manager.broadcast(f"Client says: {message}")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class RateLimitedConnectionManager(ConnectionManager):
    """Connection manager with rate limiting."""
    
    def __init__(self, max_messages_per_second: float = 10.0):
        super().__init__()
        self.max_messages_per_second = max_messages_per_second
        self.message_timestamps: Dict[WebSocket, list] = {}
    
    def _can_send(self, websocket: WebSocket) -> bool:
        """Check if connection can send message."""
        if websocket not in self.message_timestamps:
            self.message_timestamps[websocket] = []
        
        now = datetime.utcnow().timestamp()
        timestamps = self.message_timestamps[websocket]
        
        timestamps = [ts for ts in timestamps if now - ts < 1.0]
        self.message_timestamps[websocket] = timestamps
        
        return len(timestamps) < self.max_messages_per_second
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message with rate limiting."""
        if not self._can_send(websocket):
            await websocket.send_json({
                'type': 'error',
                'message': 'Rate limit exceeded'
            })
            return
        
        self.message_timestamps[websocket].append(datetime.utcnow().timestamp())
        await super().send_personal_message(message, websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection and cleanup."""
        self.message_timestamps.pop(websocket, None)
        super().disconnect(websocket)


