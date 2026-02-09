"""WebSocket routes for real-time updates."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set, Any
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["WebSocket"])

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, channel: str = "default"):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)
        logger.info(f"WebSocket connected to channel: {channel}")
    
    def disconnect(self, websocket: WebSocket, channel: str = "default"):
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
        logger.info(f"WebSocket disconnected from channel: {channel}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str, channel: str = "default"):
        if channel in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_text(message)
                except Exception:
                    disconnected.add(connection)
            # Clean up disconnected
            for conn in disconnected:
                self.active_connections[channel].discard(conn)


manager = ConnectionManager()


@router.websocket("/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str = "default"):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket, channel)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo message back (in production, process the message)
            await manager.send_personal_message(
                json.dumps({"echo": data, "channel": channel}),
                websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)


@router.post("/broadcast/{channel}")
async def broadcast_message(channel: str, message: Dict[str, Any]):
    """Broadcast message to all connections in a channel."""
    await manager.broadcast(json.dumps(message), channel)
    return {"status": "broadcasted", "channel": channel}

