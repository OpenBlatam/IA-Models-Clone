"""
WebSocket endpoints for real-time updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import logging

from ...utils.json_helpers import safe_json_loads, safe_json_dumps

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, conversation_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = set()
        self.active_connections[conversation_id].add(websocket)
        logger.info(f"WebSocket connected for conversation {conversation_id}")
    
    def disconnect(self, websocket: WebSocket, conversation_id: str):
        """Remove a WebSocket connection."""
        if conversation_id in self.active_connections:
            self.active_connections[conversation_id].discard(websocket)
            if not self.active_connections[conversation_id]:
                del self.active_connections[conversation_id]
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific connection."""
        await websocket.send_text(message)
    
    async def broadcast_to_conversation(self, conversation_id: str, message: Dict):
        """Broadcast a message to all connections in a conversation."""
        if conversation_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[conversation_id]:
                try:
                    await connection.send_text(safe_json_dumps(message))
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
                    disconnected.add(connection)
            
            for conn in disconnected:
                self.disconnect(conn, conversation_id)


manager = ConnectionManager()


@router.websocket("/conversation/{conversation_id}")
async def websocket_conversation(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time conversation updates."""
    await manager.connect(websocket, conversation_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = safe_json_loads(data, default={})
            
            if message.get("type") == "ping":
                await manager.send_personal_message(
                    safe_json_dumps({"type": "pong"}),
                    websocket
                )
            elif message.get("type") == "subscribe":
                await manager.send_personal_message(
                    safe_json_dumps({
                        "type": "subscribed",
                        "conversation_id": conversation_id
                    }),
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, conversation_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, conversation_id)


def broadcast_message(conversation_id: str, message_type: str, data: Dict):
    """Helper function to broadcast messages."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(manager.broadcast_to_conversation(
                conversation_id,
                {"type": message_type, "data": data}
            ))
        else:
            loop.run_until_complete(manager.broadcast_to_conversation(
                conversation_id,
                {"type": message_type, "data": data}
            ))
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")






