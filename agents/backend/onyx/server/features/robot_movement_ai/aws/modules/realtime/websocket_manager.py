"""
WebSocket Manager
=================

WebSocket connection management.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self._connections: Dict[str, Any] = {}  # connection_id -> websocket
        self._user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self._room_connections: Dict[str, Set[str]] = {}  # room_id -> set of connection_ids
    
    def add_connection(self, connection_id: str, websocket: Any, user_id: Optional[str] = None):
        """Add WebSocket connection."""
        self._connections[connection_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "connected_at": datetime.now(),
            "rooms": set()
        }
        
        if user_id:
            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(connection_id)
        
        logger.info(f"Added connection: {connection_id}")
    
    def remove_connection(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id not in self._connections:
            return
        
        conn = self._connections[connection_id]
        user_id = conn.get("user_id")
        
        # Remove from user connections
        if user_id and user_id in self._user_connections:
            self._user_connections[user_id].discard(connection_id)
            if not self._user_connections[user_id]:
                del self._user_connections[user_id]
        
        # Remove from rooms
        for room_id in conn.get("rooms", set()):
            if room_id in self._room_connections:
                self._room_connections[room_id].discard(connection_id)
        
        del self._connections[connection_id]
        logger.info(f"Removed connection: {connection_id}")
    
    def join_room(self, connection_id: str, room_id: str):
        """Join connection to room."""
        if connection_id not in self._connections:
            return
        
        self._connections[connection_id]["rooms"].add(room_id)
        
        if room_id not in self._room_connections:
            self._room_connections[room_id] = set()
        self._room_connections[room_id].add(connection_id)
        
        logger.debug(f"Connection {connection_id} joined room {room_id}")
    
    def leave_room(self, connection_id: str, room_id: str):
        """Remove connection from room."""
        if connection_id not in self._connections:
            return
        
        self._connections[connection_id]["rooms"].discard(room_id)
        
        if room_id in self._room_connections:
            self._room_connections[room_id].discard(connection_id)
    
    async def send_to_connection(self, connection_id: str, message: Any):
        """Send message to connection."""
        if connection_id not in self._connections:
            return False
        
        try:
            websocket = self._connections[connection_id]["websocket"]
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send to {connection_id}: {e}")
            return False
    
    async def send_to_user(self, user_id: str, message: Any):
        """Send message to all user connections."""
        connection_ids = self._user_connections.get(user_id, set())
        
        results = await asyncio.gather(
            *[self.send_to_connection(cid, message) for cid in connection_ids],
            return_exceptions=True
        )
        
        return sum(1 for r in results if r is True)
    
    async def send_to_room(self, room_id: str, message: Any):
        """Send message to all connections in room."""
        connection_ids = self._room_connections.get(room_id, set())
        
        results = await asyncio.gather(
            *[self.send_to_connection(cid, message) for cid in connection_ids],
            return_exceptions=True
        )
        
        return sum(1 for r in results if r is True)
    
    def get_connection_count(self) -> int:
        """Get total connection count."""
        return len(self._connections)
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """Get user connection IDs."""
        return list(self._user_connections.get(user_id, set()))
    
    def get_room_connections(self, room_id: str) -> List[str]:
        """Get room connection IDs."""
        return list(self._room_connections.get(room_id, set()))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket statistics."""
        return {
            "total_connections": len(self._connections),
            "users_with_connections": len(self._user_connections),
            "active_rooms": len(self._room_connections),
            "connections_per_room": {
                room_id: len(conn_ids)
                for room_id, conn_ids in self._room_connections.items()
            }
        }


# Import asyncio
import asyncio















