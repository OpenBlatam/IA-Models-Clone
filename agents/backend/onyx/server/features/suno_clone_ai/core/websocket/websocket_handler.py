"""
WebSocket Handler

Utilities for WebSocket communication.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handle WebSocket connections and messages."""
    
    def __init__(self):
        """Initialize WebSocket handler."""
        self.connections: Dict[str, Any] = {}
        self.rooms: Dict[str, List[str]] = defaultdict(list)
        self.message_handlers: Dict[str, Callable] = {}
    
    def add_connection(
        self,
        connection_id: str,
        websocket: Any
    ) -> None:
        """
        Add WebSocket connection.
        
        Args:
            connection_id: Connection identifier
            websocket: WebSocket connection
        """
        self.connections[connection_id] = websocket
        logger.info(f"WebSocket connection added: {connection_id}")
    
    def remove_connection(self, connection_id: str) -> None:
        """
        Remove WebSocket connection.
        
        Args:
            connection_id: Connection identifier
        """
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        # Remove from rooms
        for room in self.rooms.values():
            if connection_id in room:
                room.remove(connection_id)
        
        logger.info(f"WebSocket connection removed: {connection_id}")
    
    async def send(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Send message to connection.
        
        Args:
            connection_id: Connection identifier
            message: Message dictionary
            
        Returns:
            True if sent successfully
        """
        if connection_id not in self.connections:
            logger.warning(f"Connection not found: {connection_id}")
            return False
        
        try:
            websocket = self.connections[connection_id]
            await websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def broadcast(
        self,
        message: Dict[str, Any],
        room: Optional[str] = None
    ) -> int:
        """
        Broadcast message to connections.
        
        Args:
            message: Message dictionary
            room: Optional room name (None = all connections)
            
        Returns:
            Number of connections messaged
        """
        connections = self.rooms[room] if room else list(self.connections.keys())
        
        count = 0
        for connection_id in connections:
            if await self.send(connection_id, message):
                count += 1
        
        return count
    
    def join_room(
        self,
        connection_id: str,
        room: str
    ) -> None:
        """
        Join connection to room.
        
        Args:
            connection_id: Connection identifier
            room: Room name
        """
        if connection_id not in self.rooms[room]:
            self.rooms[room].append(connection_id)
            logger.info(f"Connection {connection_id} joined room {room}")
    
    def leave_room(
        self,
        connection_id: str,
        room: str
    ) -> None:
        """
        Leave connection from room.
        
        Args:
            connection_id: Connection identifier
            room: Room name
        """
        if connection_id in self.rooms[room]:
            self.rooms[room].remove(connection_id)
            logger.info(f"Connection {connection_id} left room {room}")
    
    def register_handler(
        self,
        message_type: str,
        handler: Callable
    ) -> None:
        """
        Register message handler.
        
        Args:
            message_type: Message type
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def handle_message(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Handle incoming message.
        
        Args:
            connection_id: Connection identifier
            message: Message dictionary
            
        Returns:
            Response message or None
        """
        message_type = message.get('type')
        
        if message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            return await handler(connection_id, message)
        
        logger.warning(f"No handler for message type: {message_type}")
        return None


def create_websocket_handler() -> WebSocketHandler:
    """Create WebSocket handler."""
    return WebSocketHandler()


async def send_message(
    handler: WebSocketHandler,
    connection_id: str,
    message: Dict[str, Any]
) -> bool:
    """Send message via handler."""
    return await handler.send(connection_id, message)


async def broadcast_message(
    handler: WebSocketHandler,
    message: Dict[str, Any],
    room: Optional[str] = None
) -> int:
    """Broadcast message via handler."""
    return await handler.broadcast(message, room)



