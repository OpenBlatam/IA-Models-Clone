"""
WebSocket Manager - Gestor de WebSockets
=======================================

Gestión avanzada de WebSockets:
- Connection management
- Room/Channel support
- Broadcasting
- Authentication
- Rate limiting
"""

import logging
import json
from typing import Dict, Any, Optional, Set, Callable, List
from collections import defaultdict
from fastapi import WebSocket, WebSocketDisconnect, status
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Gestor de conexiones WebSocket.
    """
    
    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}
        self.rooms: Dict[str, Set[str]] = defaultdict(set)
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None) -> None:
        """Conecta un WebSocket"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        metadata = {
            "connected_at": datetime.now().isoformat(),
            "user_id": user_id,
            "ip": websocket.client.host if websocket.client else None
        }
        self.connection_metadata[connection_id] = metadata
        
        if user_id:
            self.user_connections[user_id].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str) -> None:
        """Desconecta un WebSocket"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Remover de rooms
        for room, connections in self.rooms.items():
            connections.discard(connection_id)
        
        # Remover de user connections
        metadata = self.connection_metadata.get(connection_id, {})
        user_id = metadata.get("user_id")
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
        
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str) -> bool:
        """Envía mensaje personal"""
        if connection_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False
    
    async def broadcast(self, message: Dict[str, Any], room: Optional[str] = None) -> int:
        """Broadcast a todos o a un room"""
        connections_to_send = set()
        
        if room:
            connections_to_send = self.rooms.get(room, set()).copy()
        else:
            connections_to_send = set(self.active_connections.keys())
        
        sent_count = 0
        disconnected = []
        
        for connection_id in connections_to_send:
            if connection_id in self.active_connections:
                try:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_json(message)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to broadcast to {connection_id}: {e}")
                    disconnected.append(connection_id)
            else:
                disconnected.append(connection_id)
        
        # Limpiar conexiones desconectadas
        for connection_id in disconnected:
            self.disconnect(connection_id)
        
        return sent_count
    
    async def send_to_user(self, message: Dict[str, Any], user_id: str) -> int:
        """Envía mensaje a todas las conexiones de un usuario"""
        user_connections = self.user_connections.get(user_id, set()).copy()
        sent_count = 0
        
        for connection_id in user_connections:
            if await self.send_personal_message(message, connection_id):
                sent_count += 1
        
        return sent_count
    
    def join_room(self, connection_id: str, room: str) -> None:
        """Une conexión a un room"""
        if connection_id in self.active_connections:
            self.rooms[room].add(connection_id)
            logger.info(f"Connection {connection_id} joined room {room}")
    
    def leave_room(self, connection_id: str, room: str) -> None:
        """Saca conexión de un room"""
        if room in self.rooms:
            self.rooms[room].discard(connection_id)
            logger.info(f"Connection {connection_id} left room {room}")
    
    def get_connection_count(self) -> int:
        """Obtiene número de conexiones activas"""
        return len(self.active_connections)
    
    def get_room_connections(self, room: str) -> int:
        """Obtiene número de conexiones en un room"""
        return len(self.rooms.get(room, set()))
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """Obtiene conexiones de un usuario"""
        return list(self.user_connections.get(user_id, set()))


class WebSocketRateLimiter:
    """Rate limiter para WebSockets"""
    
    def __init__(self, max_messages_per_minute: int = 60) -> None:
        self.max_messages = max_messages_per_minute
        self.message_counts: Dict[str, List[datetime]] = defaultdict(list)
    
    def is_allowed(self, connection_id: str) -> bool:
        """Verifica si está permitido enviar mensaje"""
        now = datetime.now()
        messages = self.message_counts[connection_id]
        
        # Limpiar mensajes antiguos (más de 1 minuto)
        messages[:] = [
            msg_time for msg_time in messages
            if (now - msg_time).total_seconds() < 60
        ]
        
        if len(messages) >= self.max_messages:
            return False
        
        messages.append(now)
        return True


def get_connection_manager() -> ConnectionManager:
    """Obtiene gestor de conexiones"""
    return ConnectionManager()


def get_websocket_rate_limiter(max_messages_per_minute: int = 60) -> WebSocketRateLimiter:
    """Obtiene rate limiter para WebSockets"""
    return WebSocketRateLimiter(max_messages_per_minute)















