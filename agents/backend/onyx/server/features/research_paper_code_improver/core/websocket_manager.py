"""
WebSocket Manager - Gestor de conexiones WebSocket
===================================================
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Estados de conexión"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class WebSocketConnection:
    """Conexión WebSocket"""
    id: str
    user_id: Optional[str] = None
    room_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.now)
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    status: ConnectionStatus = ConnectionStatus.CONNECTED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "room_id": self.room_id,
            "connected_at": self.connected_at.isoformat(),
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "message_count": self.message_count,
            "status": self.status.value,
            "metadata": self.metadata
        }


class WebSocketManager:
    """Gestor de conexiones WebSocket"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.rooms: Dict[str, List[str]] = {}  # room_id -> connection_ids
        self.user_connections: Dict[str, List[str]] = {}  # user_id -> connection_ids
        self.message_handlers: Dict[str, List[Callable]] = {}  # event_type -> handlers
    
    def register_connection(
        self,
        connection_id: str,
        user_id: Optional[str] = None,
        room_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WebSocketConnection:
        """Registra una conexión"""
        connection = WebSocketConnection(
            id=connection_id,
            user_id=user_id,
            room_id=room_id,
            metadata=metadata or {}
        )
        
        self.connections[connection_id] = connection
        
        if room_id:
            if room_id not in self.rooms:
                self.rooms[room_id] = []
            self.rooms[room_id].append(connection_id)
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(connection_id)
        
        logger.info(f"Conexión WebSocket {connection_id} registrada")
        return connection
    
    def unregister_connection(self, connection_id: str):
        """Desregistra una conexión"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # Remover de rooms
        if connection.room_id and connection.room_id in self.rooms:
            self.rooms[connection.room_id] = [
                cid for cid in self.rooms[connection.room_id]
                if cid != connection_id
            ]
            if not self.rooms[connection.room_id]:
                del self.rooms[connection.room_id]
        
        # Remover de user_connections
        if connection.user_id and connection.user_id in self.user_connections:
            self.user_connections[connection.user_id] = [
                cid for cid in self.user_connections[connection.user_id]
                if cid != connection_id
            ]
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        del self.connections[connection_id]
        logger.info(f"Conexión WebSocket {connection_id} desregistrada")
    
    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Obtiene una conexión"""
        return self.connections.get(connection_id)
    
    def get_room_connections(self, room_id: str) -> List[WebSocketConnection]:
        """Obtiene conexiones de una sala"""
        connection_ids = self.rooms.get(room_id, [])
        return [
            self.connections[cid] for cid in connection_ids
            if cid in self.connections
        ]
    
    def get_user_connections(self, user_id: str) -> List[WebSocketConnection]:
        """Obtiene conexiones de un usuario"""
        connection_ids = self.user_connections.get(user_id, [])
        return [
            self.connections[cid] for cid in connection_ids
            if cid in self.connections
        ]
    
    async def broadcast_to_room(
        self,
        room_id: str,
        message: Dict[str, Any],
        exclude_connection_id: Optional[str] = None
    ) -> int:
        """Transmite mensaje a una sala"""
        connections = self.get_room_connections(room_id)
        sent_count = 0
        
        for connection in connections:
            if connection.id == exclude_connection_id:
                continue
            
            # En producción, enviar a través de WebSocket real
            # Por ahora, solo registrar
            connection.message_count += 1
            connection.last_message_at = datetime.now()
            sent_count += 1
        
        logger.debug(f"Broadcast a {sent_count} conexiones en sala {room_id}")
        return sent_count
    
    async def send_to_user(
        self,
        user_id: str,
        message: Dict[str, Any]
    ) -> int:
        """Envía mensaje a un usuario"""
        connections = self.get_user_connections(user_id)
        sent_count = 0
        
        for connection in connections:
            connection.message_count += 1
            connection.last_message_at = datetime.now()
            sent_count += 1
        
        return sent_count
    
    def register_message_handler(self, event_type: str, handler: Callable):
        """Registra un handler de mensajes"""
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []
        self.message_handlers[event_type].append(handler)
    
    async def handle_message(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ):
        """Maneja un mensaje recibido"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.message_count += 1
        connection.last_message_at = datetime.now()
        
        event_type = message.get("type")
        if event_type and event_type in self.message_handlers:
            for handler in self.message_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(connection, message)
                    else:
                        handler(connection, message)
                except Exception as e:
                    logger.error(f"Error en handler de mensaje: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        return {
            "total_connections": len(self.connections),
            "total_rooms": len(self.rooms),
            "total_users": len(self.user_connections),
            "connections_by_room": {
                room_id: len(conn_ids)
                for room_id, conn_ids in self.rooms.items()
            }
        }




