"""
Real-time Collaboration - Colaboración en tiempo real con WebSockets
=====================================================================
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Tipos de mensajes"""
    JOIN = "join"
    LEAVE = "leave"
    CODE_CHANGE = "code_change"
    CURSOR_UPDATE = "cursor_update"
    SELECTION = "selection"
    COMMENT = "comment"
    TYPING = "typing"
    PRESENCE = "presence"
    HEARTBEAT = "heartbeat"


@dataclass
class CollaborationMessage:
    """Mensaje de colaboración"""
    id: str
    type: MessageType
    room_id: str
    user_id: str
    user_name: str
    payload: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "type": self.type.value,
            "room_id": self.room_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class UserPresence:
    """Presencia de usuario"""
    user_id: str
    user_name: str
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    is_typing: bool = False
    last_seen: datetime = None
    
    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.now()


class CollaborationRoom:
    """Sala de colaboración"""
    
    def __init__(self, room_id: str, room_name: str):
        self.room_id = room_id
        self.room_name = room_name
        self.users: Dict[str, UserPresence] = {}
        self.messages: List[CollaborationMessage] = []
        self.created_at = datetime.now()
        self.code_state: Optional[str] = None
        self.version: int = 0
    
    def add_user(self, user_id: str, user_name: str) -> UserPresence:
        """Agrega un usuario a la sala"""
        presence = UserPresence(user_id=user_id, user_name=user_name)
        self.users[user_id] = presence
        return presence
    
    def remove_user(self, user_id: str):
        """Remueve un usuario de la sala"""
        if user_id in self.users:
            del self.users[user_id]
    
    def update_presence(
        self,
        user_id: str,
        cursor_position: Optional[Dict[str, Any]] = None,
        selection: Optional[Dict[str, Any]] = None,
        is_typing: Optional[bool] = None
    ):
        """Actualiza la presencia de un usuario"""
        if user_id not in self.users:
            return
        
        presence = self.users[user_id]
        if cursor_position is not None:
            presence.cursor_position = cursor_position
        if selection is not None:
            presence.selection = selection
        if is_typing is not None:
            presence.is_typing = is_typing
        presence.last_seen = datetime.now()
    
    def apply_code_change(self, user_id: str, change: Dict[str, Any]) -> bool:
        """Aplica un cambio de código"""
        try:
            # Operational Transform o similar para conflict resolution
            # Por simplicidad, aplicamos cambios secuencialmente
            change_type = change.get("type")
            if change_type == "insert":
                position = change.get("position", 0)
                text = change.get("text", "")
                if self.code_state is None:
                    self.code_state = ""
                self.code_state = (
                    self.code_state[:position] +
                    text +
                    self.code_state[position:]
                )
            elif change_type == "delete":
                position = change.get("position", 0)
                length = change.get("length", 0)
                if self.code_state:
                    self.code_state = (
                        self.code_state[:position] +
                        self.code_state[position + length:]
                    )
            elif change_type == "replace":
                start = change.get("start", 0)
                end = change.get("end", 0)
                text = change.get("text", "")
                if self.code_state:
                    self.code_state = (
                        self.code_state[:start] +
                        text +
                        self.code_state[end:]
                    )
            
            self.version += 1
            return True
        except Exception as e:
            logger.error(f"Error aplicando cambio de código: {e}")
            return False
    
    def add_message(self, message: CollaborationMessage):
        """Agrega un mensaje a la sala"""
        self.messages.append(message)
        # Mantener solo los últimos 1000 mensajes
        if len(self.messages) > 1000:
            self.messages = self.messages[-1000:]


class RealTimeCollaboration:
    """Sistema de colaboración en tiempo real"""
    
    def __init__(self):
        self.rooms: Dict[str, CollaborationRoom] = {}
        self.user_rooms: Dict[str, Set[str]] = defaultdict(set)  # user_id -> room_ids
        self.websocket_connections: Dict[str, Any] = {}  # connection_id -> websocket
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
    
    def create_room(self, room_id: str, room_name: str) -> CollaborationRoom:
        """Crea una nueva sala de colaboración"""
        if room_id in self.rooms:
            return self.rooms[room_id]
        
        room = CollaborationRoom(room_id=room_id, room_name=room_name)
        self.rooms[room_id] = room
        logger.info(f"Sala {room_id} creada")
        return room
    
    def join_room(self, room_id: str, user_id: str, user_name: str) -> bool:
        """Une un usuario a una sala"""
        if room_id not in self.rooms:
            self.create_room(room_id, room_id)
        
        room = self.rooms[room_id]
        room.add_user(user_id, user_name)
        self.user_rooms[user_id].add(room_id)
        
        # Notificar a otros usuarios
        self._broadcast_to_room(
            room_id,
            CollaborationMessage(
                id=str(uuid.uuid4()),
                type=MessageType.JOIN,
                room_id=room_id,
                user_id=user_id,
                user_name=user_name,
                payload={},
                timestamp=datetime.now()
            ),
            exclude_user_id=user_id
        )
        
        return True
    
    def leave_room(self, room_id: str, user_id: str):
        """Remueve un usuario de una sala"""
        if room_id in self.rooms:
            self.rooms[room_id].remove_user(user_id)
            self.user_rooms[user_id].discard(room_id)
            
            # Notificar a otros usuarios
            self._broadcast_to_room(
                room_id,
                CollaborationMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.LEAVE,
                    room_id=room_id,
                    user_id=user_id,
                    user_name="",
                    payload={},
                    timestamp=datetime.now()
                )
            )
    
    def send_message(self, message: CollaborationMessage) -> bool:
        """Envía un mensaje a una sala"""
        if message.room_id not in self.rooms:
            return False
        
        room = self.rooms[message.room_id]
        room.add_message(message)
        
        # Procesar según tipo
        if message.type == MessageType.CODE_CHANGE:
            room.apply_code_change(message.user_id, message.payload)
        elif message.type == MessageType.CURSOR_UPDATE:
            room.update_presence(
                message.user_id,
                cursor_position=message.payload.get("cursor")
            )
        elif message.type == MessageType.SELECTION:
            room.update_presence(
                message.user_id,
                selection=message.payload.get("selection")
            )
        elif message.type == MessageType.TYPING:
            room.update_presence(
                message.user_id,
                is_typing=message.payload.get("typing", False)
            )
        
        # Broadcast a todos los usuarios de la sala
        self._broadcast_to_room(message.room_id, message)
        
        # Ejecutar handlers
        if message.type in self.message_handlers:
            for handler in self.message_handlers[message.type]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error en handler de mensaje: {e}")
        
        return True
    
    def _broadcast_to_room(
        self,
        room_id: str,
        message: CollaborationMessage,
        exclude_user_id: Optional[str] = None
    ):
        """Transmite un mensaje a todos los usuarios de una sala"""
        if room_id not in self.rooms:
            return
        
        room = self.rooms[room_id]
        message_dict = message.to_dict()
        
        for user_id in room.users:
            if user_id == exclude_user_id:
                continue
            
            # Enviar a través de WebSocket si está conectado
            # Esto se implementaría con FastAPI WebSockets
            logger.debug(f"Broadcasting message to user {user_id} in room {room_id}")
    
    def get_room_state(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado actual de una sala"""
        if room_id not in self.rooms:
            return None
        
        room = self.rooms[room_id]
        return {
            "room_id": room.room_id,
            "room_name": room.room_name,
            "users": [
                {
                    "user_id": p.user_id,
                    "user_name": p.user_name,
                    "cursor_position": p.cursor_position,
                    "selection": p.selection,
                    "is_typing": p.is_typing
                }
                for p in room.users.values()
            ],
            "code_state": room.code_state,
            "version": room.version,
            "user_count": len(room.users)
        }
    
    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[CollaborationMessage], None]
    ):
        """Registra un handler para un tipo de mensaje"""
        self.message_handlers[message_type].append(handler)
    
    def get_user_rooms(self, user_id: str) -> List[str]:
        """Obtiene las salas de un usuario"""
        return list(self.user_rooms.get(user_id, set()))




