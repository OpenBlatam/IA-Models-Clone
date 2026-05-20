"""
MCP WebSocket - Soporte WebSocket
==================================
"""

import logging
import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
from datetime import datetime

from .utils.websocket_helpers import (
    WebSocketConnection,
    WebSocketRoom,
    send_json_safe,
    receive_json_safe,
    create_websocket_message,
    validate_websocket_message,
    broadcast_to_connections,
    WebSocketHeartbeat,
)

logger = logging.getLogger(__name__)


class WebSocketMessage(BaseModel):
    """Mensaje WebSocket"""
    type: str = Field(..., description="Tipo de mensaje")
    payload: Dict[str, Any] = Field(..., description="Payload del mensaje")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketManager:
    """
    Gestor de conexiones WebSocket
    
    Maneja múltiples conexiones WebSocket y broadcasting.
    Incluye soporte para salas, heartbeat, y gestión avanzada de conexiones.
    """
    
    def __init__(
        self,
        enable_heartbeat: bool = True,
        heartbeat_interval: float = 30.0,
        max_connections: Optional[int] = None
    ):
        """
        Inicializar gestor WebSocket.
        
        Args:
            enable_heartbeat: Habilitar heartbeat (default: True)
            heartbeat_interval: Intervalo de heartbeat en segundos (default: 30.0)
            max_connections: Máximo de conexiones (opcional)
        """
        self._connections: Dict[str, WebSocketConnection] = {}
        self._rooms: Dict[str, WebSocketRoom] = {}
        self._handlers: Dict[str, Callable] = {}
        self.router = APIRouter()
        self.max_connections = max_connections
        self.enable_heartbeat = enable_heartbeat
        
        if enable_heartbeat:
            self.heartbeat = WebSocketHeartbeat(interval=heartbeat_interval)
        else:
            self.heartbeat = None
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Conecta un WebSocket
        
        Args:
            websocket: Conexión WebSocket
            user_id: ID del usuario (opcional)
            metadata: Metadata adicional (opcional)
        
        Returns:
            ID de la conexión
        
        Raises:
            RuntimeError: Si se alcanza el máximo de conexiones
        """
        if self.max_connections and len(self._connections) >= self.max_connections:
            raise RuntimeError(f"Maximum connections ({self.max_connections}) reached")
        
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        
        from .utils.websocket_helpers import WebSocketState
        
        connection = WebSocketConnection(
            websocket=websocket,
            connection_id=connection_id,
            user_id=user_id,
            metadata=metadata
        )
        connection.state = WebSocketState.CONNECTED
        
        self._connections[connection_id] = connection
        
        # Iniciar heartbeat si está habilitado
        if self.heartbeat:
            await self.heartbeat.start_for_connection(
                connection_id,
                websocket,
                on_timeout=self._on_heartbeat_timeout
            )
        
        logger.info(f"WebSocket connected: {connection_id}. Total: {len(self._connections)}")
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """
        Desconecta un WebSocket
        
        Args:
            connection_id: ID de la conexión
        """
        if connection_id in self._connections:
            from .utils.websocket_helpers import WebSocketState
            connection = self._connections[connection_id]
            connection.state = WebSocketState.DISCONNECTED
            
            # Remover de todas las salas
            for room in self._rooms.values():
                room.remove_connection(connection_id)
            
            # Detener heartbeat
            if self.heartbeat:
                await self.heartbeat.stop_for_connection(connection_id)
            
            del self._connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}. Total: {len(self._connections)}")
    
    def _on_heartbeat_timeout(self, connection_id: str):
        """Callback cuando hay timeout en heartbeat"""
        logger.warning(f"Heartbeat timeout for connection: {connection_id}")
        # Desconectar automáticamente
        asyncio.create_task(self.disconnect(connection_id))
    
    async def send_message(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Envía mensaje a un WebSocket específico
        
        Args:
            connection_id: ID de la conexión
            message: Mensaje a enviar (dict)
        
        Returns:
            True si se envió exitosamente
        """
        connection = self._connections.get(connection_id)
        if not connection:
            logger.warning(f"Connection not found: {connection_id}")
            return False
        
        success = await send_json_safe(connection.websocket, message)
        if success:
            connection.update_activity()
        else:
            connection.record_error()
            # Desconectar si hay muchos errores
            if connection.error_count > 5:
                await self.disconnect(connection_id)
        
        return success
    
    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude: Optional[List[str]] = None
    ) -> int:
        """
        Broadcast mensaje a todos los WebSockets conectados
        
        Args:
            message: Mensaje a enviar (dict)
            exclude: Lista de connection_ids a excluir (opcional)
        
        Returns:
            Número de mensajes enviados exitosamente
        """
        exclude_set = set(exclude) if exclude else set()
        connections_to_send = [
            conn.websocket
            for conn_id, conn in self._connections.items()
            if conn_id not in exclude_set
        ]
        
        return await broadcast_to_connections(connections_to_send, message)
    
    async def broadcast_to_room(
        self,
        room_id: str,
        message: Dict[str, Any],
        exclude: Optional[List[str]] = None
    ) -> int:
        """
        Broadcast mensaje a una sala específica
        
        Args:
            room_id: ID de la sala
            message: Mensaje a enviar
            exclude: Lista de connection_ids a excluir (opcional)
        
        Returns:
            Número de mensajes enviados exitosamente
        """
        room = self._rooms.get(room_id)
        if not room:
            logger.warning(f"Room not found: {room_id}")
            return 0
        
        exclude_set = set(exclude) if exclude else set()
        connections_to_send = [
            self._connections[conn_id].websocket
            for conn_id in room.connections
            if conn_id in self._connections and conn_id not in exclude_set
        ]
        
        return await broadcast_to_connections(connections_to_send, message)
    
    def join_room(self, connection_id: str, room_id: str):
        """
        Agregar conexión a una sala
        
        Args:
            connection_id: ID de la conexión
            room_id: ID de la sala
        """
        if connection_id not in self._connections:
            raise ValueError(f"Connection not found: {connection_id}")
        
        if room_id not in self._rooms:
            self._rooms[room_id] = WebSocketRoom(room_id)
        
        self._rooms[room_id].add_connection(connection_id)
        logger.debug(f"Connection {connection_id} joined room {room_id}")
    
    def leave_room(self, connection_id: str, room_id: str):
        """
        Remover conexión de una sala
        
        Args:
            connection_id: ID de la conexión
            room_id: ID de la sala
        """
        if room_id in self._rooms:
            self._rooms[room_id].remove_connection(connection_id)
            
            # Limpiar sala vacía
            if self._rooms[room_id].is_empty():
                del self._rooms[room_id]
            
            logger.debug(f"Connection {connection_id} left room {room_id}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """
        Registra handler para tipo de mensaje
        
        Args:
            message_type: Tipo de mensaje
            handler: Función handler
        """
        self._handlers[message_type] = handler
        logger.info(f"Registered WebSocket handler: {message_type}")
    
    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Maneja mensaje recibido
        
        Args:
            websocket: Conexión WebSocket
            message: Mensaje recibido
        """
        message_type = message.get("type")
        handler = self._handlers.get(message_type)
        
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(websocket, message)
                else:
                    handler(websocket, message)
            except Exception as e:
                logger.error(f"Error in WebSocket handler: {e}")
        else:
            logger.warning(f"No handler for message type: {message_type}")
    
    async def handle_connection(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Maneja conexión WebSocket
        
        Args:
            websocket: Conexión WebSocket
            user_id: ID del usuario (opcional)
            metadata: Metadata adicional (opcional)
        """
        connection_id = await self.connect(websocket, user_id, metadata)
        
        try:
            while True:
                data = await receive_json_safe(websocket)
                if data is None:
                    break
                
                await self.handle_message(connection_id, data)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
        finally:
            await self.disconnect(connection_id)
    
    def get_router(self) -> APIRouter:
        """Retorna router con endpoints WebSocket"""
        @self.router.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_connection(websocket)
        
        return self.router
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del WebSocket manager"""
        connection_stats = [
            conn.get_stats()
            for conn in self._connections.values()
        ]
        
        room_stats = [
            room.get_stats()
            for room in self._rooms.values()
        ]
        
        return {
            "active_connections": len(self._connections),
            "total_rooms": len(self._rooms),
            "handlers": list(self._handlers.keys()),
            "connections": connection_stats,
            "rooms": room_stats,
            "heartbeat_enabled": self.enable_heartbeat,
        }
    
    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """
        Obtener conexión por ID
        
        Args:
            connection_id: ID de la conexión
        
        Returns:
            WebSocketConnection o None
        """
        return self._connections.get(connection_id)
    
    def get_room(self, room_id: str) -> Optional[WebSocketRoom]:
        """
        Obtener sala por ID
        
        Args:
            room_id: ID de la sala
        
        Returns:
            WebSocketRoom o None
        """
        return self._rooms.get(room_id)
    
    async def start(self):
        """Iniciar gestor (heartbeat, etc.)"""
        if self.heartbeat:
            await self.heartbeat.start()
    
    async def stop(self):
        """Detener gestor (heartbeat, etc.)"""
        if self.heartbeat:
            await self.heartbeat.stop()
        
        # Desconectar todas las conexiones
        connection_ids = list(self._connections.keys())
        for connection_id in connection_ids:
            await self.disconnect(connection_id)

