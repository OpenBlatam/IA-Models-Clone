"""
WebSocket Manager
=================

Gestor de conexiones WebSocket.
"""

import json
import logging
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Gestor de conexiones WebSocket."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        self._logger = logger
    
    async def connect(self, websocket: WebSocket, client_id: str, user_id: Optional[str] = None):
        """
        Conectar cliente.
        
        Args:
            websocket: Conexión WebSocket
            client_id: ID del cliente
            user_id: ID del usuario (opcional)
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(client_id)
        
        self._logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str, user_id: Optional[str] = None):
        """
        Desconectar cliente.
        
        Args:
            client_id: ID del cliente
            user_id: ID del usuario (opcional)
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(client_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        self._logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """
        Enviar mensaje personal.
        
        Args:
            message: Mensaje
            client_id: ID del cliente
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                self._logger.error(f"Error sending message to {client_id}: {str(e)}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict, exclude: Optional[Set[str]] = None):
        """
        Broadcast a todos los clientes.
        
        Args:
            message: Mensaje
            exclude: IDs a excluir
        """
        exclude = exclude or set()
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id not in exclude:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    self._logger.error(f"Error broadcasting to {client_id}: {str(e)}")
                    disconnected.append(client_id)
        
        # Limpiar conexiones desconectadas
        for client_id in disconnected:
            self.disconnect(client_id)
    
    async def send_to_user(self, message: dict, user_id: str):
        """
        Enviar mensaje a usuario específico.
        
        Args:
            message: Mensaje
            user_id: ID del usuario
        """
        if user_id in self.user_connections:
            for client_id in self.user_connections[user_id]:
                await self.send_personal_message(message, client_id)
    
    def get_connection_count(self) -> int:
        """Obtener número de conexiones activas."""
        return len(self.active_connections)
    
    def get_user_connections(self, user_id: str) -> int:
        """Obtener número de conexiones de usuario."""
        return len(self.user_connections.get(user_id, set()))




