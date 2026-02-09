"""
WebSocket Manager System
========================

Sistema de gestión de conexiones WebSocket.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Estado de conexión."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class WebSocketConnection:
    """Conexión WebSocket."""
    connection_id: str
    client_id: Optional[str] = None
    status: ConnectionStatus = ConnectionStatus.CONNECTED
    connected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_message_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebSocketManager:
    """
    Gestor de WebSockets.
    
    Gestiona conexiones WebSocket y mensajes.
    """
    
    def __init__(self):
        """Inicializar gestor de WebSockets."""
        self.connections: Dict[str, Any] = {}  # connection_id -> websocket
        self.connection_info: Dict[str, WebSocketConnection] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.max_history = 10000
    
    def register_connection(
        self,
        connection_id: str,
        websocket: Any,
        client_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WebSocketConnection:
        """
        Registrar conexión WebSocket.
        
        Args:
            connection_id: ID único de conexión
            websocket: Objeto WebSocket
            client_id: ID del cliente
            metadata: Metadata adicional
            
        Returns:
            Información de conexión
        """
        connection_info = WebSocketConnection(
            connection_id=connection_id,
            client_id=client_id,
            status=ConnectionStatus.CONNECTED,
            metadata=metadata or {}
        )
        
        self.connections[connection_id] = websocket
        self.connection_info[connection_id] = connection_info
        
        logger.info(f"Registered WebSocket connection: {connection_id}")
        
        return connection_info
    
    def unregister_connection(self, connection_id: str) -> bool:
        """
        Desregistrar conexión.
        
        Args:
            connection_id: ID de conexión
            
        Returns:
            True si se desregistró, False si no existe
        """
        if connection_id in self.connections:
            del self.connections[connection_id]
            if connection_id in self.connection_info:
                self.connection_info[connection_id].status = ConnectionStatus.DISCONNECTED
            return True
        return False
    
    async def send_message(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Enviar mensaje a conexión específica.
        
        Args:
            connection_id: ID de conexión
            message: Mensaje a enviar
            
        Returns:
            True si se envió, False si no existe
        """
        if connection_id not in self.connections:
            return False
        
        try:
            websocket = self.connections[connection_id]
            await websocket.send_json(message)
            
            # Actualizar última mensaje
            if connection_id in self.connection_info:
                self.connection_info[connection_id].last_message_at = datetime.now().isoformat()
            
            # Registrar en historial
            self._record_message(connection_id, "sent", message)
            
            return True
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            if connection_id in self.connection_info:
                self.connection_info[connection_id].status = ConnectionStatus.ERROR
            return False
    
    async def broadcast_message(
        self,
        message: Dict[str, Any],
        exclude: Optional[Set[str]] = None
    ) -> int:
        """
        Enviar mensaje a todas las conexiones.
        
        Args:
            message: Mensaje a enviar
            exclude: IDs de conexiones a excluir
            
        Returns:
            Número de conexiones que recibieron el mensaje
        """
        exclude = exclude or set()
        sent_count = 0
        
        for connection_id, websocket in self.connections.items():
            if connection_id not in exclude:
                if await self.send_message(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def broadcast_to_clients(
        self,
        client_ids: List[str],
        message: Dict[str, Any]
    ) -> int:
        """
        Enviar mensaje a clientes específicos.
        
        Args:
            client_ids: IDs de clientes
            message: Mensaje a enviar
            
        Returns:
            Número de conexiones que recibieron el mensaje
        """
        sent_count = 0
        
        for connection_id, connection_info in self.connection_info.items():
            if connection_info.client_id in client_ids:
                if await self.send_message(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    def _record_message(
        self,
        connection_id: str,
        direction: str,
        message: Dict[str, Any]
    ) -> None:
        """Registrar mensaje en historial."""
        self.message_history.append({
            "connection_id": connection_id,
            "direction": direction,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
    
    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Obtener información de conexión."""
        return self.connection_info.get(connection_id)
    
    def list_connections(self) -> List[WebSocketConnection]:
        """Listar todas las conexiones."""
        return [
            info for info in self.connection_info.values()
            if info.status == ConnectionStatus.CONNECTED
        ]
    
    def get_connection_count(self) -> int:
        """Obtener número de conexiones activas."""
        return len([
            c for c in self.connection_info.values()
            if c.status == ConnectionStatus.CONNECTED
        ])
    
    def get_message_history(
        self,
        connection_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de mensajes.
        
        Args:
            connection_id: Filtrar por conexión
            limit: Límite de resultados
            
        Returns:
            Lista de mensajes
        """
        messages = self.message_history
        
        if connection_id:
            messages = [m for m in messages if m["connection_id"] == connection_id]
        
        return messages[-limit:]


# Instancia global
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Obtener instancia global del gestor de WebSockets."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager






