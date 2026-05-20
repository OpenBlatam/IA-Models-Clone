"""
WebSocket Helpers - Utilidades para WebSocket
==============================================

Funciones helper para facilitar el trabajo con WebSockets.
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from enum import Enum

logger = logging.getLogger(__name__)


class WebSocketState(str, Enum):
    """Estados de conexión WebSocket"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class WebSocketConnection:
    """
    Representa una conexión WebSocket con metadata.
    """
    
    def __init__(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar conexión WebSocket.
        
        Args:
            websocket: Conexión WebSocket
            connection_id: ID único de la conexión
            user_id: ID del usuario (opcional)
            metadata: Metadata adicional (opcional)
        """
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.state = WebSocketState.CONNECTING
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.message_count = 0
        self.error_count = 0
    
    def update_activity(self):
        """Actualizar timestamp de última actividad"""
        self.last_activity = datetime.utcnow()
        self.message_count += 1
    
    def record_error(self):
        """Registrar error"""
        self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la conexión"""
        return {
            "connection_id": self.connection_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "error_count": self.error_count,
            "metadata": self.metadata,
        }


class WebSocketRoom:
    """
    Sala de WebSocket para agrupar conexiones.
    """
    
    def __init__(self, room_id: str):
        """
        Inicializar sala.
        
        Args:
            room_id: ID único de la sala
        """
        self.room_id = room_id
        self.connections: Set[str] = set()
        self.created_at = datetime.utcnow()
    
    def add_connection(self, connection_id: str):
        """Agregar conexión a la sala"""
        self.connections.add(connection_id)
    
    def remove_connection(self, connection_id: str):
        """Remover conexión de la sala"""
        self.connections.discard(connection_id)
    
    def is_empty(self) -> bool:
        """Verificar si la sala está vacía"""
        return len(self.connections) == 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la sala"""
        return {
            "room_id": self.room_id,
            "connection_count": len(self.connections),
            "created_at": self.created_at.isoformat(),
        }


async def send_json_safe(
    websocket: WebSocket,
    data: Dict[str, Any],
    timeout: float = 5.0
) -> bool:
    """
    Enviar JSON de forma segura con timeout.
    
    Args:
        websocket: Conexión WebSocket
        data: Datos a enviar
        timeout: Timeout en segundos
    
    Returns:
        True si se envió exitosamente
    """
    try:
        await asyncio.wait_for(
            websocket.send_json(data),
            timeout=timeout
        )
        return True
    except asyncio.TimeoutError:
        logger.warning(f"Timeout sending message to WebSocket")
        return False
    except Exception as e:
        logger.error(f"Error sending message to WebSocket: {e}")
        return False


async def receive_json_safe(
    websocket: WebSocket,
    timeout: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Recibir JSON de forma segura con timeout.
    
    Args:
        websocket: Conexión WebSocket
        timeout: Timeout en segundos (opcional)
    
    Returns:
        Datos recibidos o None si hay error
    """
    try:
        if timeout:
            data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=timeout
            )
        else:
            data = await websocket.receive_json()
        return data
    except asyncio.TimeoutError:
        logger.debug(f"Timeout receiving message from WebSocket")
        return None
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        return None
    except Exception as e:
        logger.error(f"Error receiving message from WebSocket: {e}")
        return None


async def ping_websocket(
    websocket: WebSocket,
    timeout: float = 5.0
) -> bool:
    """
    Enviar ping a WebSocket para verificar conexión.
    
    Args:
        websocket: Conexión WebSocket
        timeout: Timeout en segundos
    
    Returns:
        True si el ping fue exitoso
    """
    try:
        await asyncio.wait_for(
            websocket.send_json({"type": "ping", "timestamp": datetime.utcnow().isoformat()}),
            timeout=timeout
        )
        return True
    except Exception as e:
        logger.debug(f"Ping failed: {e}")
        return False


def create_websocket_message(
    message_type: str,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Crear mensaje WebSocket estandarizado.
    
    Args:
        message_type: Tipo de mensaje
        data: Datos del mensaje (opcional)
        error: Mensaje de error (opcional)
    
    Returns:
        Mensaje WebSocket formateado
    """
    message = {
        "type": message_type,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if data:
        message["data"] = data
    
    if error:
        message["error"] = error
    
    return message


def validate_websocket_message(
    message: Dict[str, Any],
    required_fields: Optional[List[str]] = None
) -> tuple[bool, Optional[str]]:
    """
    Validar mensaje WebSocket.
    
    Args:
        message: Mensaje a validar
        required_fields: Campos requeridos (opcional)
    
    Returns:
        Tupla (is_valid, error_message)
    """
    if not isinstance(message, dict):
        return False, "Message must be a dictionary"
    
    if "type" not in message:
        return False, "Message must have 'type' field"
    
    if required_fields:
        missing = [field for field in required_fields if field not in message]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"
    
    return True, None


async def broadcast_to_connections(
    connections: List[WebSocket],
    message: Dict[str, Any],
    exclude: Optional[List[WebSocket]] = None
) -> int:
    """
    Broadcast mensaje a múltiples conexiones.
    
    Args:
        connections: Lista de conexiones
        message: Mensaje a enviar
        exclude: Conexiones a excluir (opcional)
    
    Returns:
        Número de mensajes enviados exitosamente
    """
    exclude_set = set(exclude) if exclude else set()
    sent = 0
    
    async def send_one(ws: WebSocket):
        if ws not in exclude_set:
            if await send_json_safe(ws, message):
                return 1
        return 0
    
    results = await asyncio.gather(
        *[send_one(ws) for ws in connections],
        return_exceptions=True
    )
    
    sent = sum(1 for r in results if r == 1)
    return sent


class WebSocketHeartbeat:
    """
    Gestor de heartbeat para WebSocket.
    """
    
    def __init__(
        self,
        interval: float = 30.0,
        timeout: float = 10.0
    ):
        """
        Inicializar heartbeat.
        
        Args:
            interval: Intervalo entre heartbeats en segundos
            timeout: Timeout para respuesta en segundos
        """
        self.interval = interval
        self.timeout = timeout
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def start_for_connection(
        self,
        connection_id: str,
        websocket: WebSocket,
        on_timeout: Optional[Callable[[str], None]] = None
    ):
        """
        Iniciar heartbeat para una conexión.
        
        Args:
            connection_id: ID de la conexión
            websocket: Conexión WebSocket
            on_timeout: Callback cuando hay timeout (opcional)
        """
        if connection_id in self._tasks:
            return
        
        async def heartbeat_loop():
            while self._running:
                try:
                    await asyncio.sleep(self.interval)
                    
                    # Enviar ping
                    if not await ping_websocket(websocket, self.timeout):
                        if on_timeout:
                            on_timeout(connection_id)
                        break
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in heartbeat loop: {e}")
                    break
        
        self._tasks[connection_id] = asyncio.create_task(heartbeat_loop())
    
    async def stop_for_connection(self, connection_id: str):
        """Detener heartbeat para una conexión"""
        if connection_id in self._tasks:
            task = self._tasks.pop(connection_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def start(self):
        """Iniciar todos los heartbeats"""
        self._running = True
    
    async def stop(self):
        """Detener todos los heartbeats"""
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

