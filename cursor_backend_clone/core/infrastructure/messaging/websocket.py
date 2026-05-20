"""
WebSocket Handler - Comunicación en tiempo real
================================================

Maneja conexiones WebSocket para recibir comandos y enviar actualizaciones en tiempo real.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Set, Optional, Callable
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Gestiona conexiones WebSocket con mejoras de rendimiento y seguridad.
    
    Características:
    - Límite de conexiones simultáneas
    - Heartbeat para detectar conexiones muertas
    - Manejo robusto de errores
    - Métricas de conexiones
    """
    
    def __init__(self, max_connections: int = 100, heartbeat_interval: float = 30.0):
        """
        Inicializar WebSocket Manager.
        
        Args:
            max_connections: Número máximo de conexiones simultáneas
            heartbeat_interval: Intervalo de heartbeat en segundos
        """
        from .constants import MAX_WEBSOCKET_CONNECTIONS, DEFAULT_HEARTBEAT_INTERVAL
        
        self.max_connections = max_connections if max_connections else MAX_WEBSOCKET_CONNECTIONS
        self.heartbeat_interval = heartbeat_interval if heartbeat_interval else DEFAULT_HEARTBEAT_INTERVAL
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """
        Aceptar nueva conexión WebSocket con validación.
        
        Args:
            websocket: Conexión WebSocket
            client_id: ID opcional del cliente
            
        Returns:
            ID del cliente conectado
            
        Raises:
            ConnectionError: Si se excede el límite de conexiones
        """
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Maximum connections reached")
            raise ConnectionError(f"Maximum connections ({self.max_connections}) reached")
        
        await websocket.accept()
        self.active_connections.add(websocket)
        
        metadata = {
            "client_id": client_id or f"client_{int(time.time() * 1000)}",
            "connected_at": datetime.now().isoformat(),
            "last_activity": datetime.now(),
            "message_count": 0,
            "bytes_sent": 0,
            "bytes_received": 0
        }
        self.connection_metadata[websocket] = metadata
        
        logger.info(f"🔌 WebSocket connected: {metadata['client_id']} ({len(self.active_connections)}/{self.max_connections})")
        
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        await self.send_personal_message({
            "type": "connected",
            "client_id": metadata["client_id"],
            "message": "Connected to Cursor Agent 24/7",
            "heartbeat_interval": self.heartbeat_interval
        }, websocket)
        
        return metadata["client_id"]
    
    async def disconnect(self, websocket: WebSocket):
        """Desconectar WebSocket"""
        if websocket in self.active_connections:
            metadata = self.connection_metadata.get(websocket, {})
            client_id = metadata.get("client_id", "unknown")
            
            self.active_connections.remove(websocket)
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
            
            logger.info(f"🔌 WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket) -> bool:
        """
        Enviar mensaje a un WebSocket específico.
        
        Args:
            message: Mensaje a enviar
            websocket: Conexión WebSocket destino
            
        Returns:
            True si se envió exitosamente, False en caso contrario
        """
        if websocket not in self.active_connections:
            return False
        
        try:
            # Usar orjson si está disponible
            try:
                import orjson
                data = orjson.dumps(message)
                await websocket.send_bytes(data)
                bytes_sent = len(data)
            except ImportError:
                await websocket.send_json(message)
                bytes_sent = len(json.dumps(message).encode('utf-8'))
            
            # Actualizar metadata
            if websocket in self.connection_metadata:
                metadata = self.connection_metadata[websocket]
                metadata["last_activity"] = datetime.now()
                metadata["bytes_sent"] = metadata.get("bytes_sent", 0) + bytes_sent
            
            return True
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            await self.disconnect(websocket)
            return False
    
    async def broadcast(self, message: dict) -> int:
        """
        Enviar mensaje a todas las conexiones activas.
        
        Args:
            message: Mensaje a broadcast
            
        Returns:
            Número de conexiones que recibieron el mensaje exitosamente
        """
        disconnected = []
        success_count = 0
        
        for connection in list(self.active_connections):
            try:
                # Usar orjson si está disponible
                try:
                    import orjson
                    data = orjson.dumps(message)
                    await connection.send_bytes(data)
                    bytes_sent = len(data)
                except ImportError:
                    await connection.send_json(message)
                    bytes_sent = len(json.dumps(message).encode('utf-8'))
                
                success_count += 1
                
                # Actualizar metadata
                if connection in self.connection_metadata:
                    metadata = self.connection_metadata[connection]
                    metadata["last_activity"] = datetime.now()
                    metadata["bytes_sent"] = metadata.get("bytes_sent", 0) + bytes_sent
                    
            except Exception as e:
                logger.debug(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Limpiar conexiones desconectadas
        for connection in disconnected:
            await self.disconnect(connection)
        
        return success_count
    
    def register_handler(self, message_type: str, handler: Callable):
        """Registrar handler para tipo de mensaje"""
        self.message_handlers[message_type] = handler
        logger.info(f"📝 Registered handler for message type: {message_type}")
    
    async def _heartbeat_loop(self) -> None:
        """Loop de heartbeat para detectar conexiones muertas"""
        while self.active_connections:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                now = datetime.now()
                timeout_threshold = now - timedelta(seconds=self.heartbeat_interval * 3)
                
                dead_connections = []
                for websocket, metadata in list(self.connection_metadata.items()):
                    if metadata.get("last_activity", now) < timeout_threshold:
                        dead_connections.append(websocket)
                    else:
                        # Enviar ping
                        try:
                            await self.send_personal_message({
                                "type": "ping",
                                "timestamp": now.isoformat()
                            }, websocket)
                        except Exception:
                            dead_connections.append(websocket)
                
                # Limpiar conexiones muertas
                for websocket in dead_connections:
                    await self.disconnect(websocket)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def handle_message(self, websocket: WebSocket, message: dict) -> None:
        """
        Manejar mensaje recibido desde WebSocket.
        
        Args:
            websocket: Conexión WebSocket que envió el mensaje
            message: Mensaje recibido
        """
        # Actualizar metadata
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata[websocket]
            metadata["last_activity"] = datetime.now()
            metadata["message_count"] = metadata.get("message_count", 0) + 1
            if isinstance(message, (str, bytes)):
                metadata["bytes_received"] = metadata.get("bytes_received", 0) + len(message)
        
        message_type = message.get("type") if isinstance(message, dict) else None
        
        # Manejar pong
        if message_type == "pong":
            return
        
        if message_type and message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(websocket, message)
                else:
                    handler(websocket, message)
            except Exception as e:
                logger.error(f"Error in message handler for {message_type}: {e}")
                await self.send_personal_message({
                    "type": "error",
                    "message": f"Error processing {message_type}: {str(e)}"
                }, websocket)
        else:
            logger.debug(f"Unknown message type: {message_type}")
    
    async def listen(self, websocket: WebSocket) -> None:
        """
        Escuchar mensajes de un WebSocket.
        
        Args:
            websocket: Conexión WebSocket a escuchar
        """
        client_id = await self.connect(websocket)
        
        try:
            while websocket in self.active_connections:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self.heartbeat_interval * 2
                    )
                    
                    try:
                        message = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from {client_id}: {data[:100]}")
                        continue
                    
                    await self.handle_message(websocket, message)
                    
                except asyncio.TimeoutError:
                    # Enviar ping para verificar conexión
                    try:
                        await self.send_personal_message({
                            "type": "ping",
                            "timestamp": datetime.now().isoformat()
                        }, websocket)
                    except Exception:
                        break
                        
        except WebSocketDisconnect:
            logger.debug(f"WebSocket disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket listener for {client_id}: {e}")
        finally:
            await self.disconnect(websocket)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de conexiones WebSocket.
        
        Returns:
            Diccionario con estadísticas
        """
        total_messages = sum(
            m.get("message_count", 0) for m in self.connection_metadata.values()
        )
        total_bytes_sent = sum(
            m.get("bytes_sent", 0) for m in self.connection_metadata.values()
        )
        total_bytes_received = sum(
            m.get("bytes_received", 0) for m in self.connection_metadata.values()
        )
        
        return {
            "active_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            "total_messages": total_messages,
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received,
            "heartbeat_interval": self.heartbeat_interval
        }
    
    async def disconnect_all(self) -> None:
        """Desconectar todas las conexiones"""
        connections = list(self.active_connections)
        for websocket in connections:
            await self.disconnect(websocket)
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass


