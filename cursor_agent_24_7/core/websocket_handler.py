"""
WebSocket Handler - Comunicación en tiempo real
================================================

Maneja conexiones WebSocket para recibir comandos y enviar actualizaciones en tiempo real.
"""

import asyncio
import logging
import json
from typing import Dict, Set, Optional, Callable
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Gestiona conexiones WebSocket"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Aceptar nueva conexión WebSocket"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        metadata = {
            "client_id": client_id or f"client_{datetime.now().timestamp()}",
            "connected_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        self.connection_metadata[websocket] = metadata
        
        logger.info(f"🔌 WebSocket connected: {metadata['client_id']}")
        
        # Enviar mensaje de bienvenida
        await self.send_personal_message({
            "type": "connected",
            "client_id": metadata["client_id"],
            "message": "Connected to Cursor Agent 24/7"
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
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Enviar mensaje a un WebSocket específico"""
        try:
            # Usar orjson si está disponible
            try:
                import orjson
                await websocket.send_bytes(orjson.dumps(message))
            except ImportError:
                await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Enviar mensaje a todas las conexiones"""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                # Usar orjson si está disponible
                try:
                    import orjson
                    await connection.send_bytes(orjson.dumps(message))
                except ImportError:
                    await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Limpiar conexiones desconectadas
        for connection in disconnected:
            await self.disconnect(connection)
    
    def register_handler(self, message_type: str, handler: Callable):
        """Registrar handler para tipo de mensaje"""
        self.message_handlers[message_type] = handler
        logger.info(f"📝 Registered handler for message type: {message_type}")
    
    async def handle_message(self, websocket: WebSocket, message: dict):
        """Manejar mensaje recibido"""
        message_type = message.get("type")
        
        # Actualizar última actividad
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["last_activity"] = datetime.now().isoformat()
        
        # Buscar handler
        if message_type in self.message_handlers:
            try:
                handler = self.message_handlers[message_type]
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
            logger.warning(f"Unknown message type: {message_type}")
            await self.send_personal_message({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }, websocket)
    
    async def listen(self, websocket: WebSocket):
        """Escuchar mensajes de un WebSocket"""
        client_id = None
        
        try:
            client_id = await self.connect(websocket)
            
            while True:
                # Recibir mensaje
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self.handle_message(websocket, message)
                except json.JSONDecodeError:
                    # Intentar recibir como bytes
                    try:
                        data = await websocket.receive_bytes()
                        try:
                            import orjson
                            message = orjson.loads(data)
                        except ImportError:
                            message = json.loads(data.decode('utf-8'))
                        await self.handle_message(websocket, message)
                    except Exception as e:
                        logger.error(f"Error parsing message: {e}")
                        await self.send_personal_message({
                            "type": "error",
                            "message": "Invalid message format"
                        }, websocket)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
        finally:
            await self.disconnect(websocket)
    
    def get_connection_count(self) -> int:
        """Obtener número de conexiones activas"""
        return len(self.active_connections)
    
    def get_connections_info(self) -> list:
        """Obtener información de todas las conexiones"""
        return [
            {
                "client_id": meta.get("client_id"),
                "connected_at": meta.get("connected_at"),
                "last_activity": meta.get("last_activity")
            }
            for meta in self.connection_metadata.values()
        ]



