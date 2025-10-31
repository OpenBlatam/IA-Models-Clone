"""
WebSocket Support for Real-time Communication
Sistema de WebSocket para comunicación en tiempo real ultra-optimizada
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

logger = logging.getLogger(__name__)


class WebSocketEventType(Enum):
    """Tipos de eventos WebSocket"""
    CONNECTION = "connection"
    DISCONNECTION = "disconnection"
    MESSAGE = "message"
    ANALYSIS_START = "analysis_start"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETE = "analysis_complete"
    SIMILARITY_START = "similarity_start"
    SIMILARITY_PROGRESS = "similarity_progress"
    SIMILARITY_COMPLETE = "similarity_complete"
    QUALITY_START = "quality_start"
    QUALITY_PROGRESS = "quality_progress"
    QUALITY_COMPLETE = "quality_complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SYSTEM_STATUS = "system_status"
    AI_PREDICTION = "ai_prediction"
    REAL_TIME_UPDATE = "real_time_update"


@dataclass
class WebSocketMessage:
    """Mensaje WebSocket"""
    event_type: WebSocketEventType
    data: Dict[str, Any]
    timestamp: float
    message_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class WebSocketConnection:
    """Conexión WebSocket"""
    websocket: WebSocket
    user_id: Optional[str]
    session_id: Optional[str]
    connected_at: float
    last_activity: float
    subscriptions: List[str]
    is_active: bool = True


class WebSocketManager:
    """Manager de conexiones WebSocket"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.rooms: Dict[str, List[str]] = {}
        self.message_handlers: Dict[WebSocketEventType, List[Callable]] = {}
        self.heartbeat_interval = 30  # segundos
        self.max_connections = 1000
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None, 
                     session_id: Optional[str] = None) -> str:
        """Conectar nuevo WebSocket"""
        async with self._lock:
            if len(self.connections) >= self.max_connections:
                await websocket.close(code=1013, reason="Server overloaded")
                raise Exception("Maximum connections reached")
            
            connection_id = f"conn_{int(time.time())}_{id(websocket)}"
            connection = WebSocketConnection(
                websocket=websocket,
                user_id=user_id,
                session_id=session_id,
                connected_at=time.time(),
                last_activity=time.time(),
                subscriptions=[]
            )
            
            self.connections[connection_id] = connection
            
            # Enviar mensaje de conexión
            await self._send_message(connection_id, WebSocketEventType.CONNECTION, {
                "connection_id": connection_id,
                "message": "Connected successfully",
                "server_time": time.time()
            })
            
            logger.info(f"WebSocket connected: {connection_id}")
            return connection_id
    
    async def disconnect(self, connection_id: str):
        """Desconectar WebSocket"""
        async with self._lock:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                connection.is_active = False
                
                # Remover de todas las rooms
                for room_id, connections in self.rooms.items():
                    if connection_id in connections:
                        connections.remove(connection_id)
                
                del self.connections[connection_id]
                logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_message(self, connection_id: str, event_type: WebSocketEventType, 
                          data: Dict[str, Any]):
        """Enviar mensaje a conexión específica"""
        await self._send_message(connection_id, event_type, data)
    
    async def broadcast_message(self, event_type: WebSocketEventType, 
                               data: Dict[str, Any], room_id: Optional[str] = None):
        """Broadcast mensaje a todas las conexiones o room específica"""
        if room_id and room_id in self.rooms:
            connection_ids = self.rooms[room_id]
        else:
            connection_ids = list(self.connections.keys())
        
        for connection_id in connection_ids:
            await self._send_message(connection_id, event_type, data)
    
    async def join_room(self, connection_id: str, room_id: str):
        """Unir conexión a room"""
        async with self._lock:
            if connection_id in self.connections:
                if room_id not in self.rooms:
                    self.rooms[room_id] = []
                
                if connection_id not in self.rooms[room_id]:
                    self.rooms[room_id].append(connection_id)
                    self.connections[connection_id].subscriptions.append(room_id)
                    
                    await self._send_message(connection_id, WebSocketEventType.MESSAGE, {
                        "message": f"Joined room: {room_id}",
                        "room_id": room_id
                    })
    
    async def leave_room(self, connection_id: str, room_id: str):
        """Salir de room"""
        async with self._lock:
            if room_id in self.rooms and connection_id in self.rooms[room_id]:
                self.rooms[room_id].remove(connection_id)
                if connection_id in self.connections:
                    self.connections[connection_id].subscriptions.remove(room_id)
                
                await self._send_message(connection_id, WebSocketEventType.MESSAGE, {
                    "message": f"Left room: {room_id}",
                    "room_id": room_id
                })
    
    async def _send_message(self, connection_id: str, event_type: WebSocketEventType, 
                           data: Dict[str, Any]):
        """Enviar mensaje interno"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        if not connection.is_active:
            return
        
        try:
            message = WebSocketMessage(
                event_type=event_type,
                data=data,
                timestamp=time.time(),
                message_id=f"msg_{int(time.time())}_{id(data)}",
                user_id=connection.user_id,
                session_id=connection.session_id
            )
            
            await connection.websocket.send_text(json.dumps({
                "event_type": message.event_type.value,
                "data": message.data,
                "timestamp": message.timestamp,
                "message_id": message.message_id,
                "user_id": message.user_id,
                "session_id": message.session_id
            }))
            
            connection.last_activity = time.time()
            
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            await self.disconnect(connection_id)
    
    def register_handler(self, event_type: WebSocketEventType, handler: Callable):
        """Registrar handler para tipo de evento"""
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []
        self.message_handlers[event_type].append(handler)
    
    async def handle_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Manejar mensaje recibido"""
        event_type_str = message_data.get("event_type")
        if not event_type_str:
            return
        
        try:
            event_type = WebSocketEventType(event_type_str)
            data = message_data.get("data", {})
            
            # Ejecutar handlers registrados
            if event_type in self.message_handlers:
                for handler in self.message_handlers[event_type]:
                    try:
                        await handler(connection_id, data)
                    except Exception as e:
                        logger.error(f"Error in WebSocket handler: {e}")
            
            # Actualizar actividad
            if connection_id in self.connections:
                self.connections[connection_id].last_activity = time.time()
                
        except ValueError:
            logger.warning(f"Unknown WebSocket event type: {event_type_str}")
    
    async def start_heartbeat(self):
        """Iniciar heartbeat para mantener conexiones vivas"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                current_time = time.time()
                inactive_connections = []
                
                for connection_id, connection in self.connections.items():
                    if current_time - connection.last_activity > self.heartbeat_interval * 2:
                        inactive_connections.append(connection_id)
                    else:
                        await self._send_message(connection_id, WebSocketEventType.HEARTBEAT, {
                            "server_time": current_time,
                            "connection_id": connection_id
                        })
                
                # Limpiar conexiones inactivas
                for connection_id in inactive_connections:
                    await self.disconnect(connection_id)
                    
            except Exception as e:
                logger.error(f"Error in WebSocket heartbeat: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de conexiones"""
        active_connections = sum(1 for conn in self.connections.values() if conn.is_active)
        total_rooms = len(self.rooms)
        
        return {
            "total_connections": len(self.connections),
            "active_connections": active_connections,
            "total_rooms": total_rooms,
            "max_connections": self.max_connections,
            "heartbeat_interval": self.heartbeat_interval
        }


# Instancia global del manager
websocket_manager = WebSocketManager()


class WebSocketRouter:
    """Router para endpoints WebSocket"""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Configurar rutas WebSocket"""
        
        @self.router.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Endpoint principal WebSocket"""
            await websocket.accept()
            connection_id = None
            
            try:
                connection_id = await websocket_manager.connect(websocket)
                
                while True:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    await websocket_manager.handle_message(connection_id, message_data)
                    
            except WebSocketDisconnect:
                if connection_id:
                    await websocket_manager.disconnect(connection_id)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if connection_id:
                    await websocket_manager.disconnect(connection_id)
        
        @self.router.websocket("/ws/{room_id}")
        async def websocket_room_endpoint(websocket: WebSocket, room_id: str):
            """Endpoint WebSocket para room específica"""
            await websocket.accept()
            connection_id = None
            
            try:
                connection_id = await websocket_manager.connect(websocket)
                await websocket_manager.join_room(connection_id, room_id)
                
                while True:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    await websocket_manager.handle_message(connection_id, message_data)
                    
            except WebSocketDisconnect:
                if connection_id:
                    await websocket_manager.leave_room(connection_id, room_id)
                    await websocket_manager.disconnect(connection_id)
            except Exception as e:
                logger.error(f"WebSocket room error: {e}")
                if connection_id:
                    await websocket_manager.leave_room(connection_id, room_id)
                    await websocket_manager.disconnect(connection_id)
        
        @self.router.get("/ws/stats")
        async def get_websocket_stats():
            """Obtener estadísticas WebSocket"""
            return websocket_manager.get_connection_stats()
        
        @self.router.post("/ws/broadcast")
        async def broadcast_message_endpoint(message_data: dict):
            """Broadcast mensaje a todas las conexiones"""
            event_type_str = message_data.get("event_type")
            data = message_data.get("data", {})
            room_id = message_data.get("room_id")
            
            if not event_type_str:
                return {"error": "event_type is required"}
            
            try:
                event_type = WebSocketEventType(event_type_str)
                await websocket_manager.broadcast_message(event_type, data, room_id)
                return {"message": "Message broadcasted successfully"}
            except ValueError:
                return {"error": "Invalid event_type"}
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                return {"error": "Failed to broadcast message"}


# Instancia del router
websocket_router = WebSocketRouter()


# Handlers de ejemplo para diferentes tipos de eventos
async def handle_analysis_start(connection_id: str, data: Dict[str, Any]):
    """Handler para inicio de análisis"""
    logger.info(f"Analysis started for connection {connection_id}")
    await websocket_manager.send_message(connection_id, WebSocketEventType.ANALYSIS_PROGRESS, {
        "progress": 0,
        "message": "Analysis started"
    })


async def handle_analysis_progress(connection_id: str, data: Dict[str, Any]):
    """Handler para progreso de análisis"""
    progress = data.get("progress", 0)
    logger.info(f"Analysis progress {progress}% for connection {connection_id}")
    await websocket_manager.send_message(connection_id, WebSocketEventType.ANALYSIS_PROGRESS, {
        "progress": progress,
        "message": f"Analysis {progress}% complete"
    })


async def handle_analysis_complete(connection_id: str, data: Dict[str, Any]):
    """Handler para análisis completado"""
    logger.info(f"Analysis completed for connection {connection_id}")
    await websocket_manager.send_message(connection_id, WebSocketEventType.ANALYSIS_COMPLETE, {
        "message": "Analysis completed successfully",
        "result": data.get("result", {})
    })


async def handle_ai_prediction(connection_id: str, data: Dict[str, Any]):
    """Handler para predicción AI"""
    logger.info(f"AI prediction for connection {connection_id}")
    await websocket_manager.send_message(connection_id, WebSocketEventType.AI_PREDICTION, {
        "prediction": data.get("prediction", {}),
        "confidence": data.get("confidence", 0.0),
        "model_name": data.get("model_name", "unknown")
    })


# Registrar handlers
websocket_manager.register_handler(WebSocketEventType.ANALYSIS_START, handle_analysis_start)
websocket_manager.register_handler(WebSocketEventType.ANALYSIS_PROGRESS, handle_analysis_progress)
websocket_manager.register_handler(WebSocketEventType.ANALYSIS_COMPLETE, handle_analysis_complete)
websocket_manager.register_handler(WebSocketEventType.AI_PREDICTION, handle_ai_prediction)


# Función para iniciar heartbeat en background
async def start_websocket_heartbeat():
    """Iniciar heartbeat WebSocket en background"""
    asyncio.create_task(websocket_manager.start_heartbeat())
    logger.info("WebSocket heartbeat started")


# Función para enviar actualizaciones en tiempo real
async def send_realtime_update(event_type: WebSocketEventType, data: Dict[str, Any], 
                              room_id: Optional[str] = None):
    """Enviar actualización en tiempo real"""
    await websocket_manager.broadcast_message(event_type, data, room_id)


# Función para enviar notificación de sistema
async def send_system_notification(message: str, level: str = "info", 
                                  room_id: Optional[str] = None):
    """Enviar notificación del sistema"""
    await websocket_manager.broadcast_message(WebSocketEventType.SYSTEM_STATUS, {
        "message": message,
        "level": level,
        "timestamp": time.time()
    }, room_id)


logger.info("WebSocket support module loaded successfully")

