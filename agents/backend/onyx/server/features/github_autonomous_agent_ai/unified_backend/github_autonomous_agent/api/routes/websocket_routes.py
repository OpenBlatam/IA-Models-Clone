"""
WebSocket routes para actualizaciones en tiempo real.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
from datetime import datetime

from config.logging_config import get_logger
from config.di_setup import get_service

logger = get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manager para conexiones WebSocket."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connections_by_type: Dict[str, Set[WebSocket]] = {
            "tasks": set(),
            "agent": set(),
            "all": set()
        }
    
    async def connect(self, websocket: WebSocket, connection_type: str = "all"):
        """Aceptar nueva conexión."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connections_by_type[connection_type].add(websocket)
        logger.info(f"WebSocket connected: {connection_type} (total: {len(self.active_connections)})")
    
    def disconnect(self, websocket: WebSocket, connection_type: str = "all"):
        """Desconectar cliente."""
        self.active_connections.discard(websocket)
        self.connections_by_type[connection_type].discard(websocket)
        logger.info(f"WebSocket disconnected: {connection_type} (total: {len(self.active_connections)})")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Enviar mensaje a un cliente específico."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict, connection_type: str = "all"):
        """Enviar mensaje a todos los clientes de un tipo."""
        disconnected = set()
        for connection in self.connections_by_type.get(connection_type, set()):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)
        
        # Limpiar conexiones desconectadas
        for conn in disconnected:
            self.disconnect(conn, connection_type)
    
    async def broadcast_to_all(self, message: dict):
        """Enviar mensaje a todos los clientes."""
        await self.broadcast(message, "all")


# Manager global
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket principal."""
    await manager.connect(websocket, "all")
    
    try:
        while True:
            # Recibir mensajes del cliente
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "subscribe":
                    # Cliente se suscribe a un tipo de actualizaciones
                    subscribe_type = message.get("data", {}).get("type", "all")
                    await manager.connect(websocket, subscribe_type)
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "data": {"type": subscribe_type},
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                
                elif message_type == "ping":
                    # Heartbeat
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "data": {"message": "Invalid JSON"},
                    "timestamp": datetime.now().isoformat()
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "all")
        logger.info("WebSocket client disconnected")


@router.websocket("/ws/tasks")
async def websocket_tasks(websocket: WebSocket):
    """WebSocket para actualizaciones de tareas."""
    await manager.connect(websocket, "tasks")
    
    try:
        while True:
            data = await websocket.receive_text()
            # Echo para mantener conexión viva
            if data == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket, "tasks")


@router.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    """WebSocket para actualizaciones del agente."""
    await manager.connect(websocket, "agent")
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket, "agent")


# Funciones helper para broadcasting desde otros módulos
async def broadcast_task_update(task: dict):
    """Broadcast actualización de tarea."""
    await manager.broadcast({
        "type": "task_update",
        "data": task,
        "timestamp": datetime.now().isoformat()
    }, "tasks")


async def broadcast_agent_status(status: dict):
    """Broadcast estado del agente."""
    await manager.broadcast({
        "type": "agent_status",
        "data": status,
        "timestamp": datetime.now().isoformat()
    }, "agent")



