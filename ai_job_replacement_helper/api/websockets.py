"""
WebSocket endpoints for real-time features
"""

import logging
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manager de conexiones WebSocket"""
    
    def __init__(self):
        """Inicializar manager"""
        self.active_connections: Dict[str, List[WebSocket]] = {}  # user_id -> [websockets]
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Conectar WebSocket"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        logger.info(f"WebSocket connected for user {user_id}")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Desconectar WebSocket"""
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Enviar mensaje personal"""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast a todos los usuarios"""
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")


# Instancia global
connection_manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Endpoint WebSocket principal"""
    await connection_manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Procesar mensaje
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "notification":
                # Reenviar notificación
                await connection_manager.send_personal_message(
                    message, user_id
                )
    
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, user_id)




