"""
WebSocket API para streaming en tiempo real y notificaciones
"""

import logging
import json
import asyncio
from typing import Dict, Set
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.routing import WebSocketRoute

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Gestor de conexiones WebSocket"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Conecta un cliente WebSocket"""
        await websocket.accept()
        key = user_id or "anonymous"
        if key not in self.active_connections:
            self.active_connections[key] = set()
        self.active_connections[key].add(websocket)
        logger.info(f"WebSocket connected: {key} (total: {len(self.active_connections[key])})")
    
    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """Desconecta un cliente WebSocket"""
        key = user_id or "anonymous"
        if key in self.active_connections:
            self.active_connections[key].discard(websocket)
            if not self.active_connections[key]:
                del self.active_connections[key]
        logger.info(f"WebSocket disconnected: {key}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Envía un mensaje personal a un WebSocket específico"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def broadcast_to_user(self, message: dict, user_id: str):
        """Envía un mensaje a todos los WebSockets de un usuario"""
        if user_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to user {user_id}: {e}")
                    disconnected.add(connection)
            
            # Limpiar conexiones desconectadas
            for conn in disconnected:
                self.active_connections[user_id].discard(conn)
    
    async def broadcast(self, message: dict):
        """Envía un mensaje a todos los WebSockets conectados"""
        disconnected = {}
        for user_id, connections in self.active_connections.items():
            disconnected[user_id] = set()
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")
                    disconnected[user_id].add(connection)
        
        # Limpiar conexiones desconectadas
        for user_id, conns in disconnected.items():
            for conn in conns:
                if user_id in self.active_connections:
                    self.active_connections[user_id].discard(conn)


# Instancia global del gestor de conexiones
manager = ConnectionManager()


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Endpoint WebSocket para un usuario específico"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Recibir mensajes del cliente
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "ping":
                    # Responder a ping
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, websocket)
                
                elif message_type == "subscribe":
                    # Suscribirse a actualizaciones de una canción
                    song_id = message.get("song_id")
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "song_id": song_id,
                        "message": f"Subscribed to updates for song {song_id}"
                    }, websocket)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected: {user_id}")


@router.websocket("/ws")
async def websocket_anonymous(websocket: WebSocket):
    """Endpoint WebSocket para usuarios anónimos"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, websocket)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Función helper para notificar a usuarios
async def notify_song_status(user_id: str, song_id: str, status: str, message: str = None):
    """Notifica el estado de una canción a través de WebSocket"""
    notification = {
        "type": "song_status",
        "song_id": song_id,
        "status": status,
        "message": message or f"Song {song_id} status: {status}"
    }
    await manager.broadcast_to_user(notification, user_id)


# Función helper para notificar progreso
async def notify_generation_progress(user_id: str, song_id: str, progress: float, message: str = None):
    """Notifica el progreso de generación"""
    notification = {
        "type": "generation_progress",
        "song_id": song_id,
        "progress": progress,  # 0.0 - 1.0
        "message": message or f"Generation progress: {progress * 100:.1f}%",
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_user(notification, user_id)


# Función helper para notificar inicio de generación
async def notify_generation_started(user_id: str, song_id: str, prompt: str = None):
    """Notifica el inicio de una generación"""
    notification = {
        "type": "generation_started",
        "song_id": song_id,
        "prompt": prompt,
        "message": "Generation started",
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_user(notification, user_id)


# Función helper para notificar finalización de generación
async def notify_generation_completed(user_id: str, song_id: str, audio_url: str = None, metadata: dict = None):
    """Notifica la finalización de una generación"""
    notification = {
        "type": "generation_completed",
        "song_id": song_id,
        "audio_url": audio_url,
        "metadata": metadata or {},
        "message": "Generation completed",
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_user(notification, user_id)


# Función helper para notificar errores
async def notify_generation_error(user_id: str, song_id: str, error_message: str):
    """Notifica un error en la generación"""
    notification = {
        "type": "generation_error",
        "song_id": song_id,
        "error": error_message,
        "message": f"Generation error: {error_message}",
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_user(notification, user_id)


@router.websocket("/ws/generate/{user_id}")
async def websocket_generate_stream(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint para streaming de generación en tiempo real.
    
    Permite recibir actualizaciones de progreso durante la generación de música.
    """
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "start_generation":
                    # Iniciar generación
                    prompt = message.get("prompt")
                    duration = message.get("duration", 30)
                    song_id = message.get("song_id")
                    
                    if not song_id:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "song_id is required"
                        }, websocket)
                        continue
                    
                    # Notificar inicio
                    await notify_generation_started(user_id, song_id, prompt)
                    
                    # Aquí se integraría con el sistema de generación real
                    # Por ahora, simulamos el progreso
                    for progress in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                        await asyncio.sleep(0.5)  # Simular procesamiento
                        await notify_generation_progress(
                            user_id, 
                            song_id, 
                            progress,
                            f"Processing... {progress * 100:.0f}%"
                        )
                    
                    # Notificar finalización
                    await notify_generation_completed(
                        user_id,
                        song_id,
                        audio_url=f"/suno/songs/{song_id}/audio",
                        metadata={"duration": duration, "prompt": prompt}
                    )
                
                elif message_type == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, websocket)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected: {user_id}")
