"""
WebSocket API - Streaming bidireccional
========================================

API WebSocket para streaming en tiempo real bidireccional.
"""

import asyncio
import json
import logging
from typing import Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.routing import APIRouter

from ..core.chat_engine import ContinuousChatEngine
from ..core.chat_session import ChatSession, ChatState

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Gestor de conexiones WebSocket."""
    
    def __init__(self, chat_engine: ContinuousChatEngine):
        self.chat_engine = chat_engine
        self.active_connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, session_id: str, websocket: WebSocket):
        """Conectar un WebSocket."""
        await websocket.accept()
        
        async with self._lock:
            self.active_connections[session_id] = websocket
        
        logger.info(f"WebSocket connected for session: {session_id}")
        
        # Enviar estado inicial
        session = self.chat_engine.get_session(session_id)
        if session:
            await self.send_message(session_id, {
                "type": "session_state",
                "data": session.to_dict(),
            })
    
    async def disconnect(self, session_id: str):
        """Desconectar un WebSocket."""
        async with self._lock:
            if session_id in self.active_connections:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_message(self, session_id: str, message: Dict):
        """Enviar mensaje a través de WebSocket."""
        async with self._lock:
            websocket = self.active_connections.get(session_id)
        
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                await self.disconnect(session_id)
    
    async def broadcast_to_session(self, session_id: str, message: Dict):
        """Enviar mensaje a todos los WebSockets de una sesión."""
        await self.send_message(session_id, message)
    
    async def send_chunk(self, session_id: str, chunk: str, is_complete: bool = False):
        """Enviar chunk de respuesta."""
        await self.send_message(session_id, {
            "type": "chunk",
            "data": chunk,
            "complete": is_complete,
        })


def create_websocket_router(chat_engine: ContinuousChatEngine) -> APIRouter:
    """Crear router de WebSocket."""
    router = APIRouter()
    ws_manager = WebSocketManager(chat_engine)
    
    @router.websocket("/ws/chat/{session_id}")
    async def websocket_chat(session_id: str, websocket: WebSocket):
        """
        Endpoint WebSocket para chat en tiempo real.
        
        Mensajes recibidos:
        - {"type": "message", "content": "..."} - Enviar mensaje
        - {"type": "pause"} - Pausar chat
        - {"type": "resume"} - Reanudar chat
        - {"type": "stop"} - Detener chat
        
        Mensajes enviados:
        - {"type": "chunk", "data": "...", "complete": false} - Chunk de respuesta
        - {"type": "message", "role": "assistant", "content": "..."} - Mensaje completo
        - {"type": "session_state", "data": {...}} - Estado de sesión
        - {"type": "error", "message": "..."} - Error
        """
        # Verificar que la sesión existe
        session = chat_engine.get_session(session_id)
        if not session:
            await websocket.close(code=1008, reason="Session not found")
            return
        
        await ws_manager.connect(session_id, websocket)
        
        # Iniciar tarea para streaming de respuestas
        streaming_task = None
        
        try:
            # Iniciar chat continuo si no está activo
            if session.state == ChatState.IDLE:
                await chat_engine.start_continuous_chat(session_id)
            
            # Loop para recibir mensajes
            while True:
                try:
                    # Recibir mensaje del cliente
                    data = await websocket.receive_json()
                    message_type = data.get("type")
                    
                    if message_type == "message":
                        # Agregar mensaje del usuario
                        content = data.get("content", "")
                        await chat_engine.add_user_message(session_id, content)
                        
                        # Enviar confirmación
                        await ws_manager.send_message(session_id, {
                            "type": "message_received",
                            "content": content,
                        })
                    
                    elif message_type == "pause":
                        await chat_engine.pause_session(session_id)
                        await ws_manager.send_message(session_id, {
                            "type": "paused",
                        })
                    
                    elif message_type == "resume":
                        await chat_engine.resume_session(session_id)
                        await ws_manager.send_message(session_id, {
                            "type": "resumed",
                        })
                    
                    elif message_type == "stop":
                        await chat_engine.stop_session(session_id)
                        await ws_manager.send_message(session_id, {
                            "type": "stopped",
                        })
                        break
                    
                    elif message_type == "ping":
                        await ws_manager.send_message(session_id, {
                            "type": "pong",
                        })
                
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket message handling: {e}")
                    await ws_manager.send_message(session_id, {
                        "type": "error",
                        "message": str(e),
                    })
            
            # Monitorear nuevos mensajes y enviarlos
            last_message_count = len(session.messages)
            
            while True:
                await asyncio.sleep(0.5)  # Check cada 500ms
                
                # Verificar si hay nuevos mensajes
                current_message_count = len(session.messages)
                
                if current_message_count > last_message_count:
                    # Enviar nuevos mensajes
                    new_messages = session.messages[last_message_count:]
                    for msg in new_messages:
                        if msg.role == "assistant":
                            await ws_manager.send_message(session_id, {
                                "type": "message",
                                "role": msg.role,
                                "content": msg.content,
                                "timestamp": msg.timestamp.isoformat(),
                            })
                    
                    last_message_count = current_message_count
                
                # Verificar si la sesión se detuvo
                if session.is_stopped():
                    await ws_manager.send_message(session_id, {
                        "type": "stopped",
                    })
                    break
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_manager.send_message(session_id, {
                "type": "error",
                "message": str(e),
            })
        finally:
            if streaming_task:
                streaming_task.cancel()
                try:
                    await streaming_task
                except asyncio.CancelledError:
                    pass
            
            await ws_manager.disconnect(session_id)
    
    return router
































