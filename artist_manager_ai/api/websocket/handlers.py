"""
WebSocket Handlers
==================

Handlers para WebSocket.
"""

import json
import logging
import uuid
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from .manager import WebSocketManager

logger = logging.getLogger(__name__)
manager = WebSocketManager()


async def handle_connection(websocket: WebSocket, user_id: Optional[str] = None):
    """
    Manejar nueva conexión.
    
    Args:
        websocket: Conexión WebSocket
        user_id: ID del usuario
    """
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id, user_id)
    
    try:
        # Enviar mensaje de bienvenida
        await manager.send_personal_message({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "timestamp": str(uuid.uuid4())
        }, client_id)
        
        # Escuchar mensajes
        while True:
            data = await websocket.receive_text()
            await handle_message(client_id, data, user_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id, user_id)
    except Exception as e:
        logger.error(f"Error in connection handler: {str(e)}")
        manager.disconnect(client_id, user_id)


async def handle_message(client_id: str, message: str, user_id: Optional[str] = None):
    """
    Manejar mensaje recibido.
    
    Args:
        client_id: ID del cliente
        message: Mensaje recibido
        user_id: ID del usuario
    """
    try:
        data = json.loads(message)
        message_type = data.get("type")
        
        if message_type == "ping":
            # Responder ping
            await manager.send_personal_message({
                "type": "pong",
                "timestamp": str(uuid.uuid4())
            }, client_id)
        
        elif message_type == "subscribe":
            # Suscribirse a eventos
            event_type = data.get("event_type")
            await manager.send_personal_message({
                "type": "subscribed",
                "event_type": event_type,
                "timestamp": str(uuid.uuid4())
            }, client_id)
        
        elif message_type == "unsubscribe":
            # Desuscribirse de eventos
            event_type = data.get("event_type")
            await manager.send_personal_message({
                "type": "unsubscribed",
                "event_type": event_type,
                "timestamp": str(uuid.uuid4())
            }, client_id)
        
        else:
            # Mensaje desconocido
            await manager.send_personal_message({
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": str(uuid.uuid4())
            }, client_id)
    
    except json.JSONDecodeError:
        await manager.send_personal_message({
            "type": "error",
            "message": "Invalid JSON",
            "timestamp": str(uuid.uuid4())
        }, client_id)
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        await manager.send_personal_message({
            "type": "error",
            "message": str(e),
            "timestamp": str(uuid.uuid4())
        }, client_id)




