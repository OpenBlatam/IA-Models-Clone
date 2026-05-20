"""
WebSocket - Soporte para operaciones en tiempo real
"""

import logging
import json
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from ..core.editor import ContentEditor

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Gestor de conexiones WebSocket"""

    def __init__(self, editor: ContentEditor):
        """
        Inicializar el gestor WebSocket.

        Args:
            editor: Instancia del editor de contenido
        """
        self.editor = editor
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Conectar un cliente WebSocket.

        Args:
            websocket: Conexión WebSocket
            client_id: ID del cliente
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Cliente WebSocket conectado: {client_id}")

    def disconnect(self, client_id: str):
        """
        Desconectar un cliente.

        Args:
            client_id: ID del cliente
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Cliente WebSocket desconectado: {client_id}")

    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """
        Enviar mensaje a un cliente específico.

        Args:
            message: Mensaje a enviar
            client_id: ID del cliente
        """
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error enviando mensaje a {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: Dict[str, Any]):
        """
        Enviar mensaje a todos los clientes conectados.

        Args:
            message: Mensaje a enviar
        """
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error en broadcast a {client_id}: {e}")
                disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)


async def websocket_endpoint(websocket: WebSocket, editor: ContentEditor):
    """
    Endpoint WebSocket principal.

    Args:
        websocket: Conexión WebSocket
        editor: Instancia del editor
    """
    import uuid
    client_id = str(uuid.uuid4())
    manager = WebSocketManager(editor)
    
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "message": "Conexión establecida"
        })
        
        while True:
            data = await websocket.receive_json()
            operation = data.get("operation")
            
            if operation == "add":
                result = await editor.add(
                    content=data.get("content", ""),
                    addition=data.get("addition", ""),
                    position=data.get("position", "end"),
                    context=data.get("context")
                )
                await websocket.send_json({
                    "type": "result",
                    "operation": "add",
                    "result": result
                })
            
            elif operation == "remove":
                result = await editor.remove(
                    content=data.get("content", ""),
                    pattern=data.get("pattern"),
                    selector=data.get("selector"),
                    context=data.get("context")
                )
                await websocket.send_json({
                    "type": "result",
                    "operation": "remove",
                    "result": result
                })
            
            elif operation == "analyze":
                analysis = await editor.analyzer.analyze(
                    data.get("content", ""),
                    data.get("context")
                )
                await websocket.send_json({
                    "type": "result",
                    "operation": "analyze",
                    "result": {"analysis": analysis}
                })
            
            elif operation == "diff":
                diff_result = editor.diff.compute_diff(
                    data.get("original", ""),
                    data.get("modified", "")
                )
                await websocket.send_json({
                    "type": "result",
                    "operation": "diff",
                    "result": {"diff": diff_result}
                })
            
            elif operation == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": __import__("datetime").datetime.utcnow().isoformat()
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Operación desconocida: {operation}"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Cliente desconectado: {client_id}")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        manager.disconnect(client_id)






