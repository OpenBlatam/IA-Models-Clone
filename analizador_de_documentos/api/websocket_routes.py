"""
WebSocket Routes para Updates en Tiempo Real
=============================================

Endpoints WebSocket para actualizaciones en tiempo real.
"""

import logging
import json
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.routing import APIRoute

from ..core.document_analyzer import DocumentAnalyzer
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/ws",
    tags=["WebSocket"]
)


class ConnectionManager:
    """Gestor de conexiones WebSocket"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Conectar cliente"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Cliente conectado: {client_id}")
    
    def disconnect(self, client_id: str):
        """Desconectar cliente"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Cliente desconectado: {client_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Enviar mensaje a cliente específico"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Enviar mensaje a todos los clientes"""
        for client_id, websocket in list(self.active_connections.items()):
            try:
                await websocket.send_json(message)
            except:
                self.disconnect(client_id)


# Instancia global
manager = ConnectionManager()


@router.websocket("/analysis/{client_id}")
async def websocket_analysis(
    websocket: WebSocket,
    client_id: str
):
    """WebSocket para análisis en tiempo real"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Recibir mensaje
            data = await websocket.receive_json()
            
            action = data.get("action")
            
            if action == "analyze":
                # Analizar documento
                content = data.get("content")
                tasks = data.get("tasks", [])
                
                await websocket.send_json({
                    "status": "processing",
                    "message": "Análisis iniciado"
                })
                
                # Obtener analizador
                analyzer = get_analyzer()
                
                # Realizar análisis
                from ..core.document_analyzer import AnalysisTask
                analysis_tasks = [AnalysisTask(t) for t in tasks] if tasks else None
                
                result = await analyzer.analyze_document(
                    document_content=content,
                    tasks=analysis_tasks
                )
                
                # Enviar resultado
                await websocket.send_json({
                    "status": "completed",
                    "result": result.__dict__ if hasattr(result, "__dict__") else result
                })
            
            elif action == "ping":
                await websocket.send_json({"status": "pong"})
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        await websocket.send_json({
            "status": "error",
            "message": str(e)
        })


@router.websocket("/notifications/{client_id}")
async def websocket_notifications(
    websocket: WebSocket,
    client_id: str
):
    """WebSocket para notificaciones en tiempo real"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Mantener conexión viva
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(client_id)

