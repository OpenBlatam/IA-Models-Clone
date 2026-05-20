"""
WebSocket Endpoints
===================
Endpoints para comunicación WebSocket en tiempo real.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List
import json

from ...services.coaching_service import DogTrainingCoach
from ...core.exceptions import ServiceException, OpenRouterException
from ...utils.logger import get_logger

router = APIRouter(prefix="/api/v1/ws", tags=["websocket"])
logger = get_logger(__name__)


class ConnectionManager:
    """Manager para conexiones WebSocket."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Conectar cliente."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Desconectar cliente."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Enviar mensaje personal."""
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast a todos los clientes."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass


manager = ConnectionManager()


@router.websocket("/coach")
async def websocket_coach(websocket: WebSocket):
    """
    WebSocket endpoint para coaching en tiempo real.
    """
    await manager.connect(websocket)
    service = DogTrainingCoach()
    
    try:
        while True:
            # Recibir mensaje del cliente
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "coaching_request":
                question = message.get("question", "")
                dog_breed = message.get("dog_breed")
                dog_age = message.get("dog_age")
                
                # Enviar respuesta de coaching
                try:
                    result = await service.get_coaching_advice(
                        question=question,
                        dog_breed=dog_breed,
                        dog_age=dog_age
                    )
                    
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "coaching_response",
                            "data": result
                        }),
                        websocket
                    )
                except Exception as e:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "error": str(e)
                        }),
                        websocket
                    )
            
            elif message.get("type") == "ping":
                # Responder ping
                await manager.send_personal_message(
                    json.dumps({"type": "pong"}),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint para chat en tiempo real.
    """
    await manager.connect(websocket)
    service = DogTrainingCoach()
    conversation_history = []
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "chat_message":
                user_message = message.get("message", "")
                
                try:
                    result = await service.chat(
                        message=user_message,
                        conversation_history=conversation_history
                    )
                    
                    # Agregar a historial
                    conversation_history.append({"role": "user", "content": user_message})
                    conversation_history.append({"role": "assistant", "content": result.get("response", "")})
                    
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "chat_response",
                            "data": result
                        }),
                        websocket
                    )
                except Exception as e:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "error": str(e)
                        }),
                        websocket
                    )
            
            elif message.get("type") == "clear_history":
                conversation_history = []
                await manager.send_personal_message(
                    json.dumps({"type": "history_cleared"}),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

