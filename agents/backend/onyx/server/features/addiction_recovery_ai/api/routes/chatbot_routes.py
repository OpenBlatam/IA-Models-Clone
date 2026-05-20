"""
Chatbot routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.chatbot_service import ChatbotService
except ImportError:
    from ...services.chatbot_service import ChatbotService

router = APIRouter()

chatbot = ChatbotService()


@router.post("/chatbot/message")
async def send_chatbot_message(
    user_id: str = Body(...),
    message: str = Body(...),
    context: Optional[Dict] = Body(None)
):
    """Envía un mensaje al chatbot y recibe respuesta"""
    try:
        response = chatbot.process_message(user_id, message, context)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando mensaje: {str(e)}")


@router.post("/chatbot/start")
async def start_chatbot_conversation(
    user_id: str = Body(...),
    user_data: Optional[Dict] = Body(None)
):
    """Inicia una nueva conversación con el chatbot"""
    try:
        welcome = chatbot.start_conversation(user_id, user_data)
        return JSONResponse(content=welcome)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando conversación: {str(e)}")


@router.get("/chatbot/history/{user_id}")
async def get_chatbot_history(user_id: str, limit: int = Query(20, ge=1, le=100)):
    """Obtiene historial de conversación con el chatbot"""
    try:
        history = chatbot.get_conversation_history(user_id, limit)
        return JSONResponse(content={
            "user_id": user_id,
            "history": history,
            "limit": limit,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo historial: {str(e)}")



