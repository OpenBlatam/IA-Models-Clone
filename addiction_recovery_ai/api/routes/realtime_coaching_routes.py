"""
Realtime coaching routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse

try:
    from services.realtime_coaching_service import RealtimeCoachingService
except ImportError:
    from ...services.realtime_coaching_service import RealtimeCoachingService

router = APIRouter()

realtime_coaching = RealtimeCoachingService()


@router.post("/coaching/start-session")
async def start_coaching_session(
    user_id: str = Body(...),
    session_type: str = Body("general")
):
    """Inicia sesión de coaching"""
    try:
        session = realtime_coaching.start_coaching_session(user_id, session_type)
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando sesión: {str(e)}")


@router.post("/coaching/send-message")
async def send_coaching_message(
    session_id: str = Body(...),
    user_id: str = Body(...),
    message: str = Body(...)
):
    """Envía mensaje en sesión de coaching"""
    try:
        response = realtime_coaching.send_coaching_message(session_id, user_id, message)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enviando mensaje: {str(e)}")



