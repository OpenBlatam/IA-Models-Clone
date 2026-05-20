"""
Real-Time Mentoring endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.real_time_mentoring import RealTimeMentoringService, MentorType

router = APIRouter()
mentoring_service = RealTimeMentoringService()


@router.post("/start/{user_id}")
async def start_session(
    user_id: str,
    mentor_type: str,
    initial_question: Optional[str] = None
) -> Dict[str, Any]:
    """Iniciar sesión de mentoría"""
    try:
        mentor_type_enum = MentorType(mentor_type)
        session = mentoring_service.start_session(user_id, mentor_type_enum, initial_question)
        return {
            "session_id": session.id,
            "mentor_type": session.mentor_type.value,
            "status": session.status.value,
            "messages": session.messages,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/{session_id}")
async def send_message(
    session_id: str,
    message: str
) -> Dict[str, Any]:
    """Enviar mensaje en sesión activa"""
    try:
        response = mentoring_service.send_message(session_id, message)
        return {
            "message": response.message,
            "suggestions": response.suggestions,
            "resources": response.resources,
            "next_steps": response.next_steps,
            "confidence": response.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/end/{session_id}")
async def end_session(session_id: str) -> Dict[str, Any]:
    """Finalizar sesión"""
    try:
        result = mentoring_service.end_session(session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active/{user_id}")
async def get_active_session(user_id: str) -> Dict[str, Any]:
    """Obtener sesión activa del usuario"""
    try:
        session = mentoring_service.get_active_session(user_id)
        if not session:
            return {"active": False}
        
        return {
            "active": True,
            "session_id": session.id,
            "mentor_type": session.mentor_type.value,
            "duration_minutes": session.duration_minutes,
            "messages_count": len(session.messages),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}")
async def get_session_history(user_id: str) -> Dict[str, Any]:
    """Obtener historial de sesiones"""
    try:
        history = mentoring_service.get_session_history(user_id)
        return {
            "user_id": user_id,
            "sessions": history,
            "total": len(history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




