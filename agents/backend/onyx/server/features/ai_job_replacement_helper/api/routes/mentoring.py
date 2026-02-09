"""
Mentoring endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.mentoring import MentoringService, SessionType, MentorType

router = APIRouter()
mentoring_service = MentoringService()


@router.post("/start/{user_id}")
async def start_session(
    user_id: str,
    session_type: str,
    mentor_type: Optional[str] = None
) -> Dict[str, Any]:
    """Iniciar sesión de coaching"""
    try:
        session_type_enum = SessionType(session_type)
        mentor_type_enum = MentorType(mentor_type) if mentor_type else None
        
        session = mentoring_service.start_session(
            user_id, session_type_enum, mentor_type_enum
        )
        
        return {
            "session_id": session.id,
            "session_type": session.session_type.value,
            "mentor_type": session.mentor_type.value,
            "started_at": session.started_at.isoformat(),
            "initial_message": {
                "message": session.messages[0].message if session.messages else "",
                "suggestions": session.messages[0].suggestions if session.messages else [],
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/{user_id}/{session_id}")
async def ask_mentor(
    user_id: str,
    session_id: str,
    question: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Hacer pregunta al mentor"""
    try:
        response = mentoring_service.ask_mentor(user_id, session_id, question, context)
        return {
            "message": response.message,
            "suggestions": response.suggestions,
            "resources": response.resources,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/career-advice/{user_id}")
async def get_career_advice(
    user_id: str,
    current_situation: str,
    goals: str
) -> Dict[str, Any]:
    """Obtener consejo de carrera"""
    try:
        goals_list = goals.split(",") if goals else []
        advice = mentoring_service.get_career_advice(user_id, current_situation, goals_list)
        return {
            "message": advice.message,
            "suggestions": advice.suggestions,
            "resources": advice.resources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interview-tips/{user_id}")
async def get_interview_tips(
    user_id: str,
    job_title: str,
    company: str
) -> Dict[str, Any]:
    """Obtener tips para entrevista"""
    try:
        tips = mentoring_service.get_interview_tips(user_id, job_title, company)
        return {
            "message": tips.message,
            "suggestions": tips.suggestions,
            "resources": tips.resources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/motivation/{user_id}")
async def get_motivation(
    user_id: str,
    current_mood: Optional[str] = None
) -> Dict[str, Any]:
    """Obtener mensaje motivacional"""
    try:
        message = mentoring_service.get_motivational_message(user_id, current_mood)
        return {
            "message": message.message,
            "suggestions": message.suggestions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




