"""
Mentorship routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict

try:
    from services.mentorship_service import MentorshipService
except ImportError:
    from ...services.mentorship_service import MentorshipService

router = APIRouter()

mentorship = MentorshipService()


@router.post("/mentorship/request")
async def create_mentorship_request(
    mentee_id: str = Body(...),
    preferences: Dict = Body(...),
    goals: List[str] = Body(...)
):
    """Crea una solicitud de mentoría"""
    try:
        request = mentorship.create_mentorship_request(mentee_id, preferences, goals)
        return JSONResponse(content=request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando solicitud: {str(e)}")


@router.get("/mentorship/available")
async def get_available_mentors(
    addiction_type: Optional[str] = Query(None),
    experience_min: Optional[int] = Query(None)
):
    """Obtiene mentores disponibles"""
    try:
        mentors = mentorship.get_available_mentors(addiction_type, experience_min)
        return JSONResponse(content={
            "mentors": mentors,
            "total": len(mentors),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo mentores: {str(e)}")



