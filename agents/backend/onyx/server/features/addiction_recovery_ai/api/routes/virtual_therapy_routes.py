"""
Virtual therapy routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional

try:
    from services.virtual_therapy_service import VirtualTherapyService
except ImportError:
    from ...services.virtual_therapy_service import VirtualTherapyService

router = APIRouter()

virtual_therapy = VirtualTherapyService()


@router.post("/therapy/schedule")
async def schedule_therapy(
    user_id: str = Body(...),
    therapy_type: str = Body(...),
    scheduled_time: str = Body(...),
    duration_minutes: int = Body(60)
):
    """Programa una sesión de terapia"""
    try:
        scheduled = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
        session = virtual_therapy.schedule_therapy_session(
            user_id, therapy_type, None, scheduled, duration_minutes
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error programando sesión: {str(e)}")


@router.get("/therapy/therapists")
async def get_available_therapists(
    therapy_type: Optional[str] = Query(None),
    specialization: Optional[str] = Query(None)
):
    """Obtiene terapeutas disponibles"""
    try:
        therapists = virtual_therapy.get_available_therapists(therapy_type, specialization)
        return JSONResponse(content={
            "therapists": therapists,
            "total": len(therapists),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo terapeutas: {str(e)}")



