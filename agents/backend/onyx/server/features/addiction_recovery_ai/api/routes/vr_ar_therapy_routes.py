"""
VR/AR therapy routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.vr_ar_therapy_service import VRARTherapyService
except ImportError:
    from ...services.vr_ar_therapy_service import VRARTherapyService

router = APIRouter()

vr_ar_therapy = VRARTherapyService()


@router.post("/vr-ar/create-session")
async def create_vr_therapy_session(
    user_id: str = Body(...),
    therapy_type: str = Body(...),
    duration_minutes: int = Body(30)
):
    """Crea sesión de terapia VR/AR"""
    try:
        session = vr_ar_therapy.create_therapy_session(
            user_id, therapy_type, None, duration_minutes
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando sesión: {str(e)}")


@router.get("/vr-ar/scenarios")
async def get_available_scenarios(
    therapy_type: Optional[str] = Query(None)
):
    """Obtiene escenarios VR/AR disponibles"""
    try:
        scenarios = vr_ar_therapy.get_available_scenarios(therapy_type)
        return JSONResponse(content={
            "scenarios": scenarios,
            "total": len(scenarios),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo escenarios: {str(e)}")



