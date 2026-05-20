"""
Advanced medication tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_medication_tracking_service import AdvancedMedicationTrackingService
except ImportError:
    from ...services.advanced_medication_tracking_service import AdvancedMedicationTrackingService

router = APIRouter()

medication_tracking = AdvancedMedicationTrackingService()


@router.post("/medications/register")
async def register_medication(
    user_id: str = Body(...),
    medication_data: Dict = Body(...)
):
    """Registra medicamento"""
    try:
        medication = medication_tracking.register_medication(user_id, medication_data)
        return JSONResponse(content=medication)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando medicamento: {str(e)}")


@router.post("/medications/analyze-adherence")
async def analyze_medication_adherence(
    user_id: str = Body(...),
    medication_id: str = Body(...),
    doses: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza adherencia a medicamentos"""
    try:
        analysis = medication_tracking.analyze_medication_adherence(
            user_id, medication_id, doses, days
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando adherencia: {str(e)}")



