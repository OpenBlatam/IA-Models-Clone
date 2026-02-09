"""
Symptom tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_symptom_tracking_service import AdvancedSymptomTrackingService
except ImportError:
    from ...services.advanced_symptom_tracking_service import AdvancedSymptomTrackingService

router = APIRouter()

symptom_tracking = AdvancedSymptomTrackingService()


@router.post("/symptoms/record")
async def record_symptom(
    user_id: str = Body(...),
    symptom_data: Dict = Body(...)
):
    """Registra síntoma"""
    try:
        symptom = symptom_tracking.record_symptom(user_id, symptom_data)
        return JSONResponse(content=symptom)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando síntoma: {str(e)}")


@router.post("/symptoms/analyze-patterns")
async def analyze_symptom_patterns(
    user_id: str = Body(...),
    symptoms: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones de síntomas"""
    try:
        analysis = symptom_tracking.analyze_symptom_patterns(user_id, symptoms, days)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")



