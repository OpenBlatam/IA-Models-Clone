"""
Advanced medication routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.advanced_medication_service import AdvancedMedicationService
except ImportError:
    from ...services.advanced_medication_service import AdvancedMedicationService

router = APIRouter()

advanced_medication = AdvancedMedicationService()


@router.post("/medication/advanced/register")
async def register_medication_advanced(
    user_id: str = Body(...),
    medication_data: Dict = Body(...)
):
    """Registra medicamento avanzado"""
    try:
        medication = advanced_medication.register_medication_advanced(
            user_id, medication_data
        )
        return JSONResponse(content=medication)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando medicamento: {str(e)}")



