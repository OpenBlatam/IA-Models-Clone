"""
Medication routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse

try:
    from services.medication_service import MedicationService
except ImportError:
    from ...services.medication_service import MedicationService

router = APIRouter()

medication = MedicationService()


@router.post("/medication/add")
async def add_medication(
    user_id: str = Body(...),
    medication_name: str = Body(...),
    dosage: str = Body(...),
    frequency: str = Body(...),
    start_date: str = Body(...)
):
    """Agrega un medicamento al seguimiento"""
    try:
        med = medication.add_medication(user_id, medication_name, dosage, frequency, start_date)
        return JSONResponse(content=med)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando medicamento: {str(e)}")


@router.get("/medication/schedule/{user_id}")
async def get_medication_schedule(user_id: str):
    """Obtiene horario de medicamentos"""
    try:
        schedule = medication.get_medication_schedule(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "schedule": schedule,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo horario: {str(e)}")



