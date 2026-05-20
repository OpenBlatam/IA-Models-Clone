"""
Withdrawal tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.withdrawal_tracking_service import WithdrawalTrackingService
except ImportError:
    from ...services.withdrawal_tracking_service import WithdrawalTrackingService

router = APIRouter()

withdrawal_tracking = WithdrawalTrackingService()


@router.post("/withdrawal/record-symptom")
async def record_withdrawal_symptom(
    user_id: str = Body(...),
    symptom_name: str = Body(...),
    severity: str = Body(...),
    notes: Optional[str] = Body(None)
):
    """Registra un síntoma de abstinencia"""
    try:
        symptom = withdrawal_tracking.record_symptom(user_id, symptom_name, severity, notes)
        return JSONResponse(content=symptom)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando síntoma: {str(e)}")


@router.get("/withdrawal/timeline/{user_id}")
async def get_withdrawal_timeline(
    user_id: str,
    addiction_type: str = Query(...),
    days_sober: int = Query(...)
):
    """Obtiene línea de tiempo de síntomas de abstinencia"""
    try:
        timeline = withdrawal_tracking.get_withdrawal_timeline(user_id, addiction_type, days_sober)
        return JSONResponse(content=timeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo timeline: {str(e)}")



