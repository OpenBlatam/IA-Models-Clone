"""
Financial tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse

try:
    from services.financial_tracking_service import FinancialTrackingService
except ImportError:
    from ...services.financial_tracking_service import FinancialTrackingService

router = APIRouter()

financial_tracking = FinancialTrackingService()


@router.post("/financial/calculate-savings")
async def calculate_savings(
    user_id: str = Body(...),
    addiction_type: str = Body(...),
    days_sober: int = Body(...),
    daily_cost: float = Body(...)
):
    """Calcula ahorros por días de sobriedad"""
    try:
        savings = financial_tracking.calculate_savings(user_id, addiction_type, days_sober, daily_cost)
        return JSONResponse(content=savings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando ahorros: {str(e)}")


@router.get("/financial/summary/{user_id}")
async def get_financial_summary(user_id: str, days: int = Query(30)):
    """Obtiene resumen financiero"""
    try:
        summary = financial_tracking.get_financial_summary(user_id, days)
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo resumen: {str(e)}")



