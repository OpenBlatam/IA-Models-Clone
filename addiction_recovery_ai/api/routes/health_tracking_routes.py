"""
Health tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.health_tracking_service import HealthTrackingService
except ImportError:
    from ...services.health_tracking_service import HealthTrackingService

router = APIRouter()

health_tracking = HealthTrackingService()


@router.post("/health/metric")
async def record_health_metric(
    user_id: str = Body(...),
    metric_type: str = Body(...),
    value: float = Body(...),
    unit: str = Body(...),
    notes: Optional[str] = Body(None)
):
    """Registra una métrica de salud"""
    try:
        metric = health_tracking.record_health_metric(user_id, metric_type, value, unit, notes)
        return JSONResponse(content=metric)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando métrica: {str(e)}")


@router.get("/health/summary/{user_id}")
async def get_health_summary(
    user_id: str,
    days_sober: int = Query(...),
    addiction_type: str = Query(...)
):
    """Obtiene resumen de salud del usuario"""
    try:
        summary = health_tracking.get_health_summary(user_id, days_sober, addiction_type)
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo resumen: {str(e)}")



