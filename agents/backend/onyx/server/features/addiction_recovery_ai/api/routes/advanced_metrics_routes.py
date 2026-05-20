"""
Advanced metrics routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_metrics_service import AdvancedMetricsService
except ImportError:
    from ...services.advanced_metrics_service import AdvancedMetricsService

router = APIRouter()

advanced_metrics = AdvancedMetricsService()


@router.post("/metrics/calculate-comprehensive")
async def calculate_comprehensive_metrics(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    metrics_type: str = Body("all")
):
    """Calcula métricas comprensivas"""
    try:
        metrics = advanced_metrics.calculate_comprehensive_metrics(
            user_id, data, metrics_type
        )
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando métricas: {str(e)}")



