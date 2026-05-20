"""
Continuous monitoring routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.continuous_monitoring_service import ContinuousMonitoringService
except ImportError:
    from ...services.continuous_monitoring_service import ContinuousMonitoringService

router = APIRouter()

continuous_monitoring = ContinuousMonitoringService()


@router.post("/monitoring/start")
async def start_continuous_monitoring(
    user_id: str = Body(...),
    monitoring_config: Dict = Body(...)
):
    """Inicia monitoreo continuo"""
    try:
        monitoring = continuous_monitoring.start_monitoring(user_id, monitoring_config)
        return JSONResponse(content=monitoring)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando monitoreo: {str(e)}")


@router.post("/monitoring/analyze-data")
async def analyze_monitoring_data(
    user_id: str = Body(...),
    monitoring_data: Dict = Body(...)
):
    """Analiza datos de monitoreo"""
    try:
        analysis = continuous_monitoring.analyze_monitoring_data(user_id, monitoring_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando datos: {str(e)}")



