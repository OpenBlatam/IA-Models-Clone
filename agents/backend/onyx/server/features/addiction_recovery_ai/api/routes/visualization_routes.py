"""
Visualization routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.visualization_service import VisualizationService
except ImportError:
    from ...services.visualization_service import VisualizationService

router = APIRouter()

visualization = VisualizationService()


@router.post("/visualization/progress-chart")
async def generate_progress_chart(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    chart_type: str = Body("line")
):
    """Genera gráfico de progreso"""
    try:
        chart = visualization.generate_progress_chart(user_id, data, chart_type)
        return JSONResponse(content=chart)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráfico: {str(e)}")


@router.post("/visualization/radar")
async def generate_radar_chart(
    user_id: str = Body(...),
    metrics: Dict = Body(...)
):
    """Genera gráfico de radar"""
    try:
        chart = visualization.generate_radar_chart(user_id, metrics)
        return JSONResponse(content=chart)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráfico: {str(e)}")



