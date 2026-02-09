"""
Advanced visual progress routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_visual_progress_service import AdvancedVisualProgressService
except ImportError:
    from ...services.advanced_visual_progress_service import AdvancedVisualProgressService

router = APIRouter()

visual_progress = AdvancedVisualProgressService()


@router.post("/visual-progress/generate-timeline")
async def generate_progress_timeline(
    user_id: str = Body(...),
    progress_data: List[Dict] = Body(...)
):
    """Genera línea de tiempo de progreso"""
    try:
        timeline = visual_progress.generate_progress_timeline(user_id, progress_data)
        return JSONResponse(content=timeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando línea de tiempo: {str(e)}")


@router.post("/visual-progress/create-chart")
async def create_progress_chart(
    user_id: str = Body(...),
    metrics: List[str] = Body(...),
    time_period: str = Body("30_days")
):
    """Crea gráfico de progreso"""
    try:
        chart = visual_progress.create_progress_chart(user_id, metrics, time_period)
        return JSONResponse(content=chart)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando gráfico: {str(e)}")



