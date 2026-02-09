"""
Advanced progress tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_progress_tracking_service import AdvancedProgressTrackingService
except ImportError:
    from ...services.advanced_progress_tracking_service import AdvancedProgressTrackingService

router = APIRouter()

advanced_progress = AdvancedProgressTrackingService()


@router.post("/progress/visualization")
async def generate_progress_visualization(
    user_id: str = Body(...),
    data: List[Dict] = Body(...),
    visualization_type: str = Body("comprehensive")
):
    """Genera visualización avanzada de progreso"""
    try:
        visualization = advanced_progress.generate_progress_visualization(
            user_id, data, visualization_type
        )
        return JSONResponse(content=visualization)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando visualización: {str(e)}")


@router.post("/progress/comparison")
async def generate_comparison_view(
    user_id: str = Body(...),
    period1_data: List[Dict] = Body(...),
    period2_data: List[Dict] = Body(...)
):
    """Genera vista comparativa entre períodos"""
    try:
        comparison = advanced_progress.generate_comparison_view(
            user_id, period1_data, period2_data
        )
        return JSONResponse(content=comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando comparación: {str(e)}")



