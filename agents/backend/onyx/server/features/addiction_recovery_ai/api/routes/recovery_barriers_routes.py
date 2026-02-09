"""
Recovery barriers analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.recovery_barriers_analysis_service import RecoveryBarriersAnalysisService
except ImportError:
    from ...services.recovery_barriers_analysis_service import RecoveryBarriersAnalysisService

router = APIRouter()

barriers_analysis = RecoveryBarriersAnalysisService()


@router.post("/barriers/identify")
async def identify_barriers(
    user_id: str = Body(...),
    user_data: Dict = Body(...)
):
    """Identifica barreras de recuperación"""
    try:
        barriers = barriers_analysis.identify_barriers(user_id, user_data)
        return JSONResponse(content=barriers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error identificando barreras: {str(e)}")


@router.post("/barriers/suggest-solutions")
async def suggest_barrier_solutions(
    user_id: str = Body(...),
    barrier: Dict = Body(...)
):
    """Sugiere soluciones para barrera"""
    try:
        solutions = barriers_analysis.suggest_barrier_solutions(user_id, barrier)
        return JSONResponse(content=solutions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sugiriendo soluciones: {str(e)}")



