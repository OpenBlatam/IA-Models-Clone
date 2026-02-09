"""
Environment context analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.environment_context_analysis_service import EnvironmentContextAnalysisService
except ImportError:
    from ...services.environment_context_analysis_service import EnvironmentContextAnalysisService

router = APIRouter()

environment_context = EnvironmentContextAnalysisService()


@router.post("/environment/record-context")
async def record_environment_context(
    user_id: str = Body(...),
    context_data: Dict = Body(...)
):
    """Registra contexto de entorno"""
    try:
        context = environment_context.record_environment_context(user_id, context_data)
        return JSONResponse(content=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando contexto: {str(e)}")


@router.post("/environment/predict-risk")
async def predict_environment_risk(
    user_id: str = Body(...),
    current_context: Dict = Body(...)
):
    """Predice riesgo de entorno"""
    try:
        prediction = environment_context.predict_environment_risk(user_id, current_context)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo riesgo: {str(e)}")



