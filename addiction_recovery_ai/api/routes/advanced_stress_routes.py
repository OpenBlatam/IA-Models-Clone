"""
Advanced stress analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_stress_analysis_service import AdvancedStressAnalysisService
except ImportError:
    from ...services.advanced_stress_analysis_service import AdvancedStressAnalysisService

router = APIRouter()

stress_analysis = AdvancedStressAnalysisService()


@router.post("/stress/assess")
async def assess_stress(
    user_id: str = Body(...),
    stress_data: Dict = Body(...)
):
    """Evalúa estrés"""
    try:
        assessment = stress_analysis.assess_stress(user_id, stress_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando estrés: {str(e)}")


@router.post("/stress/predict-episode")
async def predict_stress_episode(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    stress_history: List[Dict] = Body(...)
):
    """Predice episodio de estrés"""
    try:
        prediction = stress_analysis.predict_stress_episode(
            user_id, current_state, stress_history
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo episodio: {str(e)}")



