"""
Advanced motivation analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_motivation_analysis_service import AdvancedMotivationAnalysisService
except ImportError:
    from ...services.advanced_motivation_analysis_service import AdvancedMotivationAnalysisService

router = APIRouter()

motivation_analysis = AdvancedMotivationAnalysisService()


@router.post("/motivation/assess")
async def assess_motivation(
    user_id: str = Body(...),
    motivation_data: Dict = Body(...)
):
    """Evalúa motivación"""
    try:
        assessment = motivation_analysis.assess_motivation(user_id, motivation_data)
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando motivación: {str(e)}")


@router.post("/motivation/predict-drop")
async def predict_motivation_drop(
    user_id: str = Body(...),
    current_motivation: Dict = Body(...),
    historical_data: List[Dict] = Body(...)
):
    """Predice caída de motivación"""
    try:
        prediction = motivation_analysis.predict_motivation_drop(
            user_id, current_motivation, historical_data
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo caída: {str(e)}")



