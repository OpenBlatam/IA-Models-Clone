"""
Activity pattern analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_activity_pattern_analysis_service import AdvancedActivityPatternAnalysisService
except ImportError:
    from ...services.advanced_activity_pattern_analysis_service import AdvancedActivityPatternAnalysisService

router = APIRouter()

activity_patterns = AdvancedActivityPatternAnalysisService()


@router.post("/activity-patterns/analyze")
async def analyze_activity_patterns(
    user_id: str = Body(...),
    activity_data: List[Dict] = Body(...),
    analysis_type: str = Body("comprehensive")
):
    """Analiza patrones de actividad"""
    try:
        analysis = activity_patterns.analyze_activity_patterns(
            user_id, activity_data, analysis_type
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/activity-patterns/predict-outcome")
async def predict_activity_outcome(
    user_id: str = Body(...),
    current_activities: Dict = Body(...),
    activity_history: List[Dict] = Body(...)
):
    """Predice resultado de actividad"""
    try:
        prediction = activity_patterns.predict_activity_outcome(
            user_id, current_activities, activity_history
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo resultado: {str(e)}")



