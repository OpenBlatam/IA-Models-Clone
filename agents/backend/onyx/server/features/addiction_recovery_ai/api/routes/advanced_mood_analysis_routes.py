"""
Advanced mood analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_mood_analysis_service import AdvancedMoodAnalysisService
except ImportError:
    from ...services.advanced_mood_analysis_service import AdvancedMoodAnalysisService

router = APIRouter()

mood_analysis = AdvancedMoodAnalysisService()


@router.post("/mood/analyze-patterns")
async def analyze_mood_patterns(
    user_id: str = Body(...),
    mood_data: List[Dict] = Body(...)
):
    """Analiza patrones de estado de ánimo"""
    try:
        analysis = mood_analysis.analyze_mood_patterns(user_id, mood_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/mood/predict-episode")
async def predict_mood_episode(
    user_id: str = Body(...),
    current_state: Dict = Body(...),
    mood_history: List[Dict] = Body(...)
):
    """Predice episodio de estado de ánimo"""
    try:
        prediction = mood_analysis.predict_mood_episode(
            user_id, current_state, mood_history
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediciendo episodio: {str(e)}")



