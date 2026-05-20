"""
Advanced exercise analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_exercise_analysis_service import AdvancedExerciseAnalysisService
except ImportError:
    from ...services.advanced_exercise_analysis_service import AdvancedExerciseAnalysisService

router = APIRouter()

exercise_analysis = AdvancedExerciseAnalysisService()


@router.post("/exercise/analyze-patterns")
async def analyze_exercise_patterns(
    user_id: str = Body(...),
    exercise_data: List[Dict] = Body(...)
):
    """Analiza patrones de ejercicio"""
    try:
        analysis = exercise_analysis.analyze_exercise_patterns(user_id, exercise_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")



