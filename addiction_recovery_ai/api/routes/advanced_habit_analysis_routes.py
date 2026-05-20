"""
Advanced habit analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_habit_analysis_service import AdvancedHabitAnalysisService
except ImportError:
    from ...services.advanced_habit_analysis_service import AdvancedHabitAnalysisService

router = APIRouter()

habit_analysis = AdvancedHabitAnalysisService()


@router.post("/habits/analyze")
async def analyze_habits(
    user_id: str = Body(...),
    habits: List[Dict] = Body(...)
):
    """Analiza hábitos"""
    try:
        analysis = habit_analysis.analyze_habits(user_id, habits)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando hábitos: {str(e)}")



