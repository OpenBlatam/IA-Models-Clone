"""
Advanced nutrition analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_nutrition_analysis_service import AdvancedNutritionAnalysisService
except ImportError:
    from ...services.advanced_nutrition_analysis_service import AdvancedNutritionAnalysisService

router = APIRouter()

nutrition_analysis = AdvancedNutritionAnalysisService()


@router.post("/nutrition/analyze-patterns")
async def analyze_nutrition_patterns(
    user_id: str = Body(...),
    nutrition_data: List[Dict] = Body(...)
):
    """Analiza patrones de nutrición"""
    try:
        analysis = nutrition_analysis.analyze_nutrition_patterns(user_id, nutrition_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")


@router.post("/nutrition/assess-adequacy")
async def assess_nutritional_adequacy(
    user_id: str = Body(...),
    daily_intake: Dict = Body(...),
    nutritional_requirements: Dict = Body(...)
):
    """Evalúa adecuación nutricional"""
    try:
        assessment = nutrition_analysis.assess_nutritional_adequacy(
            user_id, daily_intake, nutritional_requirements
        )
        return JSONResponse(content=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluando adecuación: {str(e)}")



