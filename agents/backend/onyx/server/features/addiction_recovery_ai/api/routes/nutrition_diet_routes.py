"""
Nutrition and diet analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.nutrition_diet_analysis_service import NutritionDietAnalysisService
except ImportError:
    from ...services.nutrition_diet_analysis_service import NutritionDietAnalysisService

router = APIRouter()

nutrition_diet = NutritionDietAnalysisService()


@router.post("/nutrition/record-meal")
async def record_meal(
    user_id: str = Body(...),
    meal_data: Dict = Body(...)
):
    """Registra una comida"""
    try:
        meal = nutrition_diet.record_meal(user_id, meal_data)
        return JSONResponse(content=meal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando comida: {str(e)}")


@router.post("/nutrition/analyze-patterns")
async def analyze_nutrition_patterns(
    user_id: str = Body(...),
    meals: List[Dict] = Body(...),
    days: int = Body(30)
):
    """Analiza patrones nutricionales"""
    try:
        analysis = nutrition_diet.analyze_nutrition_patterns(user_id, meals, days)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando patrones: {str(e)}")



