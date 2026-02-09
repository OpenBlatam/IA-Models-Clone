"""
Advanced goal tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_goal_tracking_service import AdvancedGoalTrackingService
except ImportError:
    from ...services.advanced_goal_tracking_service import AdvancedGoalTrackingService

router = APIRouter()

goal_tracking = AdvancedGoalTrackingService()


@router.post("/goals/create-advanced")
async def create_advanced_goal(
    user_id: str = Body(...),
    goal_data: Dict = Body(...)
):
    """Crea objetivo avanzado"""
    try:
        goal = goal_tracking.create_advanced_goal(user_id, goal_data)
        return JSONResponse(content=goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando objetivo: {str(e)}")


@router.post("/goals/analyze-performance")
async def analyze_goal_performance(
    user_id: str = Body(...),
    goals: List[Dict] = Body(...)
):
    """Analiza rendimiento de objetivos"""
    try:
        analysis = goal_tracking.analyze_goal_performance(user_id, goals)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando rendimiento: {str(e)}")



