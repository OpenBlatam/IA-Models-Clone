"""
Goals routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.goals_service import GoalsService
except ImportError:
    from ...services.goals_service import GoalsService

router = APIRouter()

goals = GoalsService()


@router.post("/goals/create")
async def create_goal(
    user_id: str = Body(...),
    goal_type: str = Body(...),
    title: str = Body(...),
    description: str = Body(...),
    target_date: str = Body(...),
    target_value: Optional[float] = Body(None)
):
    """Crea una nueva meta"""
    try:
        goal = goals.create_goal(user_id, goal_type, title, description, target_date, target_value)
        return JSONResponse(content=goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando meta: {str(e)}")


@router.get("/goals/{user_id}")
async def get_user_goals(user_id: str, status: Optional[str] = Query(None)):
    """Obtiene metas del usuario"""
    try:
        user_goals = goals.get_user_goals(user_id, status)
        return JSONResponse(content={
            "user_id": user_id,
            "goals": user_goals,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo metas: {str(e)}")



