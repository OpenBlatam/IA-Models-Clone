"""
Long term goals routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse

try:
    from services.long_term_goals_service import LongTermGoalsService
except ImportError:
    from ...services.long_term_goals_service import LongTermGoalsService

router = APIRouter()

long_term_goals = LongTermGoalsService()


@router.post("/long-term-goals/create")
async def create_long_term_goal(
    user_id: str = Body(...),
    title: str = Body(...),
    description: str = Body(...),
    target_date: str = Body(...)
):
    """Crea un objetivo a largo plazo"""
    try:
        goal = long_term_goals.create_long_term_goal(
            user_id, title, description, target_date
        )
        return JSONResponse(content=goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando objetivo: {str(e)}")



