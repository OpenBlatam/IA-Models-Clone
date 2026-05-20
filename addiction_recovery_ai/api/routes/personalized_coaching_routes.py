"""
Personalized coaching routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_personalized_coaching_service import AdvancedPersonalizedCoachingService
except ImportError:
    from ...services.advanced_personalized_coaching_service import AdvancedPersonalizedCoachingService

router = APIRouter()

personalized_coaching = AdvancedPersonalizedCoachingService()


@router.post("/coaching/create-plan")
async def create_coaching_plan(
    user_id: str = Body(...),
    user_profile: Dict = Body(...),
    goals: List[str] = Body(...)
):
    """Crea plan de coaching personalizado"""
    try:
        plan = personalized_coaching.create_coaching_plan(
            user_id, user_profile, goals
        )
        return JSONResponse(content=plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando plan: {str(e)}")


@router.post("/coaching/provide-session")
async def provide_coaching_session(
    user_id: str = Body(...),
    session_context: Dict = Body(...)
):
    """Proporciona sesión de coaching"""
    try:
        session = personalized_coaching.provide_coaching_session(
            user_id, session_context
        )
        return JSONResponse(content=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error proporcionando sesión: {str(e)}")



