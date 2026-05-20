"""
Advanced achievements routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.advanced_achievements_service import AdvancedAchievementsService
except ImportError:
    from ...services.advanced_achievements_service import AdvancedAchievementsService

router = APIRouter()

achievements = AdvancedAchievementsService()


@router.post("/achievements/award")
async def award_achievement(
    user_id: str = Body(...),
    achievement_data: Dict = Body(...)
):
    """Otorga logro"""
    try:
        achievement = achievements.award_achievement(user_id, achievement_data)
        return JSONResponse(content=achievement)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error otorgando logro: {str(e)}")


@router.post("/achievements/check-eligibility")
async def check_achievement_eligibility(
    user_id: str = Body(...),
    achievement_criteria: Dict = Body(...),
    user_data: Dict = Body(...)
):
    """Verifica elegibilidad para logro"""
    try:
        eligibility = achievements.check_achievement_eligibility(
            user_id, achievement_criteria, user_data
        )
        return JSONResponse(content=eligibility)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verificando elegibilidad: {str(e)}")



