"""
Advanced gamification routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse

try:
    from services.advanced_gamification_service import AdvancedGamificationService
except ImportError:
    from ...services.advanced_gamification_service import AdvancedGamificationService

router = APIRouter()

advanced_gamification = AdvancedGamificationService()


@router.post("/gamification/award-achievement")
async def award_achievement(
    user_id: str = Body(...),
    achievement_id: str = Body(...)
):
    """Otorga un logro"""
    try:
        achievement = advanced_gamification.award_achievement(user_id, achievement_id)
        return JSONResponse(content=achievement)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error otorgando logro: {str(e)}")


@router.get("/gamification/user-level/{user_id}")
async def get_user_level(user_id: str, total_points: int = Query(...)):
    """Obtiene nivel del usuario"""
    try:
        level_info = advanced_gamification.calculate_user_level(user_id, total_points)
        return JSONResponse(content=level_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo nivel: {str(e)}")



