"""
Advanced rewards routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.advanced_rewards_service import AdvancedRewardsService
except ImportError:
    from ...services.advanced_rewards_service import AdvancedRewardsService

router = APIRouter()

rewards_service = AdvancedRewardsService()


@router.post("/rewards/award")
async def award_reward(
    user_id: str = Body(...),
    reward_id: str = Body(...),
    achievement_data: Dict = Body(...)
):
    """Otorga recompensa"""
    try:
        reward = rewards_service.award_reward(user_id, reward_id, achievement_data)
        return JSONResponse(content=reward)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error otorgando recompensa: {str(e)}")


@router.post("/rewards/analyze-impact")
async def analyze_reward_impact(
    user_id: str = Body(...),
    rewards: List[Dict] = Body(...),
    recovery_data: List[Dict] = Body(...)
):
    """Analiza impacto de recompensas"""
    try:
        analysis = rewards_service.analyze_reward_impact(
            user_id, rewards, recovery_data
        )
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando impacto: {str(e)}")



