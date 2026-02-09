"""
Referrals endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.referrals import ReferralsService

router = APIRouter()
referrals_service = ReferralsService()


@router.get("/code/{user_id}")
async def get_referral_code(user_id: str) -> Dict[str, Any]:
    """Obtener código de referido"""
    try:
        code = referrals_service.get_referral_code(user_id)
        return {
            "user_id": user_id,
            "referral_code": code,
            "reward_points": referrals_service.reward_points,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register")
async def register_referral(
    referral_code: str,
    referred_user_id: str
) -> Dict[str, Any]:
    """Registrar referido"""
    try:
        referral = referrals_service.register_referral(referral_code, referred_user_id)
        if not referral:
            raise HTTPException(status_code=404, detail="Invalid referral code")
        
        return {
            "success": True,
            "referrer_id": referral.referrer_id,
            "referred_id": referral.referred_id,
            "code": referral.code,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{user_id}")
async def get_referral_stats(user_id: str) -> Dict[str, Any]:
    """Obtener estadísticas de referidos"""
    try:
        stats = referrals_service.get_referral_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




