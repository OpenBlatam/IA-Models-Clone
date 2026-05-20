"""
Gamification endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.gamification import GamificationService, BadgeType
from models.schemas import UserProfile

router = APIRouter()
gamification_service = GamificationService()


@router.get("/progress/{user_id}")
async def get_progress(user_id: str) -> Dict[str, Any]:
    """Obtener progreso completo del usuario"""
    try:
        progress = gamification_service.get_user_progress(user_id)
        return progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/points/{user_id}")
async def add_points(
    user_id: str,
    action: str,
    amount: int = None
) -> Dict[str, Any]:
    """Agregar puntos al usuario por una acción"""
    try:
        result = gamification_service.add_points(user_id, action, amount)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard")
async def get_leaderboard(limit: int = 10) -> Dict[str, Any]:
    """Obtener leaderboard de usuarios"""
    try:
        leaderboard = gamification_service.get_leaderboard(limit)
        return {
            "leaderboard": leaderboard,
            "total_users": len(leaderboard),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/badges/{user_id}")
async def get_badges(user_id: str) -> Dict[str, Any]:
    """Obtener badges del usuario"""
    try:
        progress = gamification_service.get_user_progress(user_id)
        return {
            "badges": progress.get("badges", []),
            "total_badges": len(progress.get("badges", [])),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




