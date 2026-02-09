"""
Gamification routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.gamification_service import GamificationService
    from core.progress_tracker import ProgressTracker
except ImportError:
    from ...services.gamification_service import GamificationService
    from ...core.progress_tracker import ProgressTracker

router = APIRouter()

gamification = GamificationService()
tracker = ProgressTracker()


@router.get("/gamification/points/{user_id}")
async def get_user_points(user_id: str):
    """Obtiene puntos y nivel del usuario"""
    try:
        progress = tracker.get_progress(user_id, None, [])
        points = gamification.calculate_points(
            days_sober=progress.get("days_sober", 0),
            entries_count=0,
            milestones_achieved=0,
            coaching_sessions=0
        )
        return JSONResponse(content=points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo puntos: {str(e)}")


@router.get("/gamification/achievements/{user_id}")
async def get_user_achievements(user_id: str):
    """Obtiene logros del usuario"""
    try:
        progress = tracker.get_progress(user_id, None, [])
        achievements = gamification.check_achievements(
            user_id=user_id,
            days_sober=progress.get("days_sober", 0),
            current_streak=progress.get("streak_days", 0),
            entries_count=0
        )
        return JSONResponse(content={
            "user_id": user_id,
            "achievements": achievements,
            "total_achievements": len(achievements),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo logros: {str(e)}")


@router.get("/gamification/leaderboard")
async def get_leaderboard(limit: int = Query(10, ge=1, le=100)):
    """Obtiene tabla de clasificación"""
    try:
        users_data = []
        leaderboard = gamification.get_leaderboard(users_data, limit)
        return JSONResponse(content={
            "leaderboard": leaderboard,
            "limit": limit,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo leaderboard: {str(e)}")



