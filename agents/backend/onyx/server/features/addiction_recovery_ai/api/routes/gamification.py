"""
Gamification routes - Refactored with best practices
"""

from fastapi import APIRouter, HTTPException, Query, status

try:
    from schemas.gamification import (
        PointsResponse,
        AchievementsListResponse,
        LeaderboardResponse
    )
    from schemas.common import ErrorResponse
    from dependencies import (
        ProgressTrackerDep,
        GamificationServiceDep
    )
except ImportError:
    from ...schemas.gamification import (
        PointsResponse,
        AchievementsListResponse,
        LeaderboardResponse
    )
    from ...schemas.common import ErrorResponse
    from ...dependencies import (
        ProgressTrackerDep,
        GamificationServiceDep
    )

router = APIRouter(prefix="/gamification", tags=["Gamification"])


@router.get(
    "/points/{user_id}",
    response_model=PointsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_user_points(
    user_id: str,
    tracker: ProgressTrackerDep,
    gamification: GamificationServiceDep
) -> PointsResponse:
    """
    Obtiene puntos y nivel del usuario
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        progress = tracker.get_progress(user_id, None, [])
        
        points = gamification.calculate_points(
            days_sober=progress.get("days_sober", 0),
            entries_count=0,
            milestones_achieved=0,
            coaching_sessions=0
        )
        
        return PointsResponse(
            user_id=user_id,
            total_points=points.get("total_points", 0),
            level=points.get("level", 1),
            points_to_next_level=points.get("points_to_next_level", 100),
            level_name=points.get("level_name", "Beginner"),
            breakdown=points.get("breakdown", {})
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo puntos: {str(e)}"
        )


@router.get(
    "/achievements/{user_id}",
    response_model=AchievementsListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_user_achievements(
    user_id: str,
    tracker: ProgressTrackerDep,
    gamification: GamificationServiceDep
) -> AchievementsListResponse:
    """
    Obtiene logros del usuario
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        progress = tracker.get_progress(user_id, None, [])
        
        achievements_data = gamification.check_achievements(
            user_id=user_id,
            days_sober=progress.get("days_sober", 0),
            current_streak=progress.get("streak_days", 0),
            entries_count=0
        )
        
        from schemas.gamification import AchievementResponse
        achievements = [
            AchievementResponse(
                achievement_id=a.get("achievement_id", ""),
                title=a.get("title", ""),
                description=a.get("description", ""),
                icon=a.get("icon"),
                points=a.get("points", 0),
                unlocked_at=a.get("unlocked_at"),
                progress=a.get("progress")
            )
            for a in achievements_data
        ]
        
        unlocked_count = len([a for a in achievements if a.unlocked_at is not None])
        
        return AchievementsListResponse(
            user_id=user_id,
            achievements=achievements,
            total_achievements=len(achievements),
            unlocked_count=unlocked_count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo logros: {str(e)}"
        )


@router.get(
    "/leaderboard",
    response_model=LeaderboardResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_leaderboard(
    limit: int = Query(10, ge=1, le=100, description="Number of entries to return"),
    gamification: GamificationServiceDep
) -> LeaderboardResponse:
    """
    Obtiene tabla de clasificación
    
    - **limit**: Número de entradas a retornar (1-100)
    """
    # Guard clause: Validate limit
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit debe estar entre 1 y 100"
        )
    
    try:
        # In real implementation, get data from multiple users
        users_data = []
        leaderboard_data = gamification.get_leaderboard(users_data, limit)
        
        from schemas.gamification import LeaderboardEntry
        entries = [
            LeaderboardEntry(
                rank=entry.get("rank", idx + 1),
                user_id=entry.get("user_id", ""),
                name=entry.get("name"),
                points=entry.get("points", 0),
                level=entry.get("level", 1),
                days_sober=entry.get("days_sober", 0)
            )
            for idx, entry in enumerate(leaderboard_data)
        ]
        
        return LeaderboardResponse(
            leaderboard=entries,
            limit=limit,
            total_users=len(entries)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo leaderboard: {str(e)}"
        )

