"""
Gamification domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.gamification import (
        PointsResponse,
        AchievementResponse,
        AchievementsListResponse,
        LeaderboardEntry,
        LeaderboardResponse
    )
    
    def register_schemas():
        register_schema("gamification", "PointsResponse", PointsResponse)
        register_schema("gamification", "AchievementResponse", AchievementResponse)
        register_schema("gamification", "AchievementsListResponse", AchievementsListResponse)
        register_schema("gamification", "LeaderboardEntry", LeaderboardEntry)
        register_schema("gamification", "LeaderboardResponse", LeaderboardResponse)
except ImportError:
    pass



