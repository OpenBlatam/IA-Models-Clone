"""
Pydantic schemas for gamification endpoints
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class PointsResponse(BaseModel):
    """Response schema for user points"""
    user_id: str = Field(..., description="User ID")
    total_points: int = Field(..., ge=0, description="Total points")
    level: int = Field(..., ge=1, description="Current level")
    points_to_next_level: int = Field(..., ge=0, description="Points needed for next level")
    level_name: str = Field(..., description="Level name")
    breakdown: Dict = Field(default_factory=dict, description="Points breakdown by category")


class AchievementResponse(BaseModel):
    """Response schema for achievement"""
    achievement_id: str = Field(..., description="Achievement ID")
    title: str = Field(..., description="Achievement title")
    description: str = Field(..., description="Achievement description")
    icon: Optional[str] = Field(default=None, description="Achievement icon")
    points: int = Field(default=0, ge=0, description="Points awarded")
    unlocked_at: Optional[datetime] = Field(default=None, description="Unlock timestamp")
    progress: Optional[float] = Field(default=None, ge=0, le=100, description="Progress percentage")


class AchievementsListResponse(BaseModel):
    """Response schema for achievements list"""
    user_id: str = Field(..., description="User ID")
    achievements: List[AchievementResponse] = Field(default_factory=list, description="List of achievements")
    total_achievements: int = Field(default=0, ge=0, description="Total number of achievements")
    unlocked_count: int = Field(default=0, ge=0, description="Number of unlocked achievements")


class LeaderboardEntry(BaseModel):
    """Response schema for leaderboard entry"""
    rank: int = Field(..., ge=1, description="User rank")
    user_id: str = Field(..., description="User ID")
    name: Optional[str] = Field(default=None, description="User name")
    points: int = Field(..., ge=0, description="Total points")
    level: int = Field(..., ge=1, description="Current level")
    days_sober: int = Field(..., ge=0, description="Days sober")


class LeaderboardResponse(BaseModel):
    """Response schema for leaderboard"""
    leaderboard: List[LeaderboardEntry] = Field(default_factory=list, description="Leaderboard entries")
    limit: int = Field(..., ge=1, le=100, description="Number of entries returned")
    total_users: Optional[int] = Field(default=None, ge=0, description="Total number of users")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")

