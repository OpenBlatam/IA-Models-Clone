"""
Pydantic schemas for progress tracking endpoints
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator
from datetime import datetime, date


class LogEntryRequest(BaseModel):
    """Request schema for logging daily entry"""
    user_id: str = Field(..., description="User ID")
    date: str = Field(..., description="Entry date (ISO format)")
    mood: str = Field(..., description="Mood level")
    cravings_level: int = Field(..., ge=0, le=10, description="Cravings level (0-10)")
    triggers_encountered: List[str] = Field(default_factory=list, description="Triggers encountered")
    consumed: bool = Field(default=False, description="Whether substance was consumed")
    notes: Optional[str] = Field(default=None, description="Additional notes")

    @validator('mood')
    def validate_mood(cls, v):
        """Validate mood value"""
        valid_moods = ['excellent', 'good', 'neutral', 'poor', 'terrible']
        if v.lower() not in valid_moods:
            raise ValueError(f"Invalid mood. Must be one of: {valid_moods}")
        return v.lower()


class LogEntryResponse(BaseModel):
    """Response schema for logged entry"""
    entry_id: str = Field(..., description="Entry ID")
    user_id: str = Field(..., description="User ID")
    date: str = Field(..., description="Entry date")
    mood: str = Field(..., description="Mood level")
    cravings_level: int = Field(..., description="Cravings level")
    triggers_encountered: List[str] = Field(default_factory=list, description="Triggers encountered")
    consumed: bool = Field(..., description="Whether substance was consumed")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    logged_at: datetime = Field(default_factory=datetime.now, description="Log timestamp")


class ProgressResponse(BaseModel):
    """Response schema for user progress"""
    user_id: str = Field(..., description="User ID")
    days_sober: int = Field(..., ge=0, description="Days sober")
    total_entries: int = Field(..., ge=0, description="Total log entries")
    streak_days: int = Field(..., ge=0, description="Current streak in days")
    longest_streak: int = Field(..., ge=0, description="Longest streak in days")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    recent_entries: List[Dict] = Field(default_factory=list, description="Recent log entries")


class StatsResponse(BaseModel):
    """Response schema for user statistics"""
    user_id: str = Field(..., description="User ID")
    total_days: int = Field(..., ge=0, description="Total days tracked")
    days_sober: int = Field(..., ge=0, description="Days sober")
    relapse_count: int = Field(..., ge=0, description="Number of relapses")
    average_cravings: float = Field(..., ge=0, le=10, description="Average cravings level")
    most_common_triggers: List[str] = Field(default_factory=list, description="Most common triggers")
    trends: Dict = Field(default_factory=dict, description="Progress trends")


class TimelineResponse(BaseModel):
    """Response schema for progress timeline"""
    user_id: str = Field(..., description="User ID")
    timeline: List[Dict] = Field(default_factory=list, description="Timeline events")
    milestones: List[Dict] = Field(default_factory=list, description="Achieved milestones")
    relapses: List[Dict] = Field(default_factory=list, description="Relapse events")

