"""
Pydantic schemas for support and motivation endpoints
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class CoachingSessionRequest(BaseModel):
    """Request schema for coaching session"""
    user_id: str = Field(..., description="User ID")
    topic: str = Field(..., description="Session topic")
    current_situation: str = Field(..., description="Current situation description")
    questions: Optional[List[str]] = Field(default=None, description="Specific questions")


class CoachingSessionResponse(BaseModel):
    """Response schema for coaching session"""
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    topic: str = Field(..., description="Session topic")
    guidance: str = Field(..., description="Guidance provided")
    questions_to_consider: List[str] = Field(default_factory=list, description="Questions to consider")
    action_items: List[Dict] = Field(default_factory=list, description="Action items")
    encouragement: str = Field(..., description="Encouragement message")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation timestamp")


class MotivationResponse(BaseModel):
    """Response schema for motivation messages"""
    user_id: str = Field(..., description="User ID")
    messages: List[str] = Field(default_factory=list, description="Motivational messages")
    personalized_message: Optional[str] = Field(default=None, description="Personalized message")
    achievements_summary: Optional[Dict] = Field(default=None, description="Achievements summary")


class MilestoneRequest(BaseModel):
    """Request schema for celebrating milestone"""
    user_id: str = Field(..., description="User ID")
    milestone_days: int = Field(..., ge=1, description="Milestone days to celebrate")


class MilestoneResponse(BaseModel):
    """Response schema for milestone celebration"""
    user_id: str = Field(..., description="User ID")
    milestone_days: int = Field(..., description="Milestone days")
    celebration_message: str = Field(..., description="Celebration message")
    rewards: List[Dict] = Field(default_factory=list, description="Rewards earned")
    next_milestone: Optional[Dict] = Field(default=None, description="Next milestone information")
    celebrated_at: datetime = Field(default_factory=datetime.now, description="Celebration timestamp")


class AchievementsResponse(BaseModel):
    """Response schema for user achievements"""
    user_id: str = Field(..., description="User ID")
    achievements: List[Dict] = Field(default_factory=list, description="List of achievements")
    total_points: int = Field(default=0, ge=0, description="Total achievement points")
    level: Optional[str] = Field(default=None, description="Current achievement level")

