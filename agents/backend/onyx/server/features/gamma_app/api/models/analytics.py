"""
Analytics Models
Analytics and reporting related Pydantic models
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from .base import BaseResponse
from .enums import ContentType

class AnalyticsRequest(BaseModel):
    """Analytics request model"""
    user_id: str
    content_id: Optional[str] = None
    time_period: str = Field("7d", pattern="^(1d|7d|30d|90d|1y)$")
    metrics: List[str] = Field(default_factory=list)

class AnalyticsResponse(BaseResponse):
    """Analytics response model"""
    data: Dict[str, Any]
    time_period: str
    generated_at: datetime

class DashboardData(BaseModel):
    """Dashboard data model"""
    total_content: int
    total_exports: int
    active_collaborations: int
    recent_activity: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    content_types: Dict[str, int]
    export_formats: Dict[str, int]

class ContentPerformance(BaseModel):
    """Content performance model"""
    content_id: str
    views: int
    exports: int
    shares: int
    engagement_score: float
    quality_score: float
    feedback: List[Dict[str, Any]]

class CollaborationStats(BaseModel):
    """Collaboration statistics model"""
    total_sessions: int
    active_sessions: int
    total_participants: int
    average_session_duration: float
    most_active_users: List[Dict[str, Any]]
    session_types: Dict[str, int]







