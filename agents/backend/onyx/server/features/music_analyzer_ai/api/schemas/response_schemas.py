"""
Response schemas for consistent API responses
"""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool = True
    data: Optional[Any] = None
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel):
    """Paginated response schema"""
    items: List[Any]
    pagination: Dict[str, Any] = Field(
        ...,
        description="Pagination information"
    )


class TrackResponse(BaseModel):
    """Track response schema"""
    id: str
    name: str
    artists: List[str]
    album: Optional[str] = None
    duration_ms: Optional[int] = None
    preview_url: Optional[str] = None
    external_urls: Optional[Dict[str, str]] = None
    popularity: int = 0


class TracksListResponse(BaseModel):
    """Tracks list response schema"""
    tracks: List[TrackResponse]
    total: int


class AnalysisResponse(BaseModel):
    """Analysis response schema"""
    track_id: str
    analysis: Dict[str, Any]
    timestamp: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Recommendation response schema"""
    recommendations: List[TrackResponse]
    total: int
    method: Optional[str] = None

