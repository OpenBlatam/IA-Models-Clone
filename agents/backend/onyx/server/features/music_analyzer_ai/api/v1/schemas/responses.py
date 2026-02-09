"""
Response Schemas

Pydantic models for response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = False
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    code: Optional[str] = Field(None, description="Error code")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Track not found",
                "detail": "Track with ID 123 not found",
                "code": "TRACK_NOT_FOUND"
            }
        }


class TrackInfo(BaseModel):
    """Track information"""
    track_id: str
    track_name: str
    artists: List[str]
    album: Optional[str] = None
    duration_seconds: Optional[float] = None


class AnalysisResponse(BaseModel):
    """Response schema for track analysis"""
    success: bool = True
    track_id: str
    track_name: str
    artists: List[str]
    album: Optional[str] = None
    duration_seconds: Optional[float] = None
    analysis: Dict[str, Any] = Field(..., description="Complete analysis data")
    coaching: Optional[Dict[str, Any]] = Field(None, description="Coaching data if requested")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "track_id": "4uLU6hMCjMI75M1A2tKUQC",
                "track_name": "Bohemian Rhapsody",
                "artists": ["Queen"],
                "analysis": {},
                "coaching": {}
            }
        }


class TrackResult(BaseModel):
    """Single track result in search"""
    track_id: str
    track_name: str
    artists: List[str]
    album: Optional[str] = None
    duration_ms: Optional[int] = None
    preview_url: Optional[str] = None
    popularity: Optional[int] = None


class SearchResponse(BaseModel):
    """Response schema for track search"""
    success: bool = True
    query: str
    results: List[TrackResult]
    total: int
    limit: int
    offset: int
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "query": "Bohemian Rhapsody",
                "results": [],
                "total": 0,
                "limit": 10,
                "offset": 0
            }
        }


class RecommendationResult(BaseModel):
    """Single recommendation result"""
    track_id: str
    track_name: str
    artists: List[str]
    similarity_score: Optional[float] = None
    reason: Optional[str] = None
    album: Optional[str] = None
    preview_url: Optional[str] = None
    popularity: Optional[int] = None


class RecommendationResponse(BaseModel):
    """Response schema for recommendations"""
    success: bool = True
    track_id: str
    method: str
    recommendations: List[RecommendationResult]
    total: int
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "track_id": "4uLU6hMCjMI75M1A2tKUQC",
                "method": "similarity",
                "recommendations": [],
                "total": 0
            }
        }


class PlaylistResponse(BaseModel):
    """Response schema for generated playlist"""
    success: bool = True
    playlist: Dict[str, Any] = Field(..., description="Playlist data")
    criteria: Optional[Dict[str, Any]] = Field(None, description="Criteria used")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "playlist": {
                    "tracks": [],
                    "total_tracks": 0
                },
                "criteria": {}
            }
        }




