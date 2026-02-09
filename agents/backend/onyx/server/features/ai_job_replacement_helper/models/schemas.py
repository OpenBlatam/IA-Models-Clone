"""
Pydantic schemas for API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class UserProfile(BaseModel):
    """Perfil del usuario"""
    user_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    current_skills: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    target_industry: Optional[str] = None
    location: Optional[str] = None
    experience_level: Optional[str] = None


class JobSwipeRequest(BaseModel):
    """Request para hacer swipe en un trabajo"""
    job_id: str
    action: str  # like, dislike, save, apply


class JobSwipeResponse(BaseModel):
    """Response de swipe"""
    success: bool
    action: str
    job_id: str
    timestamp: str


class StepStartRequest(BaseModel):
    """Request para iniciar un paso"""
    step_id: str


class StepCompleteRequest(BaseModel):
    """Request para completar un paso"""
    step_id: str
    notes: Optional[str] = None


class UserProgressResponse(BaseModel):
    """Response con progreso del usuario"""
    user_id: str
    gamification: Dict[str, Any]
    steps: Dict[str, Any]
    jobs: Dict[str, Any]
    recommendations: Optional[Dict[str, Any]] = None


class RecommendationsResponse(BaseModel):
    """Response con recomendaciones"""
    skills: List[Dict[str, Any]]
    jobs: List[Dict[str, Any]]
    next_steps: List[Dict[str, Any]]


class JobSearchRequest(BaseModel):
    """Request para buscar trabajos"""
    keywords: Optional[str] = None
    location: Optional[str] = None
    experience_level: Optional[str] = None
    job_type: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=50)




