"""
Pydantic models for API requests and responses
"""

from .schemas import (
    UserProfile,
    JobSwipeRequest,
    JobSwipeResponse,
    StepStartRequest,
    StepCompleteRequest,
    UserProgressResponse,
    RecommendationsResponse,
)

__all__ = [
    "UserProfile",
    "JobSwipeRequest",
    "JobSwipeResponse",
    "StepStartRequest",
    "StepCompleteRequest",
    "UserProgressResponse",
    "RecommendationsResponse",
]




