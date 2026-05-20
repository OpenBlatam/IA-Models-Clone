"""
API v1 Schemas

Pydantic models for request/response validation.
"""

from .requests import (
    AnalyzeTrackRequest,
    SearchTracksRequest,
    GeneratePlaylistRequest
)
from .responses import (
    AnalysisResponse,
    SearchResponse,
    RecommendationResponse,
    PlaylistResponse,
    ErrorResponse
)

__all__ = [
    "AnalyzeTrackRequest",
    "SearchTracksRequest",
    "GeneratePlaylistRequest",
    "AnalysisResponse",
    "SearchResponse",
    "RecommendationResponse",
    "PlaylistResponse",
    "ErrorResponse",
]




