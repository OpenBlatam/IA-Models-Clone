"""
Adapters

Adapters that wrap existing services to implement domain interfaces.
These allow gradual migration from existing services to new architecture.
"""

from .spotify_adapter import SpotifyServiceAdapter
from .analysis_adapter import AnalysisServiceAdapter
from .coaching_adapter import CoachingServiceAdapter
from .recommendation_adapter import RecommendationServiceAdapter

__all__ = [
    "SpotifyServiceAdapter",
    "AnalysisServiceAdapter",
    "CoachingServiceAdapter",
    "RecommendationServiceAdapter",
]




