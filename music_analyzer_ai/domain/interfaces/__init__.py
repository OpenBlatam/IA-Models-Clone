"""
Domain Interfaces

Interfaces that define contracts for domain services and repositories.
These interfaces belong to the domain layer and are implemented in the infrastructure layer.
"""

from .repositories import ITrackRepository, IUserRepository, IPlaylistRepository
from .analysis import (
    IAnalysisService,
    IHarmonicAnalyzer,
    IStructureAnalyzer,
    IEmotionAnalyzer,
    IGenreDetector
)
from .recommendations import IRecommendationService
from .coaching import ICoachingService
from .spotify import ISpotifyService
from .export import IExportService
from .cache import ICacheService

__all__ = [
    # Repositories
    "ITrackRepository",
    "IUserRepository",
    "IPlaylistRepository",
    # Analysis
    "IAnalysisService",
    "IHarmonicAnalyzer",
    "IStructureAnalyzer",
    "IEmotionAnalyzer",
    "IGenreDetector",
    # Services
    "IRecommendationService",
    "ICoachingService",
    "ISpotifyService",
    "IExportService",
    "ICacheService",
]




