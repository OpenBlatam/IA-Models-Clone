"""
Dependency Injection Setup

Configures all services and dependencies in the DI container.
This module should be imported early in the application lifecycle.
"""

import logging
from typing import Optional

from ..core.di import get_container, register_service, register_instance

logger = logging.getLogger(__name__)


def setup_dependencies() -> None:
    """
    Configures all dependencies in the DI container.
    
    This function registers all services with their dependencies,
    enabling automatic dependency resolution throughout the application.
    """
    container = get_container()
    
    logger.info("Setting up dependency injection...")
    
    # ============================================
    # Infrastructure Layer - External Services
    # ============================================
    
    # ============================================
    # Infrastructure Layer - Cache (Register First)
    # ============================================
    
    try:
        from ..utils.cache import cache_manager
        register_instance("cache_manager", cache_manager)
        logger.debug("Registered: cache_manager")
        
        # Register Cache adapter (implements ICacheService)
        from ..infrastructure.adapters.cache_adapter import CacheServiceAdapter
        register_service(
            "cache_service",
            CacheServiceAdapter,
            singleton=True,
            dependencies=["cache_manager"]
        )
        logger.debug("Registered: cache_service")
    except ImportError as e:
        logger.warning(f"Could not register cache_manager: {e}")
    
    # ============================================
    # Infrastructure Layer - External Services
    # ============================================
    
    try:
        from ..services.spotify_service import SpotifyService
        register_service(
            "spotify_service",
            SpotifyService,
            singleton=True,
            dependencies=[]  # SpotifyService manages its own auth
        )
        logger.debug("Registered: spotify_service")
        
        # Register Spotify adapter (implements ISpotifyService)
        from ..infrastructure.adapters.spotify_adapter import SpotifyServiceAdapter
        register_service(
            "spotify_service_adapter",
            SpotifyServiceAdapter,
            singleton=True,
            dependencies=["spotify_service"]
        )
        logger.debug("Registered: spotify_service_adapter")
        
        # Register Spotify track repository (implements ITrackRepository)
        # Cache service is optional - repository will work without it
        from ..infrastructure.repositories.spotify_track_repository import SpotifyTrackRepository
        try:
            # Try to register with cache_service if available
            container.get("cache_service")
            register_service(
                "track_repository",
                SpotifyTrackRepository,
                singleton=True,
                dependencies=["spotify_service", "cache_service"]
            )
        except:
            # Fallback: register without cache_service
            register_service(
                "track_repository",
                SpotifyTrackRepository,
                singleton=True,
                dependencies=["spotify_service"]
            )
        logger.debug("Registered: track_repository")
    except ImportError as e:
        logger.warning(f"Could not register spotify_service: {e}")
    
    # ============================================
    # Core Domain Services
    # ============================================
    
    try:
        from ..core.music_analyzer import MusicAnalyzer
        register_service(
            "music_analyzer",
            MusicAnalyzer,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: music_analyzer")
        
        # Register Analysis adapter (implements IAnalysisService)
        from ..infrastructure.adapters.analysis_adapter import AnalysisServiceAdapter
        register_service(
            "analysis_service",
            AnalysisServiceAdapter,
            singleton=True,
            dependencies=["music_analyzer"]
        )
        logger.debug("Registered: analysis_service")
    except ImportError as e:
        logger.warning(f"Could not register music_analyzer: {e}")
    
    # ============================================
    # Application Services
    # ============================================
    
    # Analysis Services
    try:
        from ..services.comparison_service import ComparisonService
        register_service(
            "comparison_service",
            ComparisonService,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: comparison_service")
    except ImportError as e:
        logger.warning(f"Could not register comparison_service: {e}")
    
    try:
        from ..services.music_coach import MusicCoach
        register_service(
            "music_coach",
            MusicCoach,
            singleton=True,
            dependencies=["music_analyzer"]
        )
        logger.debug("Registered: music_coach")
        
        # Register Coaching adapter (implements ICoachingService)
        from ..infrastructure.adapters.coaching_adapter import CoachingServiceAdapter
        register_service(
            "coaching_service",
            CoachingServiceAdapter,
            singleton=True,
            dependencies=["music_coach"]
        )
        logger.debug("Registered: coaching_service")
    except ImportError as e:
        logger.warning(f"Could not register music_coach: {e}")
    
    # Export and History Services
    try:
        from ..services.export_service import ExportService
        register_service(
            "export_service",
            ExportService,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: export_service")
    except ImportError as e:
        logger.warning(f"Could not register export_service: {e}")
    
    try:
        from ..services.history_service import HistoryService
        register_service(
            "history_service",
            HistoryService,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: history_service")
    except ImportError as e:
        logger.warning(f"Could not register history_service: {e}")
    
    # User Services
    try:
        from ..services.auth_service import AuthService
        register_service(
            "auth_service",
            AuthService,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: auth_service")
    except ImportError as e:
        logger.warning(f"Could not register auth_service: {e}")
    
    try:
        from ..services.favorites_service import FavoritesService
        register_service(
            "favorites_service",
            FavoritesService,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: favorites_service")
    except ImportError as e:
        logger.warning(f"Could not register favorites_service: {e}")
    
    try:
        from ..services.tagging_service import TaggingService
        register_service(
            "tagging_service",
            TaggingService,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: tagging_service")
    except ImportError as e:
        logger.warning(f"Could not register tagging_service: {e}")
    
    # Playlist Services
    try:
        from ..services.playlist_service import PlaylistService
        register_service(
            "playlist_service",
            PlaylistService,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: playlist_service")
    except ImportError as e:
        logger.warning(f"Could not register playlist_service: {e}")
    
    try:
        from ..services.playlist_analyzer import PlaylistAnalyzer
        register_service(
            "playlist_analyzer",
            PlaylistAnalyzer,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: playlist_analyzer")
    except ImportError as e:
        logger.warning(f"Could not register playlist_analyzer: {e}")
    
    # Recommendation Services
    try:
        from ..services.intelligent_recommender import IntelligentRecommender
        register_service(
            "intelligent_recommender",
            IntelligentRecommender,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: intelligent_recommender")
    except ImportError as e:
        logger.warning(f"Could not register intelligent_recommender: {e}")
    
    try:
        from ..services.contextual_recommender import ContextualRecommender
        register_service(
            "contextual_recommender",
            ContextualRecommender,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: contextual_recommender")
        
        # Register Recommendation adapter (implements IRecommendationService)
        from ..infrastructure.adapters.recommendation_adapter import RecommendationServiceAdapter
        register_service(
            "recommendation_service",
            RecommendationServiceAdapter,
            singleton=True,
            dependencies=["intelligent_recommender", "contextual_recommender"]
        )
        logger.debug("Registered: recommendation_service")
    except ImportError as e:
        logger.warning(f"Could not register contextual_recommender: {e}")
    
    # Analytics and Dashboard
    try:
        from ..services.analytics_service import analytics_service
        register_instance("analytics_service", analytics_service)
        logger.debug("Registered: analytics_service")
    except ImportError as e:
        logger.warning(f"Could not register analytics_service: {e}")
    
    try:
        from ..services.dashboard_service import DashboardService
        register_service(
            "dashboard_service",
            DashboardService,
            singleton=True,
            dependencies=["analytics_service"]
        )
        logger.debug("Registered: dashboard_service")
    except ImportError as e:
        logger.warning(f"Could not register dashboard_service: {e}")
    
    # Notification and Webhook Services
    try:
        from ..services.notification_service import NotificationService
        register_service(
            "notification_service",
            NotificationService,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: notification_service")
    except ImportError as e:
        logger.warning(f"Could not register notification_service: {e}")
    
    try:
        from ..services.webhook_service import WebhookService
        register_service(
            "webhook_service",
            WebhookService,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: webhook_service")
    except ImportError as e:
        logger.warning(f"Could not register webhook_service: {e}")
    
    # Analysis Services
    try:
        from ..services.trends_analyzer import TrendsAnalyzer
        register_service(
            "trends_analyzer",
            TrendsAnalyzer,
            singleton=True,
            dependencies=["spotify_service"]
        )
        logger.debug("Registered: trends_analyzer")
    except ImportError as e:
        logger.warning(f"Could not register trends_analyzer: {e}")
    
    try:
        from ..services.collaboration_analyzer import CollaborationAnalyzer
        register_service(
            "collaboration_analyzer",
            CollaborationAnalyzer,
            singleton=True,
            dependencies=["spotify_service"]
        )
        logger.debug("Registered: collaboration_analyzer")
    except ImportError as e:
        logger.warning(f"Could not register collaboration_analyzer: {e}")
    
    try:
        from ..services.alert_service import AlertService
        register_service(
            "alert_service",
            AlertService,
            singleton=True,
            dependencies=["trends_analyzer"]
        )
        logger.debug("Registered: alert_service")
    except ImportError as e:
        logger.warning(f"Could not register alert_service: {e}")
    
    try:
        from ..services.temporal_analyzer import TemporalAnalyzer
        register_service(
            "temporal_analyzer",
            TemporalAnalyzer,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: temporal_analyzer")
    except ImportError as e:
        logger.warning(f"Could not register temporal_analyzer: {e}")
    
    try:
        from ..services.quality_analyzer import QualityAnalyzer
        register_service(
            "quality_analyzer",
            QualityAnalyzer,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: quality_analyzer")
    except ImportError as e:
        logger.warning(f"Could not register quality_analyzer: {e}")
    
    try:
        from ..services.artist_comparator import ArtistComparator
        register_service(
            "artist_comparator",
            ArtistComparator,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: artist_comparator")
    except ImportError as e:
        logger.warning(f"Could not register artist_comparator: {e}")
    
    try:
        from ..services.discovery_service import DiscoveryService
        register_service(
            "discovery_service",
            DiscoveryService,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: discovery_service")
    except ImportError as e:
        logger.warning(f"Could not register discovery_service: {e}")
    
    try:
        from ..services.cover_remix_analyzer import CoverRemixAnalyzer
        register_service(
            "cover_remix_analyzer",
            CoverRemixAnalyzer,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: cover_remix_analyzer")
    except ImportError as e:
        logger.warning(f"Could not register cover_remix_analyzer: {e}")
    
    try:
        from ..services.instrumentation_analyzer import InstrumentationAnalyzer
        register_service(
            "instrumentation_analyzer",
            InstrumentationAnalyzer,
            singleton=True,
            dependencies=["spotify_service", "music_analyzer"]
        )
        logger.debug("Registered: instrumentation_analyzer")
    except ImportError as e:
        logger.warning(f"Could not register instrumentation_analyzer: {e}")
    
    try:
        from ..services.trend_predictor import TrendPredictor
        register_service(
            "trend_predictor",
            TrendPredictor,
            singleton=True,
            dependencies=["trends_analyzer"]
        )
        logger.debug("Registered: trend_predictor")
    except ImportError as e:
        logger.warning(f"Could not register trend_predictor: {e}")
    
    # Optional Advanced Services
    try:
        from ..services.lyrics_analyzer import LyricsAnalyzer
        register_service(
            "lyrics_analyzer",
            LyricsAnalyzer,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: lyrics_analyzer (optional)")
    except ImportError:
        pass
    
    try:
        from ..services.deep_learning_service import DeepLearningService
        register_service(
            "deep_learning_service",
            DeepLearningService,
            singleton=True,
            dependencies=[]
        )
        logger.debug("Registered: deep_learning_service (optional)")
    except ImportError:
        pass
    
    # ============================================
    # Application Layer - Use Cases
    # Note: Use cases depend on services above, so register them last
    # ============================================
    
    try:
        from ..application.use_cases.analysis import AnalyzeTrackUseCase, SearchTracksUseCase
        from ..application.use_cases.recommendations import GetRecommendationsUseCase, GeneratePlaylistUseCase
        
        # Analysis Use Cases
        # Now using proper repositories and adapters
        register_service(
            "analyze_track_use_case",
            AnalyzeTrackUseCase,
            singleton=True,
            dependencies=["spotify_service_adapter", "track_repository", "analysis_service", "coaching_service"]
        )
        logger.debug("Registered: analyze_track_use_case")
        
        register_service(
            "search_tracks_use_case",
            SearchTracksUseCase,
            singleton=True,
            dependencies=["track_repository"]
        )
        logger.debug("Registered: search_tracks_use_case")
        
        # Recommendation Use Cases
        register_service(
            "get_recommendations_use_case",
            GetRecommendationsUseCase,
            singleton=True,
            dependencies=["track_repository", "recommendation_service"]
        )
        logger.debug("Registered: get_recommendations_use_case")
        
        register_service(
            "generate_playlist_use_case",
            GeneratePlaylistUseCase,
            singleton=True,
            dependencies=["recommendation_service"]
        )
        logger.debug("Registered: generate_playlist_use_case")
    except ImportError as e:
        logger.warning(f"Could not register use cases: {e}")
    
    logger.info("Dependency injection setup completed successfully")


def get_service(service_name: str) -> Optional[Any]:
    """
    Helper function to get a service from the container.
    
    Args:
        service_name: Name of the service to retrieve.
    
    Returns:
        Service instance or None if not found.
    """
    from ..core.di import get_service as _get_service
    try:
        return _get_service(service_name)
    except ValueError:
        logger.warning(f"Service '{service_name}' not found in container")
        return None

