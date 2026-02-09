"""
FastAPI Dependencies for Dependency Injection

Provides FastAPI dependency functions for injecting services into route handlers.
"""

from fastapi import Depends
from typing import Any, Optional
import logging

from ..core.di import get_service

logger = logging.getLogger(__name__)


def get_spotify_service():
    """Dependency to get Spotify service"""
    return get_service("spotify_service")


def get_music_analyzer():
    """Dependency to get Music Analyzer"""
    return get_service("music_analyzer")


def get_music_coach():
    """Dependency to get Music Coach service"""
    return get_service("music_coach")


def get_comparison_service():
    """Dependency to get Comparison service"""
    return get_service("comparison_service")


def get_export_service():
    """Dependency to get Export service"""
    return get_service("export_service")


def get_history_service():
    """Dependency to get History service"""
    return get_service("history_service")


def get_webhook_service():
    """Dependency to get Webhook service"""
    return get_service("webhook_service")


def get_favorites_service():
    """Dependency to get Favorites service"""
    return get_service("favorites_service")


def get_tagging_service():
    """Dependency to get Tagging service"""
    return get_service("tagging_service")


def get_auth_service():
    """Dependency to get Auth service"""
    return get_service("auth_service")


def get_playlist_service():
    """Dependency to get Playlist service"""
    return get_service("playlist_service")


def get_intelligent_recommender():
    """Dependency to get Intelligent Recommender"""
    return get_service("intelligent_recommender")


def get_dashboard_service():
    """Dependency to get Dashboard service"""
    return get_service("dashboard_service")


def get_notification_service():
    """Dependency to get Notification service"""
    return get_service("notification_service")


def get_analytics_service():
    """Dependency to get Analytics service"""
    return get_service("analytics_service")


def get_trends_analyzer():
    """Dependency to get Trends Analyzer"""
    return get_service("trends_analyzer")


def get_collaboration_analyzer():
    """Dependency to get Collaboration Analyzer"""
    return get_service("collaboration_analyzer")


def get_alert_service():
    """Dependency to get Alert service"""
    return get_service("alert_service")


def get_temporal_analyzer():
    """Dependency to get Temporal Analyzer"""
    return get_service("temporal_analyzer")


def get_quality_analyzer():
    """Dependency to get Quality Analyzer"""
    return get_service("quality_analyzer")


def get_contextual_recommender():
    """Dependency to get Contextual Recommender"""
    return get_service("contextual_recommender")


def get_playlist_analyzer():
    """Dependency to get Playlist Analyzer"""
    return get_service("playlist_analyzer")


def get_artist_comparator():
    """Dependency to get Artist Comparator"""
    return get_service("artist_comparator")


def get_discovery_service():
    """Dependency to get Discovery service"""
    return get_service("discovery_service")


def get_cover_remix_analyzer():
    """Dependency to get Cover/Remix Analyzer"""
    return get_service("cover_remix_analyzer")


def get_instrumentation_analyzer():
    """Dependency to get Instrumentation Analyzer"""
    return get_service("instrumentation_analyzer")


def get_trend_predictor():
    """Dependency to get Trend Predictor"""
    return get_service("trend_predictor")


def get_optional_service(service_name: str) -> Optional[Any]:
    """
    Generic dependency function for optional services.
    
    Args:
        service_name: Name of the service to retrieve.
    
    Returns:
        Service instance or None if not available.
    """
    try:
        return get_service(service_name)
    except ValueError:
        logger.debug(f"Optional service '{service_name}' not available")
        return None


# ============================================
# Use Case Dependencies
# ============================================

def get_analyze_track_use_case():
    """Dependency to get AnalyzeTrackUseCase"""
    return get_service("analyze_track_use_case")


def get_search_tracks_use_case():
    """Dependency to get SearchTracksUseCase"""
    return get_service("search_tracks_use_case")


def get_recommendations_use_case():
    """Dependency to get GetRecommendationsUseCase"""
    return get_service("get_recommendations_use_case")


def get_generate_playlist_use_case():
    """Dependency to get GeneratePlaylistUseCase"""
    return get_service("generate_playlist_use_case")

