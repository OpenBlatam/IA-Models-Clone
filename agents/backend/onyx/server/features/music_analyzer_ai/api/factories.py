"""
Service Factories

Factory functions to get services from DI container.
This replaces direct service instantiation.
"""

import logging
from typing import Optional

from ..core.di import get_container

logger = logging.getLogger(__name__)


def get_spotify_service():
    """Get SpotifyService from DI container"""
    try:
        return get_container().get("spotify_service")
    except Exception as e:
        logger.error(f"Failed to get spotify_service: {e}")
        raise


def get_music_analyzer():
    """Get MusicAnalyzer from DI container"""
    try:
        return get_container().get("music_analyzer")
    except Exception as e:
        logger.error(f"Failed to get music_analyzer: {e}")
        raise


def get_music_coach():
    """Get MusicCoach from DI container"""
    try:
        return get_container().get("music_coach")
    except Exception as e:
        logger.warning(f"Failed to get music_coach: {e}")
        return None


def get_comparison_service():
    """Get ComparisonService from DI container"""
    try:
        return get_container().get("comparison_service")
    except Exception as e:
        logger.warning(f"Failed to get comparison_service: {e}")
        return None


def get_export_service():
    """Get ExportService from DI container"""
    try:
        return get_container().get("export_service")
    except Exception as e:
        logger.warning(f"Failed to get export_service: {e}")
        return None


def get_history_service():
    """Get HistoryService from DI container"""
    try:
        return get_container().get("history_service")
    except Exception as e:
        logger.warning(f"Failed to get history_service: {e}")
        return None


def get_webhook_service():
    """Get WebhookService from DI container"""
    try:
        return get_container().get("webhook_service")
    except Exception as e:
        logger.warning(f"Failed to get webhook_service: {e}")
        return None


def get_favorites_service():
    """Get FavoritesService from DI container"""
    try:
        return get_container().get("favorites_service")
    except Exception as e:
        logger.warning(f"Failed to get favorites_service: {e}")
        return None


def get_tagging_service():
    """Get TaggingService from DI container"""
    try:
        return get_container().get("tagging_service")
    except Exception as e:
        logger.warning(f"Failed to get tagging_service: {e}")
        return None


def get_playlist_service():
    """Get PlaylistService from DI container"""
    try:
        return get_container().get("playlist_service")
    except Exception as e:
        logger.warning(f"Failed to get playlist_service: {e}")
        return None


def get_intelligent_recommender():
    """Get IntelligentRecommender from DI container"""
    try:
        return get_container().get("intelligent_recommender")
    except Exception as e:
        logger.warning(f"Failed to get intelligent_recommender: {e}")
        return None


def get_dashboard_service():
    """Get DashboardService from DI container"""
    try:
        return get_container().get("dashboard_service")
    except Exception as e:
        logger.warning(f"Failed to get dashboard_service: {e}")
        return None


def get_notification_service():
    """Get NotificationService from DI container"""
    try:
        return get_container().get("notification_service")
    except Exception as e:
        logger.warning(f"Failed to get notification_service: {e}")
        return None


def get_analytics_service():
    """Get AnalyticsService from DI container"""
    try:
        return get_container().get("analytics_service")
    except Exception as e:
        logger.warning(f"Failed to get analytics_service: {e}")
        return None


def get_service(service_name: str):
    """
    Generic factory function to get any service from DI container.
    
    Args:
        service_name: Name of the service to retrieve
    
    Returns:
        Service instance or None if not found
    """
    try:
        return get_container().get(service_name)
    except Exception as e:
        logger.warning(f"Failed to get service {service_name}: {e}")
        return None




