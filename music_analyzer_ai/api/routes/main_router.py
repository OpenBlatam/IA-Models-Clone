"""
Main router that aggregates all domain routers
"""

from fastapi import APIRouter
import logging

from .search import get_search_router
from .analysis import get_analysis_router
from .tracks import get_tracks_router
from .coaching import get_coaching_router
from .comparison import get_comparison_router
from .cache import get_cache_router
from .export import get_export_router
from .history import get_history_router
from .analytics import get_analytics_router
from .favorites import get_favorites_router
from .tags import get_tags_router
from .webhooks import get_webhooks_router
from .auth import get_auth_router
from .playlists import get_playlists_router
from .recommendations import get_recommendations_router
from .dashboard import get_dashboard_router
from .notifications import get_notifications_router
from .trends import get_trends_router
from .collaborations import get_collaborations_router
from .alerts import get_alerts_router
from .temporal import get_temporal_router
from .quality import get_quality_router
from .artists import get_artists_router
from .discovery import get_discovery_router
from .covers_remixes import get_covers_remixes_router
from .remixes import get_remixes_router
from .instrumentation import get_instrumentation_router
from .playlist_analysis import get_playlist_analysis_router
from .predictions import get_predictions_router
from .health import get_health_router

logger = logging.getLogger(__name__)


def create_main_router() -> APIRouter:
    """
    Create the main router with all domain routers included
    """
    main_router = APIRouter()
    
    # Include all domain routers
    routers = [
        get_search_router(),
        get_analysis_router(),
        get_tracks_router(),
        get_coaching_router(),
        get_comparison_router(),
        get_cache_router(),
        get_export_router(),
        get_history_router(),
        get_analytics_router(),
        get_favorites_router(),
        get_tags_router(),
        get_webhooks_router(),
        get_auth_router(),
        get_playlists_router(),
        get_recommendations_router(),
        get_dashboard_router(),
        get_notifications_router(),
        get_trends_router(),
        get_collaborations_router(),
        get_alerts_router(),
        get_temporal_router(),
        get_quality_router(),
        get_artists_router(),
        get_discovery_router(),
        get_covers_remixes_router(),
        get_remixes_router(),
        get_instrumentation_router(),
        get_playlist_analysis_router(),
        get_predictions_router(),
        get_health_router(),
    ]
    
    for router_instance in routers:
        main_router.include_router(router_instance.router)
    
    logger.info(f"Main router created with {len(routers)} domain routers")
    
    return main_router

