"""
Analysis helper functions for common analysis patterns
"""

from typing import Dict, Any, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


async def perform_track_analysis(
    spotify_service: Any,
    music_analyzer: Any,
    track_id: str
) -> Dict[str, Any]:
    """
    Perform complete track analysis
    
    Args:
        spotify_service: Spotify service instance
        music_analyzer: Music analyzer service instance
        track_id: Track ID to analyze
    
    Returns:
        Complete analysis dictionary
    """
    spotify_data = spotify_service.get_track_full_analysis(track_id)
    analysis = music_analyzer.analyze_track(spotify_data)
    
    return {
        "track_basic_info": analysis["track_basic_info"],
        "musical_analysis": analysis["musical_analysis"],
        "technical_analysis": analysis["technical_analysis"],
        "composition_analysis": analysis["composition_analysis"],
        "performance_analysis": analysis.get("performance_analysis", {}),
        "educational_insights": analysis["educational_insights"]
    }


def add_coaching_to_analysis(
    analysis: Dict[str, Any],
    music_coach: Any
) -> Dict[str, Any]:
    """
    Add coaching analysis to existing analysis
    
    Args:
        analysis: Existing analysis dictionary
        music_coach: Music coach service instance
    
    Returns:
        Analysis with coaching added
    """
    coaching = music_coach.generate_coaching_analysis(analysis)
    analysis["coaching"] = coaching
    return analysis


async def trigger_webhook_safe(
    webhook_service: Any,
    event_type: Any,
    data: Dict[str, Any]
) -> None:
    """
    Safely trigger a webhook with error handling.
    
    Uses background_helpers for consistent background task execution.
    
    Args:
        webhook_service: Webhook service instance
        event_type: Webhook event type
        data: Event data
    """
    from .background_helpers import run_background_task
    
    await run_background_task(
        webhook_service.trigger_webhook,
        event_type,
        data,
        task_name=f"webhook_{event_type}"
    )


def save_analysis_to_history(
    history_service: Any,
    analytics_service: Any,
    track_id: str,
    analysis: Dict[str, Any],
    user_id: Optional[str] = None
) -> None:
    """
    Save analysis to history and track analytics
    
    Args:
        history_service: History service instance
        analytics_service: Analytics service instance
        track_id: Track ID
        analysis: Analysis data
        user_id: Optional user ID
    """
    try:
        history_service.add_analysis(
            track_id=track_id,
            track_name=analysis["track_basic_info"]["name"],
            artists=analysis["track_basic_info"]["artists"],
            analysis=analysis,
            user_id=user_id
        )
        analytics_service.track_analysis(track_id, user_id)
    except Exception as e:
        logger.warning(f"Error saving to history: {e}")

