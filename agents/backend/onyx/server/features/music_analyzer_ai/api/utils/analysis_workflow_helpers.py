"""
Analysis workflow helper functions.

This module provides utilities for complete analysis workflows,
combining multiple operations into reusable patterns.
"""

from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


async def perform_complete_analysis_workflow(
    track_id: str,
    spotify_service: Any,
    music_analyzer: Any,
    include_coaching: bool = False,
    music_coach: Optional[Any] = None,
    save_history: bool = True,
    trigger_webhook: bool = True,
    history_service: Optional[Any] = None,
    analytics_service: Optional[Any] = None,
    webhook_service: Optional[Any] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform complete track analysis workflow.
    
    This helper combines all the common steps:
    1. Get track data from Spotify
    2. Analyze the track
    3. Build response
    4. Optionally add coaching
    5. Optionally save to history/analytics
    6. Optionally trigger webhook
    
    Args:
        track_id: Track ID to analyze
        spotify_service: Spotify service instance
        music_analyzer: Music analyzer instance
        include_coaching: Whether to include coaching analysis
        music_coach: Optional music coach instance
        save_history: Whether to save to history
        trigger_webhook: Whether to trigger webhook
        history_service: Optional history service
        analytics_service: Optional analytics service
        webhook_service: Optional webhook service
        user_id: Optional user ID
    
    Returns:
        Complete analysis response dictionary
    
    Example:
        response = await perform_complete_analysis_workflow(
            track_id,
            spotify_service,
            music_analyzer,
            include_coaching=request.include_coaching,
            music_coach=get_music_coach(),
            history_service=get_history_service(),
            analytics_service=get_analytics_service(),
            webhook_service=get_webhook_service()
        )
    """
    from .response_helpers import build_analysis_response_from_dict
    from .conditional_helpers import execute_with_service
    from .safe_operation_helpers import safe_execute_multiple
    from .background_helpers import run_background_task
    from .object_helpers import safe_get_attribute
    
    # Step 1: Get track data
    spotify_data = spotify_service.get_track_full_analysis(track_id)
    
    # Step 2: Analyze
    analysis = music_analyzer.analyze_track(spotify_data)
    
    # Step 3: Build response
    response = build_analysis_response_from_dict(analysis)
    
    # Step 4: Add coaching if requested
    if include_coaching and music_coach:
        coaching = await execute_with_service(
            music_coach,
            "generate_coaching_analysis",
            analysis
        )
        if coaching:
            response["coaching"] = coaching
    
    # Step 5: Save to history (safe operation)
    if save_history:
        await safe_execute_multiple([
            (
                lambda: execute_with_service(
                    history_service,
                    "add_analysis",
                    track_id,
                    safe_get_attribute(analysis, "track_basic_info.name"),
                    safe_get_attribute(analysis, "track_basic_info.artists"),
                    response,
                    user_id
                ),
                (),
                {},
                "save_history"
            ),
            (
                lambda: execute_with_service(
                    analytics_service,
                    "track_analysis",
                    track_id,
                    user_id
                ),
                (),
                {},
                "track_analytics"
            ),
        ])
    
    # Step 6: Trigger webhook (background task)
    if trigger_webhook and webhook_service:
        await run_background_task(
            lambda: execute_with_service(
                webhook_service,
                "trigger_webhook",
                "ANALYSIS_COMPLETED",  # WebhookEvent.ANALYSIS_COMPLETED
                {
                    "track_id": track_id,
                    "track_name": safe_get_attribute(analysis, "track_basic_info.name"),
                    "analysis_id": track_id
                }
            ),
            task_name="analysis_webhook"
        )
    
    return response








