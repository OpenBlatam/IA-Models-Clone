"""
Request handlers for progress endpoints
Pure functions for processing requests
"""

from typing import Optional

try:
    from schemas.progress import (
        LogEntryRequest,
        LogEntryResponse,
        ProgressResponse,
        StatsResponse,
        TimelineResponse
    )
    from dependencies import ProgressTrackerDep
    from utils.validators import validate_date_string
except ImportError:
    from ...schemas.progress import (
        LogEntryRequest,
        LogEntryResponse,
        ProgressResponse,
        StatsResponse,
        TimelineResponse
    )
    from ...dependencies import ProgressTrackerDep
    from ...utils.validators import validate_date_string


async def create_log_entry(
    request: LogEntryRequest,
    tracker: ProgressTrackerDep
) -> LogEntryResponse:
    """Create log entry"""
    entry = tracker.log_entry(
        request.user_id,
        request.date,
        request.mood,
        request.cravings_level,
        request.triggers_encountered,
        request.consumed,
        request.notes
    )
    
    return LogEntryResponse(
        entry_id=entry.get("entry_id", ""),
        user_id=request.user_id,
        date=request.date,
        mood=request.mood,
        cravings_level=request.cravings_level,
        triggers_encountered=request.triggers_encountered,
        consumed=request.consumed,
        notes=request.notes
    )


async def get_user_progress(
    user_id: str,
    start_date: Optional[str],
    tracker: ProgressTrackerDep
) -> ProgressResponse:
    """Get user progress"""
    start = None
    if start_date:
        start = validate_date_string(start_date, "start_date")
    
    progress = tracker.get_progress(user_id, start, [])
    
    return ProgressResponse(
        user_id=user_id,
        days_sober=progress.get("days_sober", 0),
        total_entries=progress.get("total_entries", 0),
        streak_days=progress.get("streak_days", 0),
        longest_streak=progress.get("longest_streak", 0),
        progress_percentage=progress.get("progress_percentage", 0.0),
        recent_entries=progress.get("recent_entries", [])
    )


async def get_user_stats(
    user_id: str,
    tracker: ProgressTrackerDep
) -> StatsResponse:
    """Get user statistics"""
    stats = tracker.get_stats(user_id, [])
    
    return StatsResponse(
        user_id=user_id,
        total_days=stats.get("total_days", 0),
        days_sober=stats.get("days_sober", 0),
        relapse_count=stats.get("relapse_count", 0),
        average_cravings=stats.get("average_cravings", 0.0),
        most_common_triggers=stats.get("most_common_triggers", []),
        trends=stats.get("trends", {})
    )


async def get_user_timeline(
    user_id: str,
    tracker: ProgressTrackerDep
) -> TimelineResponse:
    """Get user timeline"""
    timeline = tracker.get_timeline(user_id, [])
    
    return TimelineResponse(
        user_id=user_id,
        timeline=timeline.get("timeline", []),
        milestones=timeline.get("milestones", []),
        relapses=timeline.get("relapses", [])
    )

