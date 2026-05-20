"""
Pure functions for progress tracking
Refactored from class-based to functional approach
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from utils.date_helpers import (
    get_current_utc,
    parse_iso_date,
    days_between,
    get_start_of_day
)
from services.functions.progress_functions import (
    calculate_days_sober,
    calculate_streak,
    calculate_longest_streak,
    calculate_progress_percentage
)


def log_entry(
    user_id: str,
    date: str,
    mood: str,
    cravings_level: int,
    triggers_encountered: List[str],
    consumed: bool,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Log daily entry (RORO pattern)
    
    Args:
        user_id: User identifier
        date: Entry date (ISO format)
        mood: Mood state
        cravings_level: Cravings level (0-10)
        triggers_encountered: List of triggers
        consumed: Whether user consumed
        notes: Optional notes
    
    Returns:
        Dictionary with logged entry
    """
    # Guard clauses
    if not user_id:
        raise ValueError("user_id is required")
    
    if not date:
        raise ValueError("date is required")
    
    if cravings_level < 0 or cravings_level > 10:
        raise ValueError("cravings_level must be between 0 and 10")
    
    # Parse and validate date
    entry_date = parse_iso_date(date)
    if entry_date > get_current_utc():
        raise ValueError("date cannot be in the future")
    
    # Return object (RORO pattern)
    return {
        "entry_id": f"entry_{user_id}_{entry_date.timestamp()}",
        "user_id": user_id,
        "date": date,
        "mood": mood,
        "cravings_level": cravings_level,
        "triggers_encountered": triggers_encountered,
        "consumed": consumed,
        "notes": notes,
        "created_at": get_current_utc().isoformat()
    }


def get_progress(
    user_id: str,
    start_date: Optional[datetime],
    entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Get user progress (RORO pattern)
    
    Args:
        user_id: User identifier
        start_date: Optional start date for filtering
        entries: List of log entries
    
    Returns:
        Dictionary with progress data
    """
    # Guard clause
    if not user_id:
        raise ValueError("user_id is required")
    
    # Filter entries if start_date provided
    if start_date:
        filtered_entries = [
            e for e in entries
            if parse_iso_date(e.get("date", "")) >= start_date
        ]
    else:
        filtered_entries = entries
    
    # Calculate metrics using pure functions
    days_sober = calculate_days_sober(filtered_entries)
    streak_days = calculate_streak(filtered_entries)
    longest_streak = calculate_longest_streak(entries)
    progress_percentage = calculate_progress_percentage(days_sober, 365)
    
    # Return object (RORO pattern)
    return {
        "user_id": user_id,
        "days_sober": days_sober,
        "total_entries": len(filtered_entries),
        "streak_days": streak_days,
        "longest_streak": longest_streak,
        "progress_percentage": progress_percentage,
        "recent_entries": filtered_entries[-10:] if filtered_entries else []
    }


def get_stats(
    user_id: str,
    entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Get user statistics (RORO pattern)
    
    Args:
        user_id: User identifier
        entries: List of log entries
    
    Returns:
        Dictionary with statistics
    """
    # Guard clause
    if not user_id:
        raise ValueError("user_id is required")
    
    if not entries:
        return {
            "user_id": user_id,
            "total_days": 0,
            "days_sober": 0,
            "relapse_count": 0,
            "average_cravings": 0.0,
            "most_common_triggers": [],
            "trends": {}
        }
    
    # Calculate statistics
    days_sober = calculate_days_sober(entries)
    relapse_count = sum(1 for e in entries if e.get("consumed", False))
    
    cravings_levels = [e.get("cravings_level", 0) for e in entries if e.get("cravings_level") is not None]
    average_cravings = sum(cravings_levels) / len(cravings_levels) if cravings_levels else 0.0
    
    # Get most common triggers
    all_triggers = []
    for e in entries:
        all_triggers.extend(e.get("triggers_encountered", []))
    
    trigger_counts = {}
    for trigger in all_triggers:
        trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
    
    most_common_triggers = sorted(
        trigger_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    # Return object (RORO pattern)
    return {
        "user_id": user_id,
        "total_days": len(entries),
        "days_sober": days_sober,
        "relapse_count": relapse_count,
        "average_cravings": round(average_cravings, 2),
        "most_common_triggers": [t[0] for t in most_common_triggers],
        "trends": {
            "cravings_trend": "decreasing" if len(cravings_levels) > 1 and cravings_levels[-1] < cravings_levels[0] else "stable"
        }
    }

