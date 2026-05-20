"""
Pure functions for progress tracking logic
Functional programming approach - no classes
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


def calculate_days_sober(entries: List[Dict[str, Any]]) -> int:
    """Calculate days sober from entries"""
    if not entries:
        return 0
    
    # Sort entries by date
    sorted_entries = sorted(entries, key=lambda x: x.get("date", ""))
    
    # Find last consumption
    last_consumption_date = None
    for entry in reversed(sorted_entries):
        if entry.get("consumed", False):
            last_consumption_date = entry.get("date")
            break
    
    if not last_consumption_date:
        # No consumption found, calculate from first entry
        first_entry_date = sorted_entries[0].get("date")
        if first_entry_date:
            try:
                first_date = datetime.fromisoformat(first_entry_date)
                return (datetime.now() - first_date).days
            except ValueError:
                return 0
        return 0
    
    try:
        last_date = datetime.fromisoformat(last_consumption_date)
        return (datetime.now() - last_date).days
    except ValueError:
        return 0


def calculate_streak(entries: List[Dict[str, Any]]) -> int:
    """Calculate current streak in days"""
    if not entries:
        return 0
    
    # Sort entries by date descending
    sorted_entries = sorted(
        entries,
        key=lambda x: x.get("date", ""),
        reverse=True
    )
    
    streak = 0
    current_date = datetime.now().date()
    
    for entry in sorted_entries:
        entry_date_str = entry.get("date", "")
        if not entry_date_str:
            continue
        
        try:
            entry_date = datetime.fromisoformat(entry_date_str).date()
            
            # Check if entry is for today or yesterday (allowing for timezone)
            days_diff = (current_date - entry_date).days
            
            if days_diff > 1:
                break
            
            if not entry.get("consumed", False):
                streak += 1
                current_date = entry_date - timedelta(days=1)
            else:
                break
        except ValueError:
            continue
    
    return streak


def calculate_longest_streak(entries: List[Dict[str, Any]]) -> int:
    """Calculate longest streak from entries"""
    if not entries:
        return 0
    
    sorted_entries = sorted(entries, key=lambda x: x.get("date", ""))
    
    longest = 0
    current = 0
    
    for entry in sorted_entries:
        if not entry.get("consumed", False):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    
    return longest


def calculate_progress_percentage(
    days_sober: int,
    target_days: int = 90
) -> float:
    """Calculate progress percentage"""
    if target_days <= 0:
        return 0.0
    
    percentage = (days_sober / target_days) * 100
    return min(100.0, max(0.0, percentage))


def get_recent_entries(
    entries: List[Dict[str, Any]],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get most recent entries"""
    if not entries:
        return []
    
    sorted_entries = sorted(
        entries,
        key=lambda x: x.get("date", ""),
        reverse=True
    )
    
    return sorted_entries[:limit]


def calculate_average_cravings(entries: List[Dict[str, Any]]) -> float:
    """Calculate average cravings level"""
    if not entries:
        return 0.0
    
    cravings_levels = [
        entry.get("cravings_level", 0)
        for entry in entries
        if isinstance(entry.get("cravings_level"), (int, float))
    ]
    
    if not cravings_levels:
        return 0.0
    
    return sum(cravings_levels) / len(cravings_levels)


def get_most_common_triggers(
    entries: List[Dict[str, Any]],
    limit: int = 5
) -> List[str]:
    """Get most common triggers"""
    if not entries:
        return []
    
    trigger_counts: Dict[str, int] = {}
    
    for entry in entries:
        triggers = entry.get("triggers_encountered", [])
        if isinstance(triggers, list):
            for trigger in triggers:
                if isinstance(trigger, str):
                    trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
    
    # Sort by count and return top triggers
    sorted_triggers = sorted(
        trigger_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [trigger for trigger, _ in sorted_triggers[:limit]]


def create_progress_summary(
    user_id: str,
    entries: List[Dict[str, Any]],
    target_days: int = 90
) -> Dict[str, Any]:
    """Create progress summary from entries"""
    days_sober = calculate_days_sober(entries)
    streak = calculate_streak(entries)
    longest_streak = calculate_longest_streak(entries)
    progress_percentage = calculate_progress_percentage(days_sober, target_days)
    recent_entries = get_recent_entries(entries)
    
    return {
        "user_id": user_id,
        "days_sober": days_sober,
        "total_entries": len(entries),
        "streak_days": streak,
        "longest_streak": longest_streak,
        "progress_percentage": progress_percentage,
        "recent_entries": recent_entries
    }

