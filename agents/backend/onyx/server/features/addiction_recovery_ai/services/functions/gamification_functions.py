"""
Pure functions for gamification logic
Functional programming approach - no classes
"""

from typing import Dict, Any, List
from utils.cache import cache_result


def calculate_points(
    days_sober: int,
    entries_count: int,
    milestones_achieved: int,
    coaching_sessions: int
) -> Dict[str, Any]:
    """
    Calculate total points for user
    
    Args:
        days_sober: Days sober
        entries_count: Number of log entries
        milestones_achieved: Number of milestones achieved
        coaching_sessions: Number of coaching sessions
    
    Returns:
        Dictionary with points breakdown
    """
    points_per_day = 10
    points_per_entry = 5
    points_per_milestone = 100
    points_per_session = 25
    
    total_points = (
        days_sober * points_per_day +
        entries_count * points_per_entry +
        milestones_achieved * points_per_milestone +
        coaching_sessions * points_per_session
    )
    
    level = calculate_level(total_points)
    points_to_next = calculate_points_to_next_level(total_points, level)
    
    return {
        "total_points": total_points,
        "level": level,
        "level_name": get_level_name(level),
        "points_to_next_level": points_to_next,
        "breakdown": {
            "days_sober": days_sober * points_per_day,
            "entries": entries_count * points_per_entry,
            "milestones": milestones_achieved * points_per_milestone,
            "coaching": coaching_sessions * points_per_session
        }
    }


def calculate_level(points: int) -> int:
    """Calculate level from points"""
    if points < 100:
        return 1
    if points < 500:
        return 2
    if points < 1000:
        return 3
    if points < 2500:
        return 4
    if points < 5000:
        return 5
    if points < 10000:
        return 6
    if points < 25000:
        return 7
    if points < 50000:
        return 8
    if points < 100000:
        return 9
    return 10


def calculate_points_to_next_level(points: int, current_level: int) -> int:
    """Calculate points needed for next level"""
    level_thresholds = [0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    
    if current_level >= len(level_thresholds):
        return 0
    
    next_level_threshold = level_thresholds[current_level]
    return max(0, next_level_threshold - points)


def get_level_name(level: int) -> str:
    """Get level name"""
    level_names = {
        1: "Beginner",
        2: "Explorer",
        3: "Warrior",
        4: "Champion",
        5: "Master",
        6: "Legend",
        7: "Mythic",
        8: "Transcendent",
        9: "Divine",
        10: "Immortal"
    }
    return level_names.get(level, "Unknown")


def check_achievement_eligibility(
    user_id: str,
    days_sober: int,
    current_streak: int,
    entries_count: int
) -> List[Dict[str, Any]]:
    """
    Check which achievements user is eligible for
    
    Returns:
        List of eligible achievements
    """
    achievements = []
    
    # Day-based achievements
    day_milestones = [1, 7, 30, 60, 90, 180, 365]
    for milestone in day_milestones:
        if days_sober >= milestone:
            achievements.append({
                "achievement_id": f"days_{milestone}",
                "title": f"{milestone} Days Sober",
                "description": f"Maintained sobriety for {milestone} days",
                "points": milestone * 10,
                "unlocked": True
            })
    
    # Streak achievements
    streak_milestones = [7, 14, 30, 60, 90]
    for milestone in streak_milestones:
        if current_streak >= milestone:
            achievements.append({
                "achievement_id": f"streak_{milestone}",
                "title": f"{milestone} Day Streak",
                "description": f"Maintained a {milestone}-day streak",
                "points": milestone * 5,
                "unlocked": True
            })
    
    # Entry-based achievements
    if entries_count >= 10:
        achievements.append({
            "achievement_id": "consistent_logger",
            "title": "Consistent Logger",
            "description": "Logged entries consistently",
            "points": 50,
            "unlocked": True
        })
    
    return achievements


@cache_result(ttl=300, key_prefix="leaderboard")
def calculate_leaderboard(
    users_data: List[Dict[str, Any]],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Calculate leaderboard from users data
    
    Args:
        users_data: List of user data with points
        limit: Number of top users to return
    
    Returns:
        List of leaderboard entries
    """
    if not users_data:
        return []
    
    # Sort by points descending
    sorted_users = sorted(
        users_data,
        key=lambda x: x.get("points", 0),
        reverse=True
    )
    
    leaderboard = []
    for rank, user in enumerate(sorted_users[:limit], start=1):
        leaderboard.append({
            "rank": rank,
            "user_id": user.get("user_id", ""),
            "name": user.get("name"),
            "points": user.get("points", 0),
            "level": calculate_level(user.get("points", 0)),
            "days_sober": user.get("days_sober", 0)
        })
    
    return leaderboard

