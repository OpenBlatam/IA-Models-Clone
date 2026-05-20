"""
Engagement and trending helper functions

Functions for calculating engagement rates and trending scores.
"""

from typing import Optional
from datetime import datetime
from .datetime_helpers import calculate_age_hours
from .math_helpers import ensure_non_negative, round_to_decimal_places


def calculate_engagement_rate(views: int, votes: int) -> Optional[float]:
    """
    Calculates engagement rate.
    
    Args:
        views: Number of views
        votes: Number of votes
        
    Returns:
        Engagement rate or None if cannot be calculated
    """
    if votes == 0:
        return None
    
    return round_to_decimal_places(views / votes) if votes > 0 else None


def calculate_trending_score(
    vote_count: int,
    remix_count: int,
    view_count: int,
    created_at: datetime,
    hours_ago: int = 24
) -> float:
    """
    Calculates a trending score based on recent activity.
    
    Args:
        vote_count: Number of votes
        remix_count: Number of remixes
        view_count: Number of views
        created_at: Creation date
        hours_ago: Hours back to consider "recent"
        
    Returns:
        Trending score
    """
    age_hours = calculate_age_hours(created_at)
    
    # If too old, low score
    if age_hours > hours_ago:
        return 0.0
    
    # Score based on recent engagement
    engagement = (vote_count * 2.0) + (remix_count * 3.0) + (view_count * 0.1)
    
    # Bonus for being recent
    recency_bonus = ensure_non_negative(1 - (age_hours / hours_ago))
    
    return round_to_decimal_places(engagement * (1 + recency_bonus))













