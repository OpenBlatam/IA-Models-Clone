import logging
from datetime import datetime
from typing import Optional

from ..constants import (
    DEFAULT_VOTE_WEIGHT,
    DEFAULT_REMIX_WEIGHT,
    DEFAULT_VIEW_WEIGHT,
    MIN_AGE_HOURS,
    HOURS_PER_DAY,
    DEFAULT_TRENDING_HOURS,
)
from ..helpers.datetime_helpers import calculate_age_hours, is_within_time_window
from ..helpers.math_helpers import round_score

logger = logging.getLogger(__name__)


class RankingService:
    """
    Service for calculating chat ranking scores.
    
    Implements a time-decay algorithm that considers:
    - Vote count (weighted)
    - Remix count (weighted)
    - View count (weighted)
    - Time since creation (decay factor)
    """
    
    @staticmethod
    def calculate_score(
        vote_count: int,
        remix_count: int,
        view_count: int,
        created_at: datetime,
        base_score: float = 0.0
    ) -> float:
        """
        Calculate ranking score for a chat.
        
        The score formula:
        score = (engagement_score + base_score) / time_decay
        
        Where:
        - engagement_score = vote_count * vote_weight + remix_count * remix_weight + view_count * view_weight
        - time_decay = max(1.0, 1 + (age_hours / 24))
        
        Args:
            vote_count: Number of votes (must be >= 0)
            remix_count: Number of remixes (must be >= 0)
            view_count: Number of views (must be >= 0)
            created_at: Creation timestamp
            base_score: Base score to add (default: 0.0)
            
        Returns:
            Calculated score (rounded to 2 decimal places)
            
        Raises:
            ValueError: If any count is negative
            TypeError: If created_at is not a datetime object
        """
        if vote_count < 0 or remix_count < 0 or view_count < 0:
            raise ValueError("Counts cannot be negative")
        
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime object")
        
        age_hours = calculate_age_hours(created_at)
        
        if age_hours < MIN_AGE_HOURS:
            age_hours = MIN_AGE_HOURS
        
        try:
            from ..config import settings
            vote_weight = settings.vote_weight
            remix_weight = settings.remix_weight
            view_weight = settings.view_weight
        except ImportError:
            logger.warning("Settings not available, using default weights")
            vote_weight = DEFAULT_VOTE_WEIGHT
            remix_weight = DEFAULT_REMIX_WEIGHT
            view_weight = DEFAULT_VIEW_WEIGHT
        
        engagement_score = (
            vote_count * vote_weight +
            remix_count * remix_weight +
            view_count * view_weight
        )
        
        time_decay = max(1.0, 1 + (age_hours / HOURS_PER_DAY))
        
        score = (engagement_score + base_score) / time_decay
        
        return round_score(score)
    
    @staticmethod
    def calculate_trending_score(
        vote_count: int,
        remix_count: int,
        view_count: int,
        created_at: datetime,
        hours: int = DEFAULT_TRENDING_HOURS
    ) -> float:
        """
        Calculate trending score for a chat within a time window.
        
        Similar to calculate_score but with emphasis on recent activity.
        
        Args:
            vote_count: Number of votes
            remix_count: Number of remixes
            view_count: Number of views
            created_at: Creation timestamp
            hours: Time window in hours (default: 24)
            
        Returns:
            Trending score
        """
        if not is_within_time_window(created_at, hours):
            return 0.0
        
        return RankingService.calculate_score(
            vote_count,
            remix_count,
            view_count,
            created_at
        )

