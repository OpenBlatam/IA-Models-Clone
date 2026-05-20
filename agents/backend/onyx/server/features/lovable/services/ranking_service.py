"""
Ranking Service for calculating chat scores and rankings.
"""

from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import math
import logging

from ..utils.service_base import BaseService

logger = logging.getLogger(__name__)


class RankingService(BaseService):
    """Service for ranking calculations."""
    
    def __init__(self, db: Session):
        """Initialize ranking service."""
        super().__init__(db)
    
    def calculate_score(
        self,
        vote_count: int = 0,
        remix_count: int = 0,
        view_count: int = 0,
        created_at: Optional[datetime] = None,
        upvote_count: int = 0,
        downvote_count: int = 0,
        is_featured: bool = False
    ) -> float:
        """
        Calculate ranking score for a chat.
        
        Args:
            vote_count: Total vote count
            remix_count: Remix count
            view_count: View count
            created_at: Creation timestamp
            upvote_count: Upvote count
            downvote_count: Downvote count
            is_featured: Whether chat is featured
            
        Returns:
            Ranking score
        """
        # Base score from votes (upvotes worth more than downvotes)
        if upvote_count > 0 or downvote_count > 0:
            # Use detailed vote breakdown if available
            vote_score = (upvote_count * 2.0) - (downvote_count * 1.0)
        elif vote_count > 0:
            # Fallback to total vote count if detailed breakdown not available
            vote_score = vote_count * 1.5
        else:
            vote_score = 0.0
        
        # Remix bonus (remixes indicate high quality)
        remix_bonus = remix_count * 3.0
        
        # View score (logarithmic to prevent view manipulation)
        view_score = math.log1p(view_count) * 0.5
        
        # Featured boost
        featured_boost = 50.0 if is_featured else 0.0
        
        # Time decay (newer content gets boost)
        time_decay = 1.0
        if created_at:
            hours_old = (datetime.now() - created_at).total_seconds() / 3600
            # Decay factor: 1.0 for < 1 hour, decreasing to 0.5 after 7 days
            if hours_old < 1:
                time_decay = 1.0
            elif hours_old < 24:
                time_decay = 0.9
            elif hours_old < 168:  # 7 days
                time_decay = 0.7
            else:
                time_decay = 0.5
        
        # Calculate final score
        base_score = vote_score + remix_bonus + view_score + featured_boost
        final_score = base_score * time_decay
        
        return round(final_score, 2)
    
    def calculate_engagement_rate(
        self,
        vote_count: int = 0,
        remix_count: int = 0,
        view_count: int = 0
    ) -> float:
        """
        Calculate engagement rate.
        
        Args:
            vote_count: Vote count
            remix_count: Remix count
            view_count: View count
            
        Returns:
            Engagement rate (0-100)
        """
        if view_count == 0:
            return 0.0
        
        # Weighted engagement: votes and remixes are more valuable than views
        engagement = (vote_count * 2.0) + (remix_count * 5.0)
        rate = (engagement / max(view_count, 1)) * 100.0
        
        return min(round(rate, 2), 100.0)




