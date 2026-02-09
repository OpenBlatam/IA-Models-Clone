"""
Chat Score Management

Handles score calculation and updates for chats.
"""

from typing import Optional
from ....models import PublishedChat
from ...ranking import RankingService
from ...helpers.math_helpers import ensure_non_negative
from ...helpers.value_helpers import get_attr_or_default


class ScoreManager:
    """Manages chat score calculations and updates."""
    
    def __init__(self, ranking_service: RankingService):
        """
        Initialize score manager.
        
        Args:
            ranking_service: Ranking service for score calculations
        """
        self.ranking_service = ranking_service
    
    def calculate_score(
        self,
        chat: PublishedChat,
        vote_count: Optional[int] = None,
        remix_count: Optional[int] = None,
        view_count: Optional[int] = None
    ) -> float:
        """
        Calculate chat score using ranking service.
        
        Args:
            chat: Chat object
            vote_count: Optional new vote count (uses chat.vote_count if None)
            remix_count: Optional new remix count (uses chat.remix_count if None)
            view_count: Optional new view count (uses chat.view_count if None)
            
        Returns:
            Calculated score (float)
            
        Raises:
            ValueError: If chat is None or ranking_service is None
        """
        if not chat:
            raise ValueError("chat cannot be None")
        
        if not self.ranking_service:
            raise ValueError("ranking_service cannot be None")
        
        vote_count = get_attr_or_default(vote_count, lambda: chat.vote_count, 0)
        remix_count = get_attr_or_default(remix_count, lambda: chat.remix_count, 0)
        view_count = get_attr_or_default(view_count, lambda: chat.view_count, 0)
        
        # Ensure non-negative counts
        vote_count = ensure_non_negative(vote_count)
        remix_count = ensure_non_negative(remix_count)
        view_count = ensure_non_negative(view_count)
        
        return self.ranking_service.calculate_score(
            vote_count,
            remix_count,
            view_count,
            chat.created_at
        )






