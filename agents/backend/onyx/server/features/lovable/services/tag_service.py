"""
Tag Service for tag-related business logic.
"""

from typing import Dict, Any, List, Tuple, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

from ..repositories.chat_repository import ChatRepository
from ..constants import TRENDING_PERIODS
from ..exceptions import NotFoundError
from ..utils.decorators import log_execution_time, handle_errors
from ..utils.service_base import BaseService

logger = logging.getLogger(__name__)


class TagService(BaseService):
    """Service for tag operations."""
    
    def __init__(self, db: Session):
        """Initialize tag service."""
        super().__init__(db)
        self.chat_repo = ChatRepository(db)
    
    @log_execution_time
    @handle_errors
    def get_popular_tags(
        self,
        limit: Optional[int] = None,
        min_usage: int = 1
    ) -> Dict[str, Any]:
        """
        Get popular tags with usage statistics.
        
        Args:
            limit: Maximum number of tags to return (uses constant if None)
            min_usage: Minimum usage count to include tag
            
        Returns:
            Dictionary with popular tags and statistics
        """
        from ..constants import DEFAULT_TAG_LIMIT
        from ..utils.statistics_helpers import calculate_percentage, group_and_count
        
        if limit is None:
            limit = DEFAULT_TAG_LIMIT
        
        # Use repository to get all public chats, then filter for tags in Python
        # This is acceptable since we need to process tags anyway
        all_chats, _ = self.chat_repo.get_all(
            page=1,
            page_size=10000,  # Large page size to get all chats
            filters={"is_public": True}
        )
        
        # Filter chats that have tags
        all_chats = [chat for chat in all_chats if chat.tags]
        
        tag_counts = self._extract_tag_counts(all_chats)
        
        # Filter by min_usage and sort
        filtered_tags = {
            tag: count for tag, count in tag_counts.items()
            if count >= min_usage
        }
        
        sorted_tags = sorted(
            filtered_tags.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        total_chats = len(all_chats)
        
        return {
            "tags": [
                {
                    "name": tag,
                    "count": count,
                    "percentage": calculate_percentage(count, total_chats) if total_chats else 0
                }
                for tag, count in sorted_tags
            ],
            "total_unique_tags": len(tag_counts),
            "total_chats_with_tags": len([c for c in all_chats if c.tags])
        }
    
    @log_execution_time
    @handle_errors
    def get_trending_tags(
        self,
        period: str = "day",
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get trending tags for a specific period.
        
        Args:
            period: Trending period ('hour', 'day', 'week', 'month')
            limit: Maximum number of tags to return (uses constant if None)
            
        Returns:
            Dictionary with trending tags and statistics
        """
        from ..constants import DEFAULT_TRENDING_TAG_LIMIT
        
        if limit is None:
            limit = DEFAULT_TRENDING_TAG_LIMIT
        
        period_hours = TRENDING_PERIODS.get(period, 24.0)
        
        cutoff = datetime.now() - timedelta(hours=period_hours)
        
        # Use repository to get recent chats with tags
        recent_chats = self.chat_repo.get_by_date_range(
            start_date=cutoff,
            is_public=True,
            limit=10000  # Large limit for trending analysis
        )
        
        # Filter chats that have tags
        recent_chats = [chat for chat in recent_chats if chat.tags]
        
        tag_counts = self._extract_tag_counts(recent_chats)
        
        sorted_tags = sorted(
            tag_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return {
            "tags": [
                {
                    "name": tag,
                    "count": count,
                    "period": period
                }
                for tag, count in sorted_tags
            ],
            "period": period,
            "total_chats": len(recent_chats)
        }
    
    @log_execution_time
    @handle_errors
    def get_tag_stats(self, tag_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific tag.
        
        Args:
            tag_name: Name of the tag
            
        Returns:
            Dictionary with tag statistics or None if tag not found
            
        Raises:
            NotFoundError: If tag doesn't exist
        """
        matching_chats = self._get_chats_with_tag(tag_name)
        
        if not matching_chats:
            return None
        
        from ..utils.statistics_helpers import (
            calculate_field_stats,
            count_by_condition
        )
        
        # Calculate statistics using helper
        stats = calculate_field_stats(
            matching_chats,
            {
                'vote_count': {'type': 'sum'},
                'remix_count': {'type': 'sum'},
                'view_count': {'type': 'sum'},
                'score': {'type': 'avg', 'round': 2}
            }
        )
        
        return {
            "tag": tag_name,
            "total_chats": len(matching_chats),
            "total_votes": stats['vote_count'],
            "total_remixes": stats['remix_count'],
            "total_views": stats['view_count'],
            "average_score": stats['score'],
            "featured_chats": count_by_condition(
                matching_chats,
                lambda c: c.is_featured
            )
        }
    
    @log_execution_time
    @handle_errors
    def get_tag_chats(
        self,
        tag_name: str,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "score"
    ) -> Tuple[List, int]:
        """
        Get chats with a specific tag.
        
        Args:
            tag_name: Name of the tag
            page: Page number (1-indexed)
            page_size: Number of items per page
            sort_by: Sort option ('score', 'created_at', 'vote_count')
            
        Returns:
            Tuple of (paginated chats, total count)
        """
        matching_chats = self._get_chats_with_tag(tag_name)
        
        if not matching_chats:
            return [], 0
        
        # Sort
        if sort_by == "score":
            matching_chats.sort(key=lambda x: x.score, reverse=True)
        elif sort_by == "created_at":
            matching_chats.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "vote_count":
            matching_chats.sort(key=lambda x: x.vote_count, reverse=True)
        
        # Paginate
        total = len(matching_chats)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_chats = matching_chats[start:end]
        
        return paginated_chats, total
    
    def _extract_tag_counts(self, chats: List[PublishedChat]) -> Dict[str, int]:
        """Extract tag counts from chats."""
        tag_counts = {}
        for chat in chats:
            if chat.tags:
                tags = [tag.strip() for tag in chat.tags.split(",") if tag.strip()]
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts
    
    def _get_chats_with_tag(self, tag_name: str, limit: Optional[int] = None) -> List:
        """
        Get all chats with a specific tag.
        
        Args:
            tag_name: Name of the tag
            limit: Maximum number of chats to return (uses constant if None)
            
        Returns:
            List of chats with the tag
        """
        from ..constants import MAX_TAG_CHATS_LIMIT
        
        if limit is None:
            limit = MAX_TAG_CHATS_LIMIT
        
        # Use repository method for tag-based queries
        return self.chat_repo.get_by_tag(tag_name, limit=limit)






