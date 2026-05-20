"""
Recommendation Service for content recommendations.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

from ..repositories.chat_repository import ChatRepository
from ..utils.service_base import BaseService
from ..utils.validators import validate_user_id, validate_chat_id

logger = logging.getLogger(__name__)


class RecommendationService(BaseService):
    """Service for content recommendations."""
    
    def __init__(self, db: Session):
        """Initialize recommendation service."""
        super().__init__(db)
        self.chat_repo = ChatRepository(db)
    
    def get_recommendations(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
        strategy: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Get content recommendations.
        
        Args:
            user_id: Optional user ID for personalized recommendations
            limit: Number of recommendations
            strategy: Recommendation strategy ('popular', 'trending', 'similar', 'hybrid', 'recent', 'high_engagement')
            
        Returns:
            List of recommended chats
        """
        from ..utils.validators import validate_user_id
        from ..exceptions import ValidationError
        
        # Validate inputs using BaseService methods where possible
        limit = self.validate_limit(limit, max_limit=100, min_limit=1)
        if user_id:
            try:
                user_id = validate_user_id(user_id)
            except ValueError as e:
                raise ValidationError(str(e))
        
        valid_strategies = ["popular", "trending", "similar", "hybrid", "recent", "high_engagement"]
        if strategy not in valid_strategies:
            raise ValidationError(f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}")
        
        if strategy == "popular":
            return self._get_popular(limit)
        elif strategy == "trending":
            return self._get_trending(limit)
        elif strategy == "similar" and user_id:
            return self._get_similar(user_id, limit)
        elif strategy == "hybrid":
            return self._get_hybrid(user_id, limit)
        elif strategy == "recent":
            return self._get_recent(limit)
        elif strategy == "high_engagement":
            return self._get_high_engagement(limit)
        else:
            # Default to popular
            return self._get_popular(limit)
    
    def get_related_chats(
        self,
        chat_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get chats related to a specific chat.
        
        Args:
            chat_id: Chat ID
            limit: Number of related chats
            
        Returns:
            List of related chats
        """
        from ..exceptions import ValidationError
        
        # Validate inputs using BaseService methods
        limit = self.validate_limit(limit, max_limit=50, min_limit=1)
        chat_id = self.validate_with_conversion(validate_chat_id, chat_id)
        
        # Get the chat
        chat = self.chat_repo.get_by_id(chat_id)
        
        if not chat:
            return []
        
        # Find related chats by category
        filters = {
            "is_public": True
        }
        if chat.category:
            filters["category"] = chat.category
        
        # Get related chats using repository
        related, _ = self.chat_repo.get_all(
            page=1,
            page_size=limit,
            sort_by="score",
            order="desc",
            filters=filters
        )
        
        # Filter out the original chat
        related = [c for c in related if c.id != chat_id]
        return self.serialize_list(related[:limit])
    
    def _get_popular(self, limit: int) -> List[Dict[str, Any]]:
        """Get popular chats."""
        chats, _ = self.chat_repo.get_all(
            page=1,
            page_size=limit,
            sort_by="score",
            order="desc",
            is_public=True
        )
        return self.serialize_list(chats)
    
    def _get_trending(self, limit: int) -> List[Dict[str, Any]]:
        """Get trending chats."""
        # Use repository's get_trending method
        chats = self.chat_repo.get_trending(period_hours=24, limit=limit)
        return self.serialize_list(chats)
    
    def _get_similar(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get similar chats based on user activity."""
        # Get user's previous chats to find similar content
        user_chats, _ = self.chat_repo.get_all(
            page=1,
            page_size=10,
            user_id=user_id
        )
        
        if not user_chats:
            return self._get_popular(limit)
        
        # Get chats with similar categories
        categories = [chat.category for chat in user_chats if chat.category]
        
        if categories:
            # Use repository with category filter
            chats, _ = self.chat_repo.get_all(
                page=1,
                page_size=limit,
                sort_by="score",
                order="desc",
                category=categories[0] if categories else None,
                is_public=True
            )
            # Filter out user's own chats
            chats = [c for c in chats if c.user_id != user_id]
            return self.serialize_list(chats[:limit])
        
        return self._get_popular(limit)
    
    def _get_hybrid(self, user_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Get hybrid recommendations (mix of strategies)."""
        # Mix of trending and popular
        trending_limit = limit // 2
        popular_limit = limit - trending_limit
        
        trending = self._get_trending(trending_limit)
        popular = self._get_popular(popular_limit)
        
        # Combine and deduplicate
        seen_ids = set()
        result = []
        
        for chat in trending + popular:
            if chat["id"] not in seen_ids:
                result.append(chat)
                seen_ids.add(chat["id"])
                if len(result) >= limit:
                    break
        
        return result
    
    def _get_recent(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent chats."""
        chats, _ = self.chat_repo.get_all(
            page=1,
            page_size=limit,
            sort_by="created_at",
            order="desc",
            is_public=True
        )
        return self.serialize_list(chats)
    
    def _get_high_engagement(self, limit: int) -> List[Dict[str, Any]]:
        """Get chats with high engagement rates."""
        # Get all public chats with views
        chats, _ = self.chat_repo.get_all(
            page=1,
            page_size=1000,  # Get more to calculate engagement
            sort_by="view_count",
            order="desc",
            is_public=True
        )
        
        # Filter chats with views and calculate engagement
        chats_with_views = [c for c in chats if c.view_count and c.view_count > 0]
        
        # Calculate engagement for each chat
        engagement_scores = []
        for chat in chats_with_views:
            engagement = ((chat.vote_count or 0) + (chat.remix_count or 0) * 2) / max(chat.view_count, 1)
            engagement_scores.append((engagement, chat))
        
        # Sort by engagement and return top N
        engagement_scores.sort(key=lambda x: x[0], reverse=True)
        chats_only = [chat for _, chat in engagement_scores[:limit]]
        return self.serialize_list(chats_only)




