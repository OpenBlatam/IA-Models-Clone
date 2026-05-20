"""
Chat repository for database operations on chats.
"""

from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_
from datetime import datetime, timedelta
import logging

from .base_repository import BaseRepository
from ..models.published_chat import PublishedChat

logger = logging.getLogger(__name__)


class ChatRepository(BaseRepository):
    """Repository for chat operations."""
    
    def __init__(self, db: Session):
        """Initialize chat repository."""
        super().__init__(db, PublishedChat)
    
    def get_top_ranked(
        self,
        limit: int = 20,
        category: Optional[str] = None
    ) -> List[PublishedChat]:
        """
        Get top ranked chats.
        
        Args:
            limit: Maximum number of chats to return
            category: Optional category filter
            
        Returns:
            List of top ranked chats
        """
        query = self.db.query(PublishedChat).filter(
            PublishedChat.is_public == True
        )
        
        if category:
            query = query.filter(PublishedChat.category == category)
        
        return query.order_by(desc(PublishedChat.score), desc(PublishedChat.created_at)).limit(limit).all()
    
    def get_trending(
        self,
        period_hours: int = 24,
        limit: int = 20
    ) -> List[PublishedChat]:
        """
        Get trending chats for a period.
        
        Args:
            period_hours: Number of hours to look back
            limit: Maximum number of chats to return
            
        Returns:
            List of trending chats
        """
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        
        return self.db.query(PublishedChat).filter(
            and_(
                PublishedChat.is_public == True,
                PublishedChat.created_at >= cutoff_time
            )
        ).order_by(desc(PublishedChat.score), desc(PublishedChat.view_count)).limit(limit).all()
    
    def get_by_user_id(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[PublishedChat], int]:
        """
        Get chats by user ID with pagination.
        
        Args:
            user_id: User ID
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Tuple of (chats, total count)
        """
        query = self.db.query(PublishedChat).filter(
            PublishedChat.user_id == user_id
        )
        
        total = query.count()
        
        offset = (page - 1) * page_size
        chats = query.order_by(desc(PublishedChat.created_at)).offset(offset).limit(page_size).all()
        
        return chats, total
    
    def search_chats(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sort_by: str = "relevance",
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[PublishedChat], int]:
        """
        Search chats with advanced filtering and sorting.
        
        Args:
            query: Text search query (searches title, content, description)
            tags: List of tags to filter by
            category: Category filter
            start_date: Start date for date range filter
            end_date: End date for date range filter
            sort_by: Sort strategy ('relevance', 'score', 'created_at', 'trending')
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Tuple of (chats, total count)
        """
        base_query = self.db.query(PublishedChat).filter(
            PublishedChat.is_public == True
        )
        
        # Apply category filter
        if category:
            base_query = base_query.filter(PublishedChat.category == category)
        
        # Apply date range filter
        if start_date:
            base_query = base_query.filter(PublishedChat.created_at >= start_date)
        if end_date:
            base_query = base_query.filter(PublishedChat.created_at <= end_date)
        
        # Apply tag filters
        if tags:
            tag_filters = []
            for tag in tags:
                tag_filters.append(PublishedChat.tags.like(f"%{tag}%"))
            if tag_filters:
                base_query = base_query.filter(or_(*tag_filters))
        
        # Apply text search
        if query and query.strip():
            query_lower = query.lower().strip()
            search_filters = [
                PublishedChat.title.ilike(f"%{query_lower}%"),
                PublishedChat.chat_content.ilike(f"%{query_lower}%"),
                PublishedChat.description.ilike(f"%{query_lower}%")
            ]
            base_query = base_query.filter(or_(*search_filters))
        
        # Apply sorting
        if sort_by == "relevance":
            base_query = base_query.order_by(
                desc(PublishedChat.score),
                desc(PublishedChat.created_at)
            )
        elif sort_by == "score":
            base_query = base_query.order_by(desc(PublishedChat.score))
        elif sort_by == "created_at":
            base_query = base_query.order_by(desc(PublishedChat.created_at))
        elif sort_by == "trending":
            # Trending: recent + high engagement
            cutoff_time = datetime.now() - timedelta(hours=24)
            base_query = base_query.filter(
                PublishedChat.created_at >= cutoff_time
            ).order_by(
                desc(PublishedChat.score),
                desc(PublishedChat.view_count)
            )
        else:
            # Default to relevance
            base_query = base_query.order_by(
                desc(PublishedChat.score),
                desc(PublishedChat.created_at)
            )
        
        # Get total count
        total = base_query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        chats = base_query.offset(offset).limit(page_size).all()
        
        return chats, total
    
    def get_by_tag(
        self,
        tag_name: str,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None
    ) -> List[PublishedChat]:
        """
        Get chats with a specific tag.
        
        Args:
            tag_name: Name of the tag
            limit: Maximum number of chats to return
            start_date: Optional start date filter for trending tags
            
        Returns:
            List of chats with the tag
        """
        tag_lower = tag_name.lower()
        tag_pattern = f"%{tag_lower}%"
        
        query = self.db.query(PublishedChat).filter(
            PublishedChat.is_public == True,
            PublishedChat.tags.isnot(None),
            PublishedChat.tags.ilike(tag_pattern)
        )
        
        if start_date:
            query = query.filter(PublishedChat.created_at >= start_date)
        
        if limit:
            query = query.limit(limit)
        
        matching_chats = query.all()
        
        # Filter exact matches in Python (more precise)
        exact_matches = []
        for chat in matching_chats:
            if chat.tags:
                tags = [tag.strip().lower() for tag in chat.tags.split(",") if tag.strip()]
                if tag_lower in tags:
                    exact_matches.append(chat)
        
        return exact_matches
    
    def get_by_date_range(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        is_public: bool = True,
        limit: Optional[int] = None
    ) -> List[PublishedChat]:
        """
        Get chats within a date range.
        
        Args:
            start_date: Start date for range
            end_date: End date for range
            is_public: Filter by public status
            limit: Maximum number of chats to return
            
        Returns:
            List of chats in the date range
        """
        query = self.db.query(PublishedChat)
        
        if is_public is not None:
            query = query.filter(PublishedChat.is_public == is_public)
        
        if start_date:
            query = query.filter(PublishedChat.created_at >= start_date)
        
        if end_date:
            query = query.filter(PublishedChat.created_at <= end_date)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def increment_view(self, chat_id: str) -> bool:
        """
        Increment view count for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            True if updated, False if not found
        """
        try:
            chat = self.get_by_id(chat_id)
            if not chat:
                return False
            
            chat.view_count = (chat.view_count or 0) + 1
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error incrementing view count: {e}")
            raise
    
    def increment_remix_count(self, chat_id: str) -> bool:
        """
        Increment remix count for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            True if updated, False if not found
        """
        try:
            chat = self.get_by_id(chat_id)
            if not chat:
                return False
            
            chat.remix_count = (chat.remix_count or 0) + 1
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error incrementing remix count: {e}")
            raise
    
    def increment_vote(self, chat_id: str) -> bool:
        """
        Increment vote count for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            True if updated, False if not found
        """
        try:
            chat = self.get_by_id(chat_id)
            if not chat:
                return False
            
            chat.vote_count = (chat.vote_count or 0) + 1
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error incrementing vote count: {e}")
            raise
    
    def decrement_vote(self, chat_id: str) -> bool:
        """
        Decrement vote count for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            True if updated, False if not found
        """
        try:
            chat = self.get_by_id(chat_id)
            if not chat:
                return False
            
            chat.vote_count = max((chat.vote_count or 0) - 1, 0)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error decrementing vote count: {e}")
            raise
    
    def set_featured(self, chat_id: str, featured: bool) -> bool:
        """
        Set featured status for a chat.
        
        Args:
            chat_id: Chat ID
            featured: Whether to feature the chat
            
        Returns:
            True if updated, False if not found
        """
        try:
            chat = self.get_by_id(chat_id)
            if not chat:
                return False
            
            chat.is_featured = featured
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error setting featured status: {e}")
            raise
    
    def batch_delete(self, chat_ids: List[str]) -> int:
        """
        Delete multiple chats.
        
        Args:
            chat_ids: List of chat IDs
            
        Returns:
            Number of deleted chats
        """
        try:
            deleted = self.db.query(PublishedChat).filter(
                PublishedChat.id.in_(chat_ids)
            ).delete(synchronize_session=False)
            self.db.commit()
            return deleted
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in batch delete: {e}")
            raise
    
    def batch_feature(self, chat_ids: List[str], featured: bool) -> int:
        """
        Set featured status for multiple chats.
        
        Args:
            chat_ids: List of chat IDs
            featured: Whether to feature the chats
            
        Returns:
            Number of updated chats
        """
        try:
            updated = self.db.query(PublishedChat).filter(
                PublishedChat.id.in_(chat_ids)
            ).update({"is_featured": featured}, synchronize_session=False)
            self.db.commit()
            return updated
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in batch feature: {e}")
            raise
    
    def batch_update_public_status(
        self,
        chat_ids: List[str],
        is_public: bool
    ) -> int:
        """
        Update public status for multiple chats.
        
        Args:
            chat_ids: List of chat IDs
            is_public: Whether chats should be public
            
        Returns:
            Number of updated chats
        """
        try:
            updated = self.db.query(PublishedChat).filter(
                PublishedChat.id.in_(chat_ids)
            ).update({"is_public": is_public}, synchronize_session=False)
            self.db.commit()
            return updated
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in batch update public status: {e}")
            raise
    
    def _get_sort_mapping(self) -> dict:
        """Get sort field mapping."""
        return {
            "score": PublishedChat.score,
            "created_at": PublishedChat.created_at,
            "vote_count": PublishedChat.vote_count,
            "view_count": PublishedChat.view_count,
            "remix_count": PublishedChat.remix_count,
        }
    
    def get_all(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: Optional[str] = None,
        order: str = "desc",
        filters: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        user_id: Optional[str] = None,
        featured: Optional[bool] = None
    ) -> tuple[List[PublishedChat], int]:
        """
        Get all chats with pagination and filtering.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            sort_by: Field to sort by
            order: Sort order ('asc' or 'desc')
            filters: Dictionary of filter conditions
            category: Optional category filter
            user_id: Optional user ID filter
            featured: Optional featured filter
            
        Returns:
            Tuple of (chats, total count)
        """
        query = self.db.query(PublishedChat)
        
        # Apply specific filters
        if category:
            query = query.filter(PublishedChat.category == category)
        
        if user_id:
            query = query.filter(PublishedChat.user_id == user_id)
        
        if featured is not None:
            query = query.filter(PublishedChat.is_featured == featured)
        
        # Apply additional filters from dict
        if filters:
            from ..utils.service_helpers import apply_common_filters
            query = apply_common_filters(query, filters, PublishedChat)
        
        # Apply sorting
        if sort_by:
            sort_mapping = self._get_sort_mapping()
            from ..utils.query_helpers import apply_sorting
            query = apply_sorting(query, sort_by, order, sort_mapping)
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        from ..utils.query_helpers import apply_pagination
        paginated_query, _ = apply_pagination(query, page, page_size)
        
        # Execute query with performance tracking
        import time
        start_time = time.time()
        from ..utils.query_helpers import safe_query_execute
        chats = safe_query_execute(paginated_query, "Failed to fetch chats")
        duration = time.time() - start_time
        
        # Record query metrics
        try:
            from ..utils.performance_metrics import get_metrics
            get_metrics().record_query(
                query_type=f"{PublishedChat.__name__}.get_all",
                duration=duration,
                success=True
            )
        except Exception:
            pass
        
        return chats, total




