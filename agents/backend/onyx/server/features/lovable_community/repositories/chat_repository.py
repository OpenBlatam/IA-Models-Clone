"""
Chat Repository

Repository for PublishedChat model with specialized queries.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import or_, func, desc, asc

from ..models import PublishedChat
from .base import BaseRepository
from .optimizations import QueryOptimizer
from .query_helpers import (
    apply_ordering,
    apply_pagination,
    apply_ordering_and_pagination,
    filter_public_chats,
    execute_query_with_pagination
)
from .validation_helpers import execute_with_error_handling
from .search_helpers import build_multi_field_search_filter, build_tag_filters
from ...helpers.string_normalization import build_search_term
from ...helpers.math_helpers import safe_increment


class ChatRepository(BaseRepository[PublishedChat]):
    """
    Repository for PublishedChat with specialized query methods.
    """
    
    def __init__(self, db: Session):
        super().__init__(db, PublishedChat)
    
    def get_by_user_id(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
        is_public: Optional[bool] = None
    ) -> List[PublishedChat]:
        """
        Get chats by user ID.
        
        Args:
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records
            is_public: Filter by public status (optional)
            
        Returns:
            List of chats
        """
        query = self.db.query(PublishedChat).filter(
            PublishedChat.user_id == user_id
        )
        
        if is_public is not None:
            query = query.filter(PublishedChat.is_public == is_public)
        
        return execute_query_with_pagination(
            query, skip, limit, "created_at", "desc", PublishedChat
        )
    
    def get_public_chats(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "score",
        order: str = "desc"
    ) -> List[PublishedChat]:
        """
        Get public chats with sorting (optimized).
        
        Uses optimized query with proper indexing.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records
            sort_by: Field to sort by
            order: Sort order (asc/desc)
            
        Returns:
            List of public chats
        """
        # Use optimized batch query
        chats, _ = QueryOptimizer.get_chats_batch_optimized(
            self.db,
            skip=skip,
            limit=limit,
            sort_by=sort_by,
            order=order
        )
        return chats
    
    def get_featured_chats(self, limit: int = 10) -> List[PublishedChat]:
        """
        Get featured chats.
        
        Args:
            limit: Maximum number of featured chats
            
        Returns:
            List of featured chats
        """
        query = self.db.query(PublishedChat).filter(
            PublishedChat.is_featured == True
        )
        query = filter_public_chats(query, PublishedChat)
        query = apply_ordering(query, "score", "desc", PublishedChat)
        return apply_pagination(query, 0, limit).all()
    
    def search_by_query(
        self,
        query: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[PublishedChat]:
        """
        Search chats by query string.
        
        Args:
            query: Search query
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of matching chats
        """
        search_term = build_search_term(query)
        query_obj = self.db.query(PublishedChat)
        query_obj = filter_public_chats(query_obj, PublishedChat)
        query_obj = build_multi_field_search_filter(
            query_obj,
            search_term,
            [PublishedChat.title, PublishedChat.description, PublishedChat.tags]
        )
        return execute_query_with_pagination(
            query_obj, skip, limit, "score", "desc", PublishedChat
        )
    
    def get_by_tags(
        self,
        tags: List[str],
        skip: int = 0,
        limit: int = 100
    ) -> List[PublishedChat]:
        """
        Get chats by tags.
        
        Args:
            tags: List of tags
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of matching chats
        """
        query = self.db.query(PublishedChat)
        query = filter_public_chats(query, PublishedChat)
        query = build_tag_filters(query, tags, PublishedChat.tags)
        
        return execute_query_with_pagination(
            query, skip, limit, "score", "desc", PublishedChat
        )
    
    def get_trending(
        self,
        hours: int = 24,
        limit: int = 10
    ) -> List[PublishedChat]:
        """
        Get trending chats.
        
        Args:
            hours: Hours to look back
            limit: Maximum number of chats
            
        Returns:
            List of trending chats
        """
        from ..helpers.datetime_helpers import calculate_cutoff_time
        
        cutoff_time = calculate_cutoff_time(hours)
        
        query = self.db.query(PublishedChat).filter(
            PublishedChat.created_at >= cutoff_time
        )
        query = filter_public_chats(query, PublishedChat)
        return execute_query_with_pagination(
            query, 0, limit, "score", "desc", PublishedChat
        )
    
    def _update_chat_fields(self, chat_id: str, **updates) -> bool:
        """
        Helper method to update chat fields.
        
        Args:
            chat_id: Chat ID
            **updates: Field updates as key-value pairs
            
        Returns:
            True if updated, False if not found
            
        Raises:
            DatabaseError: If update fails
        """
        chat = self.get_by_id(chat_id)
        if not chat:
            return False
        
        def update_operation():
            for field, value in updates.items():
                if hasattr(chat, field):
                    if callable(value):
                        setattr(chat, field, value(getattr(chat, field, 0)))
                    else:
                        setattr(chat, field, value)
            return True
        
        try:
            execute_with_error_handling(
                self.db,
                update_operation,
                "update",
                "chat",
                chat_id
            )
            return True
        except DatabaseError:
            raise
    
    def increment_vote_count(self, chat_id: str, increment: int = 1) -> bool:
        """
        Increment vote count for a chat.
        
        Args:
            chat_id: Chat ID
            increment: Amount to increment (can be negative)
            
        Returns:
            True if updated, False if not found
        """
        return self._update_chat_fields(
            chat_id,
            vote_count=lambda current: safe_increment(current, increment)
        )
    
    def increment_remix_count(self, chat_id: str) -> bool:
        """
        Increment remix count for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            True if updated, False if not found
        """
        return self._update_chat_fields(
            chat_id,
            remix_count=lambda current: current + 1
        )
    
    def increment_view_count(self, chat_id: str) -> bool:
        """
        Increment view count for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            True if updated, False if not found
        """
        return self._update_chat_fields(
            chat_id,
            view_count=lambda current: current + 1
        )
    
    def update_score(self, chat_id: str, score: float) -> bool:
        """
        Update score for a chat.
        
        Args:
            chat_id: Chat ID
            score: New score
            
        Returns:
            True if updated, False if not found
        """
        return self._update_chat_fields(chat_id, score=score)
    
    def increment_view_count_and_score(
        self,
        chat_id: str,
        score: float
    ) -> bool:
        """
        Increment view count and update score in a single operation.
        
        Args:
            chat_id: Chat ID
            score: New score to set
            
        Returns:
            True if updated, False if not found
        """
        return self._update_chat_fields(
            chat_id,
            view_count=lambda current: current + 1,
            score=score
        )
    
    def increment_vote_count_and_score(
        self,
        chat_id: str,
        vote_increment: int,
        score: float
    ) -> bool:
        """
        Increment vote count and update score in a single operation.
        
        Args:
            chat_id: Chat ID
            vote_increment: Amount to increment vote count (can be negative)
            score: New score to set
            
        Returns:
            True if updated, False if not found
        """
        return self._update_chat_fields(
            chat_id,
            vote_count=lambda current: safe_increment(current, vote_increment),
            score=score
        )
    
    def increment_remix_count_and_score(
        self,
        chat_id: str,
        score: float
    ) -> bool:
        """
        Increment remix count and update score in a single operation.
        
        Args:
            chat_id: Chat ID
            score: New score to set
            
        Returns:
            True if updated, False if not found
        """
        return self._update_chat_fields(
            chat_id,
            remix_count=lambda current: current + 1,
            score=score
        )
    
    def get_rank_by_score(self, score: float) -> int:
        """
        Get rank based on score (number of chats with higher score + 1).
        
        Args:
            score: Chat score
            
        Returns:
            Rank (1-based)
        """
        from sqlalchemy import func
        query = self.db.query(func.count(PublishedChat.id)).filter(
            PublishedChat.score > score
        )
        query = filter_public_chats(query, PublishedChat)
        rank = query.scalar() + 1
        
        return rank
    
    def _build_search_query(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ):
        """
        Build base search query with filters.
        
        Args:
            query: Search query string
            tags: Filter by tags
            user_id: Filter by user ID
            
        Returns:
            SQLAlchemy query object
        """
        from sqlalchemy import or_
        
        db_query = self.db.query(PublishedChat)
        db_query = filter_public_chats(db_query, PublishedChat)
        
        if query and query.strip():
            search_term = build_search_term(query)
            db_query = build_multi_field_search_filter(
                db_query,
                search_term,
                [PublishedChat.title, PublishedChat.description, PublishedChat.tags]
            )
        elif tags:
            db_query = build_tag_filters(db_query, tags, PublishedChat.tags)
        elif user_id:
            db_query = db_query.filter(PublishedChat.user_id == user_id)
        
        return db_query
    
    def count_search_results(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> int:
        """
        Count search results for accurate pagination.
        
        Args:
            query: Search query string
            tags: Filter by tags
            user_id: Filter by user ID
            
        Returns:
            Total count of matching chats
        """
        return self._build_search_query(query, tags, user_id).count()

