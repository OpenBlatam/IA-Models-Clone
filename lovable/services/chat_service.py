"""
Chat Service for chat-related business logic.
"""

from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

from ..repositories.chat_repository import ChatRepository
from ..repositories.remix_repository import RemixRepository
from ..constants import TRENDING_PERIODS, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from ..exceptions import NotFoundError, ValidationError
from ..utils.decorators import log_execution_time, handle_errors
from ..utils.pagination import calculate_pagination_metadata
from ..utils.service_base import BaseService

logger = logging.getLogger(__name__)


class ChatService(BaseService):
    """Service for chat operations."""
    
    def __init__(self, db: Session):
        """Initialize chat service."""
        super().__init__(db)
        self.chat_repo = ChatRepository(db)
        self.remix_repo = RemixRepository(db)
    
    @log_execution_time
    @handle_errors
    def get_chat(self, chat_id: str) -> Dict[str, Any]:
        """
        Get a chat by ID and increment view count.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            Dictionary with chat data
            
        Raises:
            NotFoundError: If chat doesn't exist
        """
        chat = self.get_or_raise_not_found(self.chat_repo, chat_id, "Chat")
        
        # Increment view count
        self.chat_repo.increment_view(chat_id)
        
        return self.serialize_model(chat)
    
    @log_execution_time
    @handle_errors
    def get_chat_stats(
        self,
        chat_id: str,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Get statistics for a chat.
        
        Args:
            chat_id: Chat ID
            detailed: Whether to include detailed statistics
            
        Returns:
            Dictionary with chat statistics
            
        Raises:
            NotFoundError: If chat doesn't exist
        """
        chat = self.get_or_raise_not_found(self.chat_repo, chat_id, "Chat")
        
        if detailed:
            stats = self.get_chat_with_stats(chat_id)
            if stats:
                return stats
            raise NotFoundError("Chat", chat_id)
        
        # Basic stats (optimize: only get top N for ranking)
        from ..constants import MAX_RANKING_CHATS_LIMIT
        top_chats = self.chat_repo.get_top_ranked(limit=MAX_RANKING_CHATS_LIMIT)
        rank = next((i + 1 for i, c in enumerate(top_chats) if c.id == chat_id), None) if top_chats else None
        
        return {
            "chat_id": chat.id,
            "vote_count": chat.vote_count,
            "remix_count": chat.remix_count,
            "view_count": chat.view_count,
            "score": chat.score,
            "rank": rank
        }
    
    @log_execution_time
    @handle_errors
    def get_chat_with_stats(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            Dictionary with detailed statistics or None if not found
        """
        chat = self.chat_repo.get_by_id(chat_id)
        
        if not chat:
            return None
        
        # Get remixes
        remixes = self.remix_repo.get_by_original_chat(chat_id, limit=100)
        
        # Calculate rank
        from ..constants import MAX_RANKING_CHATS_LIMIT
        top_chats = self.chat_repo.get_top_ranked(limit=MAX_RANKING_CHATS_LIMIT)
        rank = next((i + 1 for i, c in enumerate(top_chats) if c.id == chat_id), None)
        
        return {
            "chat_id": chat.id,
            "title": chat.title,
            "vote_count": chat.vote_count,
            "remix_count": chat.remix_count,
            "view_count": chat.view_count,
            "score": chat.score,
            "rank": rank,
            "is_featured": chat.is_featured,
            "created_at": chat.created_at.isoformat() if chat.created_at else None,
            "remixes": self.serialize_list(remixes) if remixes else []
        }
    
    @log_execution_time
    @handle_errors
    def get_chat_remixes(
        self,
        chat_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all remixes of a chat.
        
        Args:
            chat_id: Chat ID
            limit: Maximum number of remixes to return
            
        Returns:
            List of remix dictionaries
        """
        limit = self.validate_limit(limit, max_limit=MAX_PAGE_SIZE, min_limit=1)
        
        remixes = self.remix_repo.get_by_original_chat(chat_id, limit=limit)
        return self.serialize_list(remixes)
    
    @log_execution_time
    @handle_errors
    def get_top_chats(
        self,
        limit: int = 20,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top ranked chats.
        
        Args:
            limit: Number of chats to return
            category: Optional category filter
            
        Returns:
            List of top chat dictionaries
        """
        limit = self.validate_limit(limit, max_limit=MAX_PAGE_SIZE, min_limit=1)
        
        chats = self.chat_repo.get_top_ranked(limit=limit, category=category)
        return self.serialize_list(chats)
    
    @log_execution_time
    @handle_errors
    def get_trending_chats(
        self,
        period: str = "day",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get trending chats for a specific period.
        
        Args:
            period: Trending period ('hour', 'day', 'week', 'month')
            limit: Maximum number of chats to return
            
        Returns:
            List of trending chat dictionaries
        """
        if period not in TRENDING_PERIODS:
            raise ValidationError(f"Invalid period. Must be one of: {list(TRENDING_PERIODS.keys())}")
        
        if limit < 1 or limit > MAX_PAGE_SIZE:
            raise ValidationError(f"Limit must be between 1 and {MAX_PAGE_SIZE}")
        
        period_hours = TRENDING_PERIODS[period]
        chats = self.chat_repo.get_trending(period_hours=period_hours, limit=limit)
        return self.serialize_list(chats)
    
    @log_execution_time
    @handle_errors
    def get_featured_chats(
        self,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get all featured chats ordered by score.
        
        Args:
            limit: Maximum number of chats to return
            
        Returns:
            Dictionary with featured chats and metadata
        """
        limit = self.validate_limit(limit, max_limit=MAX_PAGE_SIZE, min_limit=1)
        
        chats, total = self.chat_repo.get_all(
            page=1,
            page_size=limit,
            sort_by="score",
            order="desc",
            featured=True
        )
        
        pagination = calculate_pagination_metadata(1, limit, total)
        
        return {
            "chats": self.serialize_list(chats),
            **pagination
        }
    
    @log_execution_time
    @handle_errors
    def get_user_chats(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE
    ) -> Dict[str, Any]:
        """
        Get all chats by a user.
        
        Args:
            user_id: User ID
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Dictionary with chats and pagination metadata
        """
        chats, total = self.chat_repo.get_by_user_id(user_id, page=page, page_size=page_size)
        
        pagination = calculate_pagination_metadata(page, page_size, total)
        
        return {
            "chats": self.serialize_list(chats),
            "user_id": user_id,
            **pagination
        }
    
    @log_execution_time
    @handle_errors
    def update_chat(
        self,
        chat_id: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a chat.
        
        Args:
            chat_id: Chat ID
            update_data: Dictionary with fields to update
            
        Returns:
            Updated chat dictionary
            
        Raises:
            NotFoundError: If chat doesn't exist
        """
        self.get_or_raise_not_found(self.chat_repo, chat_id, "Chat")
        
        updated_chat = self.chat_repo.update(chat_id, update_data)
        
        if not updated_chat:
            raise NotFoundError("Chat", chat_id)
        
        return self.serialize_model(updated_chat)
    
    @log_execution_time
    @handle_errors
    def delete_chat(
        self,
        chat_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Delete a chat (only by owner).
        
        Args:
            chat_id: Chat ID
            user_id: User ID (for authorization)
            
        Returns:
            Success message dictionary
            
        Raises:
            NotFoundError: If chat doesn't exist
            AuthorizationError: If user is not the owner
        """
        from ..exceptions import AuthorizationError
        
        chat = self.get_or_raise_not_found(self.chat_repo, chat_id, "Chat")
        
        # Check ownership
        if chat.user_id != user_id:
            raise AuthorizationError("Not authorized to delete this chat")
        
        success = self.chat_repo.delete(chat_id)
        
        if not success:
            raise NotFoundError("Chat", chat_id)
        
        return {"message": "Chat deleted successfully", "chat_id": chat_id}
    
    @log_execution_time
    @handle_errors
    def feature_chat(
        self,
        chat_id: str,
        featured: bool
    ) -> Dict[str, Any]:
        """
        Feature or unfeature a chat.
        
        Args:
            chat_id: Chat ID
            featured: Whether to feature the chat
            
        Returns:
            Updated chat dictionary
            
        Raises:
            NotFoundError: If chat doesn't exist
        """
        success = self.chat_repo.set_featured(chat_id, featured)
        
        if not success:
            raise NotFoundError("Chat", chat_id)
        
        # Get the updated chat
        chat = self.get_or_raise_not_found(self.chat_repo, chat_id, "Chat")
        return self.serialize_model(chat)
    
    @log_execution_time
    @handle_errors
    def list_chats(
        self,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        sort_by: str = "score",
        order: str = "desc",
        category: Optional[str] = None,
        user_id: Optional[str] = None,
        featured: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        List chats with pagination and filtering.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            sort_by: Sort field ('score', 'created_at', 'vote_count')
            order: Sort order ('asc' or 'desc')
            category: Optional category filter
            user_id: Optional user ID filter
            featured: Optional featured filter
            
        Returns:
            Dictionary with chats and pagination metadata
        """
        chats, total = self.chat_repo.get_all(
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            order=order,
            category=category,
            user_id=user_id,
            featured=featured
        )
        
        pagination = calculate_pagination_metadata(page, page_size, total)
        
        return {
            "chats": self.serialize_list(chats),
            **pagination
        }
    
    @log_execution_time
    @handle_errors
    def batch_operations(
        self,
        operation: str,
        chat_ids: List[str],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Perform batch operations on multiple chats.
        
        Args:
            operation: Operation type ('delete', 'feature', 'unfeature', 'publish', 'unpublish')
            chat_ids: List of chat IDs
            user_id: User ID (for authorization)
            
        Returns:
            Dictionary with operation results
            
        Raises:
            ValidationError: If operation is invalid
            AuthorizationError: If user is not authorized
        """
        from ..exceptions import AuthorizationError
        
        valid_operations = ["delete", "feature", "unfeature", "publish", "unpublish"]
        if operation not in valid_operations:
            raise ValidationError(f"Invalid operation. Must be one of: {valid_operations}")
        
        results = {
            "success": 0,
            "failed": 0,
            "errors": []
        }
        
        # Verify ownership for all chats
        for chat_id in chat_ids:
            chat = self.chat_repo.get_by_id(chat_id)
            if not chat:
                results["failed"] += 1
                results["errors"].append(f"Chat {chat_id} not found")
                continue
            
            if chat.user_id != user_id:
                results["failed"] += 1
                results["errors"].append(f"Not authorized for chat {chat_id}")
                continue
        
        if results["failed"] > 0:
            raise AuthorizationError(results["errors"][0])
        
        # Perform operation with transaction management
        if operation == "delete":
            def delete_operation():
                return self.chat_repo.batch_delete(chat_ids)
            deleted = self.execute_in_transaction(delete_operation)
            results["success"] = deleted
        elif operation == "feature":
            def feature_operation():
                return self.chat_repo.batch_feature(chat_ids, True)
            updated = self.execute_in_transaction(feature_operation)
            results["success"] = updated
        elif operation == "unfeature":
            def unfeature_operation():
                return self.chat_repo.batch_feature(chat_ids, False)
            updated = self.execute_in_transaction(unfeature_operation)
            results["success"] = updated
        elif operation == "publish":
            def publish_operation():
                return self.chat_repo.batch_update_public_status(chat_ids, True)
            updated = self.execute_in_transaction(publish_operation)
            results["success"] = updated
        elif operation == "unpublish":
            def unpublish_operation():
                return self.chat_repo.batch_update_public_status(chat_ids, False)
            updated = self.execute_in_transaction(unpublish_operation)
            results["success"] = updated
        
        return {
            "operation": operation,
            "total": len(chat_ids),
            **results
        }
    
    @log_execution_time
    @handle_errors
    def get_personalized_feed(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE
    ) -> Dict[str, Any]:
        """
        Get personalized feed based on followed users.
        
        Args:
            user_id: User ID
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Dictionary with personalized chats and metadata
        """
        from ..repositories.user_follow_repository import UserFollowRepository
        
        follow_repo = UserFollowRepository(self.db)
        
        # Get users that this user is following
        from ..constants import MAX_FOLLOWING_LIMIT
        following_list = follow_repo.get_following(user_id, limit=MAX_FOLLOWING_LIMIT)
        following_ids = [f.following_id for f in following_list] if following_list else []
        
        if not following_ids:
            # If not following anyone, return popular content
            chats = self.chat_repo.get_top_ranked(limit=page_size)
            total = len(chats)
        else:
            # Get chats from followed users using repository
            # Note: Repository doesn't support user_id.in_() directly, so we use get_all with filters
            # For multiple user_ids, we'll need to get all and filter
            all_chats, all_total = self.chat_repo.get_all(
                page=1,
                page_size=10000,  # Large page to get all
                sort_by="score",
                order="desc",
                is_public=True
            )
            
            # Filter by following_ids
            chats = [c for c in all_chats if c.user_id in following_ids]
            total = len(chats)
            
            # Apply pagination manually
            offset = (page - 1) * page_size
            chats = chats[offset:offset + page_size]
        
        pagination = calculate_pagination_metadata(page, page_size, total)
        
        return {
            "chats": self.serialize_list(chats),
            "user_id": user_id,
            "following_count": len(following_ids),
            **pagination
        }
    
    @log_execution_time
    @handle_errors
    def publish_chat(
        self,
        user_id: str,
        title: str,
        content: str,
        description: Optional[str] = None,
        tags: Optional[str] = None,
        category: Optional[str] = None,
        is_public: bool = True
    ) -> Dict[str, Any]:
        """
        Publish a new chat.
        
        Args:
            user_id: User ID
            title: Chat title
            content: Chat content
            description: Optional description
            tags: Optional tags (comma-separated)
            category: Optional category
            is_public: Whether chat is public
            
        Returns:
            Dictionary with chat data
            
        Raises:
            ValidationError: If validation fails
        """
        import uuid
        from datetime import datetime
        
        # Validate and sanitize inputs
        from ..utils.security import sanitize_input
        from ..utils.validators import (
            validate_title,
            validate_description,
            validate_tags,
            validate_user_id,
            validate_category
        )
        from ..constants import MAX_TITLE_LENGTH, MAX_DESCRIPTION_LENGTH
        
        # Validate user_id using BaseService helper
        user_id = self.validate_with_conversion(validate_user_id, user_id)
        
        # Validate and sanitize title
        title = self.validate_with_conversion(validate_title, title)
        title = sanitize_input(title, max_length=MAX_TITLE_LENGTH)
        
        # Validate content
        if not content or len(content.strip()) == 0:
            raise ValidationError("Content is required")
        content = sanitize_input(content)
        
        # Validate description
        if description:
            description = self.validate_with_conversion(validate_description, description)
            description = sanitize_input(description, max_length=MAX_DESCRIPTION_LENGTH)
        
        # Validate tags
        if tags:
            tags = self.validate_with_conversion(validate_tags, tags)
        
        # Validate and sanitize category
        if category:
            category = self.validate_with_conversion(validate_category, category)
            category = sanitize_input(category, max_length=50)
        
        # Create chat data using BaseService helpers
        chat_id = self.generate_id()
        
        # Convert tags list to comma-separated string for storage
        tags_str = None
        if tags:
            if isinstance(tags, list):
                tags_str = ",".join(tags)
            else:
                tags_str = tags
        
        chat_data = {
            "id": chat_id,
            "user_id": user_id,
            "title": title,
            "chat_content": content,
            "description": description if description else None,
            "tags": tags_str,
            "category": category if category else None,
            "is_public": is_public,
            "is_featured": False,
            "vote_count": 0,
            "remix_count": 0,
            "view_count": 0,
            "score": 0.0,
            "created_at": self.get_current_timestamp()
        }
        
        return chat_data






