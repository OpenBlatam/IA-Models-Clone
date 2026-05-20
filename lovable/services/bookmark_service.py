"""
Bookmark Service for bookmark operations.
"""

from typing import Dict, Any, List, Tuple
from sqlalchemy.orm import Session
import logging

from ..repositories.bookmark_repository import BookmarkRepository
from ..repositories.chat_repository import ChatRepository
from ..exceptions import NotFoundError, ConflictError
from ..utils.decorators import log_execution_time, handle_errors
from ..utils.service_base import BaseService

logger = logging.getLogger(__name__)


class BookmarkService(BaseService):
    """Service for bookmark operations."""
    
    def __init__(self, db: Session):
        """Initialize bookmark service."""
        super().__init__(db)
        self.bookmark_repo = BookmarkRepository(db)
        self.chat_repo = ChatRepository(db)
    
    @log_execution_time
    @handle_errors
    def create_bookmark(self, user_id: str, chat_id: str) -> Dict[str, Any]:
        """
        Create a bookmark for a chat.
        
        Args:
            user_id: User ID creating the bookmark
            chat_id: Chat ID to bookmark
            
        Returns:
            Dictionary with bookmark data
            
        Raises:
            NotFoundError: If chat doesn't exist
            ConflictError: If bookmark already exists
        """
        # Verify chat exists
        self.get_or_raise_not_found(self.chat_repo, chat_id, "Chat")
        
        # Check if already bookmarked
        if self.bookmark_repo.is_bookmarked(user_id, chat_id):
            raise ConflictError("Chat already bookmarked", "bookmark")
        
        # Create bookmark using BaseService helper
        bookmark = self.bookmark_repo.create(
            self.create_entity_data(
                user_id=user_id,
                chat_id=chat_id
            )
        )
        
        return {"bookmark": bookmark}
    
    def delete_bookmark(self, user_id: str, chat_id: str) -> bool:
        """Remove a bookmark."""
        return self.bookmark_repo.delete(user_id, chat_id)
    
    def get_user_bookmarks(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get bookmarks for a user with chat details."""
        bookmarks, total = self.bookmark_repo.get_by_user(user_id, page=page, page_size=page_size)
        
        # Get chat details for each bookmark (optimized batch query)
        chat_ids = [bookmark.chat_id for bookmark in bookmarks]
        chats = self.batch_get_by_ids(self.chat_repo, chat_ids)
        chats_dict = {chat.id: chat for chat in chats if chat}
        
        bookmarks_data = []
        for bookmark in bookmarks:
            bookmark_dict = self.serialize_model(bookmark)
            chat = chats_dict.get(bookmark.chat_id)
            if chat:
                bookmark_dict["chat"] = self.serialize_model(chat)
            bookmarks_data.append(bookmark_dict)
        
        return bookmarks_data, total
    
    def is_bookmarked(self, user_id: str, chat_id: str) -> bool:
        """Check if a chat is bookmarked by user."""
        return self.bookmark_repo.is_bookmarked(user_id, chat_id)
    
    def get_bookmark_count(self, chat_id: str) -> int:
        """Get bookmark count for a chat."""
        return self.bookmark_repo.get_bookmark_count(chat_id)






