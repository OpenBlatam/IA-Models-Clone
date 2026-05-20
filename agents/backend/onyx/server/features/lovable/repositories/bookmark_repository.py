"""
Bookmark Repository for database operations.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
import logging
import uuid

from ..models.bookmark import Bookmark
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class BookmarkRepository(BaseRepository):
    """Repository for Bookmark operations."""
    
    def __init__(self, db: Session):
        """Initialize repository with database session."""
        super().__init__(db, Bookmark)
    
    def create(self, bookmark_data: Dict[str, Any]) -> Bookmark:
        """Create a new bookmark."""
        bookmark = super().create(bookmark_data)
        logger.info(f"Created bookmark {bookmark.id} for user {bookmark.user_id}")
        return bookmark
    
    def get_by_user(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[Bookmark], int]:
        """Get bookmarks for a user."""
        query = self.db.query(Bookmark).filter(Bookmark.user_id == user_id)
        total = query.count()
        
        bookmarks = query.order_by(desc(Bookmark.created_at)).offset(
            (page - 1) * page_size
        ).limit(page_size).all()
        
        return bookmarks, total
    
    def get_by_chat(self, chat_id: str, limit: int = 100) -> List[Bookmark]:
        """Get users who bookmarked a chat."""
        bookmarks = self.db.query(Bookmark).filter(
            Bookmark.chat_id == chat_id
        ).order_by(desc(Bookmark.created_at)).limit(limit).all()
        
        return bookmarks
    
    def is_bookmarked(self, user_id: str, chat_id: str) -> bool:
        """Check if user has bookmarked a chat."""
        bookmark = self.db.query(Bookmark).filter(
            Bookmark.user_id == user_id,
            Bookmark.chat_id == chat_id
        ).first()
        return bookmark is not None
    
    def delete(self, user_id: str, chat_id: str) -> bool:
        """Remove a bookmark."""
        bookmark = self.db.query(Bookmark).filter(
            Bookmark.user_id == user_id,
            Bookmark.chat_id == chat_id
        ).first()
        
        if not bookmark:
            return False
        
        try:
            self.db.delete(bookmark)
            self.db.commit()
            logger.info(f"Deleted bookmark for user {user_id}, chat {chat_id}")
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting bookmark: {e}")
            raise
    
    def get_bookmark_count(self, chat_id: str) -> int:
        """Get bookmark count for a chat."""
        return self.db.query(Bookmark).filter(Bookmark.chat_id == chat_id).count()







