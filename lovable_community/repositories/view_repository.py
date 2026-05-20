"""
View Repository

Repository for ChatView model with specialized queries.
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from ..models import ChatView
from ..helpers.datetime_helpers import calculate_cutoff_time
from .base import BaseRepository
from .validation_helpers import (
    validate_string_id,
    validate_positive_integer,
    validate_optional_string_id,
    execute_with_error_handling
)


class ViewRepository(BaseRepository[ChatView]):
    """
    Repository for ChatView with specialized query methods.
    """
    
    def __init__(self, db: Session):
        super().__init__(db, ChatView)
    
    def get_by_chat_id(self, chat_id: str) -> List[ChatView]:
        """
        Get all views for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            List of views
            
        Raises:
            ValueError: If chat_id is None or empty
        """
        chat_id = validate_string_id(chat_id, "chat_id")
        
        return self.db.query(ChatView).filter(
            ChatView.chat_id == chat_id
        ).order_by(desc(ChatView.created_at)).all()
    
    def count_by_chat_id(self, chat_id: str) -> int:
        """
        Count views for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            Count of views
            
        Raises:
            ValueError: If chat_id is None or empty
        """
        chat_id = validate_string_id(chat_id, "chat_id")
        
        return self.db.query(ChatView).filter(
            ChatView.chat_id == chat_id
        ).count()
    
    def get_recent_views(
        self,
        chat_id: str,
        hours: int = 24
    ) -> List[ChatView]:
        """
        Get recent views for a chat.
        
        Args:
            chat_id: Chat ID
            hours: Hours to look back (must be > 0)
            
        Returns:
            List of recent views
            
        Raises:
            ValueError: If chat_id is None or empty, or hours is invalid
        """
        chat_id = validate_string_id(chat_id, "chat_id")
        hours = validate_positive_integer(hours, "hours")
        
        cutoff_time = calculate_cutoff_time(hours)
        
        return self.db.query(ChatView).filter(
            ChatView.chat_id == chat_id,
            ChatView.created_at >= cutoff_time
        ).order_by(desc(ChatView.created_at)).all()
    
    def has_user_viewed(self, chat_id: str, user_id: Optional[str]) -> bool:
        """
        Check if user has viewed a chat.
        
        Args:
            chat_id: Chat ID
            user_id: User ID (optional)
            
        Returns:
            True if user has viewed, False otherwise
            
        Raises:
            ValueError: If chat_id is None or empty, or user_id is invalid if provided
        """
        chat_id = validate_string_id(chat_id, "chat_id")
        
        user_id = validate_optional_string_id(user_id, "user_id")
        if not user_id:
            return False
        
        return self.db.query(ChatView).filter(
            ChatView.chat_id == chat_id,
            ChatView.user_id == user_id
        ).first() is not None
    
    def delete_by_chat_id(self, chat_id: str) -> int:
        """
        Delete all views for a chat in a single query.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            Number of deleted views
            
        Raises:
            ValueError: If chat_id is None or empty
            DatabaseError: If deletion fails
        """
        chat_id = validate_string_id(chat_id, "chat_id")
        
        return execute_with_error_handling(
            self.db,
            lambda: self.db.query(ChatView).filter(
                ChatView.chat_id == chat_id
            ).delete(synchronize_session=False),
            "delete",
            "view",
            chat_id
        )


