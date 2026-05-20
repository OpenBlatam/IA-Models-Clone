"""
Remix Repository

Repository for ChatRemix model with specialized queries.
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..models import ChatRemix
from .base import BaseRepository
from .validation_helpers import (
    validate_string_id,
    validate_positive_integer,
    execute_with_error_handling
)


class RemixRepository(BaseRepository[ChatRemix]):
    """
    Repository for ChatRemix with specialized query methods.
    """
    
    def __init__(self, db: Session):
        super().__init__(db, ChatRemix)
    
    def get_by_original_chat_id(self, original_chat_id: str) -> List[ChatRemix]:
        """
        Get all remixes of an original chat.
        
        Args:
            original_chat_id: Original chat ID
            
        Returns:
            List of remixes
            
        Raises:
            ValueError: If original_chat_id is None or empty
        """
        original_chat_id = validate_string_id(original_chat_id, "original_chat_id")
        
        return self.db.query(ChatRemix).filter(
            ChatRemix.original_chat_id == original_chat_id
        ).order_by(desc(ChatRemix.created_at)).all()
    
    def get_by_user_id(self, user_id: str, limit: int = 100) -> List[ChatRemix]:
        """
        Get remixes by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of remixes (must be > 0)
            
        Returns:
            List of remixes
            
        Raises:
            ValueError: If user_id is None or empty, or limit is invalid
        """
        user_id = validate_string_id(user_id, "user_id")
        limit = validate_positive_integer(limit, "limit")
        
        return self.db.query(ChatRemix).filter(
            ChatRemix.user_id == user_id
        ).order_by(desc(ChatRemix.created_at)).limit(limit).all()
    
    def get_by_remix_chat_id(self, remix_chat_id: str) -> Optional[ChatRemix]:
        """
        Get remix by remix chat ID.
        
        Args:
            remix_chat_id: Remix chat ID
            
        Returns:
            Remix or None if not found
            
        Raises:
            ValueError: If remix_chat_id is None or empty
        """
        remix_chat_id = validate_string_id(remix_chat_id, "remix_chat_id")
        
        return self.db.query(ChatRemix).filter(
            ChatRemix.remix_chat_id == remix_chat_id
        ).first()
    
    def delete_by_original_chat_id(self, original_chat_id: str) -> int:
        """
        Delete all remixes for an original chat in a single query.
        
        Args:
            original_chat_id: Original chat ID
            
        Returns:
            Number of deleted remixes
            
        Raises:
            ValueError: If original_chat_id is None or empty
            DatabaseError: If deletion fails
        """
        original_chat_id = validate_string_id(original_chat_id, "original_chat_id")
        
        return execute_with_error_handling(
            self.db,
            lambda: self.db.query(ChatRemix).filter(
                ChatRemix.original_chat_id == original_chat_id
            ).delete(synchronize_session=False),
            "delete",
            "remix",
            original_chat_id
        )



