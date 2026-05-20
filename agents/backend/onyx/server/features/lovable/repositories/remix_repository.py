"""
Remix repository for database operations on remixes.
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging

from .base_repository import BaseRepository
from ..models.remix import Remix

logger = logging.getLogger(__name__)


class RemixRepository(BaseRepository):
    """Repository for remix operations."""
    
    def __init__(self, db: Session):
        """Initialize remix repository."""
        super().__init__(db, Remix)
    
    def get_by_original_chat(
        self,
        original_chat_id: str,
        limit: int = 20
    ) -> List[Remix]:
        """
        Get remixes by original chat ID.
        
        Args:
            original_chat_id: Original chat ID
            limit: Maximum number of remixes to return
            
        Returns:
            List of remixes
        """
        return self.db.query(Remix).filter(
            Remix.original_chat_id == original_chat_id
        ).order_by(desc(Remix.created_at)).limit(limit).all()
    
    def get_by_remix_chat(self, remix_chat_id: str) -> Optional[Remix]:
        """
        Get remix by remix chat ID.
        
        Args:
            remix_chat_id: Remix chat ID
            
        Returns:
            Remix or None if not found
        """
        return self.db.query(Remix).filter(
            Remix.remix_chat_id == remix_chat_id
        ).first()
    
    def get_by_user_id(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Remix]:
        """
        Get remixes by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of remixes to return
            
        Returns:
            List of remixes
        """
        return self.db.query(Remix).filter(
            Remix.user_id == user_id
        ).order_by(desc(Remix.created_at)).limit(limit).all()




