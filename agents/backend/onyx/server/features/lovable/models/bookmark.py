"""Bookmark model for user favorites."""

from sqlalchemy import Column, String, DateTime, UniqueConstraint
from sqlalchemy.sql import func
from datetime import datetime

from .base import Base


class Bookmark(Base):
    """Model for user bookmarks/favorites."""
    
    __tablename__ = "bookmarks"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    chat_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Ensure one bookmark per user per chat
    __table_args__ = (
        UniqueConstraint('user_id', 'chat_id', name='unique_user_bookmark'),
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }







