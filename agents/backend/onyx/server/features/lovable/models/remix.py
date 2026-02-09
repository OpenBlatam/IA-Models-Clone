"""
Remix model for chat remixes.
"""

from sqlalchemy import Column, String, DateTime, ForeignKey, Index
from sqlalchemy.sql import func
from datetime import datetime

from .base import Base


class Remix(Base):
    """Model for chat remixes."""
    
    __tablename__ = "remixes"
    
    id = Column(String, primary_key=True)
    original_chat_id = Column(String, ForeignKey("published_chats.id"), nullable=False, index=True)
    remix_chat_id = Column(String, ForeignKey("published_chats.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Composite index for common queries
    __table_args__ = (
        Index('idx_original_created', 'original_chat_id', 'created_at'),
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "original_chat_id": self.original_chat_id,
            "remix_chat_id": self.remix_chat_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }




