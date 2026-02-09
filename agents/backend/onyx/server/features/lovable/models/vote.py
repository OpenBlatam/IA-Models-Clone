"""
Vote model for chat votes.
"""

from sqlalchemy import Column, String, DateTime, Integer, UniqueConstraint, Index
from sqlalchemy.sql import func
from datetime import datetime

from .base import Base


class Vote(Base):
    """Model for chat votes."""
    
    __tablename__ = "votes"
    
    id = Column(String, primary_key=True)
    chat_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    vote_type = Column(String(10), nullable=False)  # 'upvote' or 'downvote'
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Ensure one vote per user per chat
    __table_args__ = (
        UniqueConstraint('chat_id', 'user_id', name='unique_user_vote'),
        Index('idx_chat_vote_type', 'chat_id', 'vote_type'),
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "vote_type": self.vote_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }




