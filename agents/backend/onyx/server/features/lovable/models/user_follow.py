"""
User Follow model for user following relationships.
"""

from sqlalchemy import Column, String, DateTime, UniqueConstraint, Index
from sqlalchemy.sql import func
from datetime import datetime

from .base import Base


class UserFollow(Base):
    """Model for user following relationships."""
    
    __tablename__ = "user_follows"
    
    id = Column(String, primary_key=True)
    follower_id = Column(String, nullable=False, index=True)
    following_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Ensure one follow relationship per pair
    __table_args__ = (
        UniqueConstraint('follower_id', 'following_id', name='unique_user_follow'),
        Index('idx_follower_created', 'follower_id', 'created_at'),
        Index('idx_following_created', 'following_id', 'created_at'),
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "follower_id": self.follower_id,
            "following_id": self.following_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }




