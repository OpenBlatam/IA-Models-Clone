"""
Published Chat model.
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Index
from sqlalchemy.sql import func
from datetime import datetime

from .base import Base


class PublishedChat(Base):
    """Model for published chats."""
    
    __tablename__ = "published_chats"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String(200), nullable=False)
    chat_content = Column(Text, nullable=False)  # Using chat_content to avoid conflict with SQLAlchemy's content
    description = Column(Text)
    tags = Column(String)  # Comma-separated tags
    category = Column(String(50))
    is_public = Column(Boolean, default=True, index=True)
    is_featured = Column(Boolean, default=False, index=True)
    vote_count = Column(Integer, default=0, index=True)
    remix_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0, index=True)
    score = Column(Float, default=0.0, index=True)
    original_chat_id = Column(String, nullable=True, index=True)  # For remixes
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_score_created', 'score', 'created_at'),
        Index('idx_user_created', 'user_id', 'created_at'),
        Index('idx_featured_score', 'is_featured', 'score'),
        Index('idx_category_score', 'category', 'score'),
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.chat_content,
            "description": self.description,
            "tags": self.tags.split(",") if self.tags else [],
            "category": self.category,
            "is_public": self.is_public,
            "is_featured": self.is_featured,
            "vote_count": self.vote_count,
            "remix_count": self.remix_count,
            "view_count": self.view_count,
            "score": self.score,
            "original_chat_id": self.original_chat_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }




