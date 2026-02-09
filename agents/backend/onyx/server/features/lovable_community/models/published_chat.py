"""
PublishedChat model

Model for chats published in the community.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, ForeignKey, DateTime, Index
from sqlalchemy.orm import relationship

from .base import Base


class PublishedChat(Base):
    """Model for chats published in the community"""
    __tablename__ = "published_chats"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    chat_content = Column(Text, nullable=False)
    tags = Column(String(500), nullable=True)
    
    vote_count = Column(Integer, default=0, nullable=False)
    remix_count = Column(Integer, default=0, nullable=False)
    view_count = Column(Integer, default=0, nullable=False)
    
    score = Column(Float, default=0.0, nullable=False, index=True)
    
    is_public = Column(Boolean, default=True, nullable=False)
    is_featured = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    original_chat_id = Column(String, ForeignKey("published_chats.id"), nullable=True)
    
    remixes = relationship("ChatRemix", foreign_keys="ChatRemix.original_chat_id", back_populates="original_chat")
    votes = relationship("ChatVote", back_populates="chat")
    views = relationship("ChatView", back_populates="chat")
    
    __table_args__ = (
        Index("idx_score_created", "score", "created_at"),
        Index("idx_user_created", "user_id", "created_at"),
        Index("idx_public_featured", "is_public", "is_featured"),
        Index("idx_tags", "tags"),
        Index("idx_original_chat", "original_chat_id"),
        Index("idx_public_score", "is_public", "score", "created_at"),
        Index("idx_user_public", "user_id", "is_public", "score"),
    )
    
    def get_tags_list(self) -> List[str]:
        """
        Get tags as a list (cached for performance).
        
        The cache is automatically invalidated when tags are updated.
        Tags are normalized (lowercased and deduplicated).
        
        Returns:
            List of normalized tags or empty list
            
        Example:
            >>> chat.tags = "python, fastapi, Python, API"
            >>> chat.get_tags_list()
            ['python', 'fastapi', 'api']
        """
        cache_key = '_tags_list_cache'
        cache_tag_value = '_tags_cache_value'
        
        # Check if cache is valid
        if not hasattr(self, cache_key) or getattr(self, cache_tag_value, None) != self.tags:
            if self.tags:
                # Parse and normalize tags
                tags_list = [
                    tag.strip().lower() 
                    for tag in self.tags.split(",") 
                    if tag.strip()
                ]
                # Remove duplicates while preserving order
                seen = set()
                unique_tags = []
                for tag in tags_list:
                    if tag not in seen:
                        seen.add(tag)
                        unique_tags.append(tag)
                tags_list = unique_tags
            else:
                tags_list = []
            
            # Update cache
            setattr(self, cache_key, tags_list)
            setattr(self, cache_tag_value, self.tags)
        
        return getattr(self, cache_key, [])
    
    def has_tag(self, tag: str) -> bool:
        """
        Check if chat has a specific tag.
        
        Args:
            tag: Tag to check (case-insensitive)
            
        Returns:
            True if tag exists, False otherwise
        """
        if not tag:
            return False
        
        tags_list = self.get_tags_list()
        return tag.strip().lower() in tags_list
    
    def add_tag(self, tag: str) -> bool:
        """
        Add a tag to the chat (if not already present).
        
        Args:
            tag: Tag to add
            
        Returns:
            True if tag was added, False if already exists
        """
        if not tag or not tag.strip():
            return False
        
        tag = tag.strip().lower()
        tags_list = self.get_tags_list()
        
        if tag in tags_list:
            return False
        
        tags_list.append(tag)
        self.tags = ",".join(tags_list)
        
        # Invalidate cache
        if hasattr(self, '_tags_list_cache'):
            delattr(self, '_tags_list_cache')
        if hasattr(self, '_tags_cache_value'):
            delattr(self, '_tags_cache_value')
        
        return True



