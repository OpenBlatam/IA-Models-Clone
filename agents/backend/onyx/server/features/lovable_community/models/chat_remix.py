"""
ChatRemix model

Model for chat remixes.
"""

from datetime import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime, Index
from sqlalchemy.orm import relationship

from .base import Base


class ChatRemix(Base):
    """
    Model for chat remixes.
    
    Represents a remix relationship between two chats.
    A remix is a new chat based on an original chat.
    
    Attributes:
        id: Unique identifier for the remix record
        original_chat_id: ID of the original chat
        remix_chat_id: ID of the remix chat
        user_id: ID of the user who created the remix
        created_at: Timestamp when the remix was created
    """
    __tablename__ = "chat_remixes"
    
    id = Column(String, primary_key=True, index=True)
    original_chat_id = Column(String, ForeignKey("published_chats.id"), nullable=False, index=True)
    remix_chat_id = Column(String, ForeignKey("published_chats.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    original_chat = relationship("PublishedChat", foreign_keys=[original_chat_id], back_populates="remixes")
    remix_chat = relationship("PublishedChat", foreign_keys=[remix_chat_id])
    
    __table_args__ = (
        Index("idx_original_remix", "original_chat_id", "remix_chat_id"),
        Index("idx_user_remix", "user_id", "created_at"),
    )
    
    def __repr__(self) -> str:
        """String representation of the remix."""
        return f"<ChatRemix(id={self.id}, original_chat_id={self.original_chat_id}, remix_chat_id={self.remix_chat_id}, user_id={self.user_id})>"
    
    def validate(self) -> list[str]:
        """
        Validate the remix model.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.id or not self.id.strip():
            errors.append("Remix ID cannot be empty")
        
        if not self.original_chat_id or not self.original_chat_id.strip():
            errors.append("Original chat ID cannot be empty")
        
        if not self.remix_chat_id or not self.remix_chat_id.strip():
            errors.append("Remix chat ID cannot be empty")
        
        if not self.user_id or not self.user_id.strip():
            errors.append("User ID cannot be empty")
        
        if self.original_chat_id == self.remix_chat_id:
            errors.append("Original chat ID and remix chat ID cannot be the same")
        
        return errors
    
    def is_valid(self) -> bool:
        """
        Check if the remix is valid.
        
        Returns:
            True if valid, False otherwise
        """
        return len(self.validate()) == 0











