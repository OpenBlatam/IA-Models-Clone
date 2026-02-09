"""
ChatView model

Model for chat views/visualizations.
"""

from datetime import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime, Index
from sqlalchemy.orm import relationship

from .base import Base


class ChatView(Base):
    """
    Model for chat views.
    
    Represents a view/visualization of a chat.
    User ID is optional to track anonymous views.
    
    Attributes:
        id: Unique identifier for the view record
        chat_id: ID of the chat being viewed
        user_id: ID of the user who viewed (optional for anonymous views)
        created_at: Timestamp when the view occurred
    """
    __tablename__ = "chat_views"
    
    id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("published_chats.id"), nullable=False, index=True)
    user_id = Column(String, nullable=True, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    chat = relationship("PublishedChat", back_populates="views")
    
    __table_args__ = (
        Index("idx_chat_created", "chat_id", "created_at"),
    )
    
    def __repr__(self) -> str:
        """String representation of the view."""
        return f"<ChatView(id={self.id}, chat_id={self.chat_id}, user_id={self.user_id})>"
    
    def validate(self) -> list[str]:
        """
        Validate the view model.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.id or not self.id.strip():
            errors.append("View ID cannot be empty")
        
        if not self.chat_id or not self.chat_id.strip():
            errors.append("Chat ID cannot be empty")
        
        # user_id is optional, but if provided, should not be empty
        if self.user_id is not None and (not isinstance(self.user_id, str) or not self.user_id.strip()):
            errors.append("User ID must be a non-empty string if provided")
        
        return errors
    
    def is_valid(self) -> bool:
        """
        Check if the view is valid.
        
        Returns:
            True if valid, False otherwise
        """
        return len(self.validate()) == 0











