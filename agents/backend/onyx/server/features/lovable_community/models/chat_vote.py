"""
ChatVote model

Model for chat votes.
"""

from datetime import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime, Index
from sqlalchemy.orm import relationship

from .base import Base


class ChatVote(Base):
    """
    Model for chat votes.
    
    Represents a user's vote (upvote or downvote) on a chat.
    Each user can only vote once per chat (enforced by unique index).
    
    Attributes:
        id: Unique identifier for the vote
        chat_id: ID of the chat being voted on
        user_id: ID of the user who voted
        vote_type: Type of vote ("upvote" or "downvote")
        created_at: Timestamp when the vote was created
    """
    __tablename__ = "chat_votes"
    
    id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("published_chats.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    vote_type = Column(String(10), default="upvote", nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    chat = relationship("PublishedChat", back_populates="votes")
    
    __table_args__ = (
        Index("idx_chat_user_vote", "chat_id", "user_id", unique=True),
    )
    
    def __repr__(self) -> str:
        """String representation of the vote."""
        return f"<ChatVote(id={self.id}, chat_id={self.chat_id}, user_id={self.user_id}, vote_type={self.vote_type})>"
    
    def validate(self) -> list[str]:
        """
        Validate the vote model.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.id or not self.id.strip():
            errors.append("Vote ID cannot be empty")
        
        if not self.chat_id or not self.chat_id.strip():
            errors.append("Chat ID cannot be empty")
        
        if not self.user_id or not self.user_id.strip():
            errors.append("User ID cannot be empty")
        
        if self.vote_type not in ("upvote", "downvote"):
            errors.append(f"Vote type must be 'upvote' or 'downvote', got '{self.vote_type}'")
        
        return errors
    
    def is_valid(self) -> bool:
        """
        Check if the vote is valid.
        
        Returns:
            True if valid, False otherwise
        """
        return len(self.validate()) == 0











