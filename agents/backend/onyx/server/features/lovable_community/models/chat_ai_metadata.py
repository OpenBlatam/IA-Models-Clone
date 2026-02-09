"""
ChatAIMetadata model

Model for storing AI analysis metadata (sentiment, moderation, etc.).
"""

from datetime import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime, Index, JSON, Float, Boolean
from sqlalchemy.orm import relationship

from .base import Base


class ChatAIMetadata(Base):
    """
    Model for storing AI analysis metadata.
    
    Stores results from AI analysis including sentiment, moderation, and quality scores.
    Each chat can have only one metadata record (enforced by unique constraint).
    
    Attributes:
        id: Unique identifier for the metadata record
        chat_id: ID of the chat (unique, one metadata per chat)
        sentiment_label: Sentiment label (positive, negative, neutral)
        sentiment_score: Sentiment confidence score (0.0-1.0)
        toxicity_score: Toxicity score (0.0-1.0)
        is_toxic: Whether content is flagged as toxic
        moderation_flags: List of moderation flags (stored as JSON)
        readability_score: Readability score
        text_quality_score: Text quality score
        ai_analysis_version: Version of AI analysis used
        created_at: Timestamp when metadata was created
        updated_at: Timestamp when metadata was last updated
    """
    __tablename__ = "chat_ai_metadata"
    
    id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("published_chats.id"), nullable=False, unique=True, index=True)
    
    # Sentiment Analysis
    sentiment_label = Column(String(50), nullable=True)  # positive, negative, neutral
    sentiment_score = Column(Float, nullable=True)  # Confidence score
    
    # Content Moderation
    toxicity_score = Column(Float, nullable=True)
    is_toxic = Column(Boolean, default=False, nullable=False)
    moderation_flags = Column(JSON, nullable=True)  # List of moderation flags
    
    # Text Quality
    readability_score = Column(Float, nullable=True)
    text_quality_score = Column(Float, nullable=True)
    
    # Metadata
    ai_analysis_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    chat = relationship("PublishedChat", backref="ai_metadata")
    
    __table_args__ = (
        Index("idx_chat_ai", "chat_id"),
        Index("idx_toxicity", "is_toxic", "toxicity_score"),
        Index("idx_sentiment", "sentiment_label", "sentiment_score"),
    )
    
    def __repr__(self) -> str:
        """String representation of the metadata."""
        return f"<ChatAIMetadata(id={self.id}, chat_id={self.chat_id}, sentiment={self.sentiment_label}, is_toxic={self.is_toxic})>"
    
    def validate(self) -> list[str]:
        """
        Validate the AI metadata model.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.id or not self.id.strip():
            errors.append("Metadata ID cannot be empty")
        
        if not self.chat_id or not self.chat_id.strip():
            errors.append("Chat ID cannot be empty")
        
        # Validate sentiment_label if provided
        if self.sentiment_label is not None:
            valid_sentiments = ["positive", "negative", "neutral"]
            if self.sentiment_label.lower() not in valid_sentiments:
                errors.append(f"Sentiment label must be one of {valid_sentiments}, got '{self.sentiment_label}'")
        
        # Validate sentiment_score if provided
        if self.sentiment_score is not None:
            if not isinstance(self.sentiment_score, (int, float)):
                errors.append("Sentiment score must be a number")
            elif not (0.0 <= self.sentiment_score <= 1.0):
                errors.append(f"Sentiment score must be between 0.0 and 1.0, got {self.sentiment_score}")
        
        # Validate toxicity_score if provided
        if self.toxicity_score is not None:
            if not isinstance(self.toxicity_score, (int, float)):
                errors.append("Toxicity score must be a number")
            elif not (0.0 <= self.toxicity_score <= 1.0):
                errors.append(f"Toxicity score must be between 0.0 and 1.0, got {self.toxicity_score}")
        
        # Validate sentiment_label length
        if self.sentiment_label and len(self.sentiment_label) > 50:
            errors.append(f"Sentiment label cannot exceed 50 characters, got {len(self.sentiment_label)}")
        
        # Validate ai_analysis_version length
        if self.ai_analysis_version and len(self.ai_analysis_version) > 50:
            errors.append(f"AI analysis version cannot exceed 50 characters, got {len(self.ai_analysis_version)}")
        
        return errors
    
    def is_valid(self) -> bool:
        """
        Check if the metadata is valid.
        
        Returns:
            True if valid, False otherwise
        """
        return len(self.validate()) == 0

