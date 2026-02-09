"""
ChatEmbedding model

Model for storing chat embeddings for semantic search.
"""

from datetime import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime, Index, JSON
from sqlalchemy.orm import relationship

from .base import Base


class ChatEmbedding(Base):
    """
    Model for storing chat embeddings for semantic search.
    
    Stores vector embeddings generated from chat content for semantic similarity search.
    Each chat can have only one embedding (enforced by unique constraint).
    
    Attributes:
        id: Unique identifier for the embedding record
        chat_id: ID of the chat (unique, one embedding per chat)
        embedding: Embedding vector stored as JSON array
        embedding_model: Name/identifier of the model used to generate the embedding
        created_at: Timestamp when the embedding was created
        updated_at: Timestamp when the embedding was last updated
    """
    __tablename__ = "chat_embeddings"
    
    id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("published_chats.id"), nullable=False, unique=True, index=True)
    embedding = Column(JSON, nullable=False)  # Stores embedding vector as JSON
    embedding_model = Column(String(200), nullable=False)  # Model used to generate embedding
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    chat = relationship("PublishedChat", backref="embedding")
    
    __table_args__ = (
        Index("idx_chat_embedding", "chat_id"),
    )
    
    def __repr__(self) -> str:
        """String representation of the embedding."""
        embedding_size = len(self.embedding) if isinstance(self.embedding, list) else 0
        return f"<ChatEmbedding(id={self.id}, chat_id={self.chat_id}, model={self.embedding_model}, size={embedding_size})>"
    
    def validate(self) -> list[str]:
        """
        Validate the embedding model.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.id or not self.id.strip():
            errors.append("Embedding ID cannot be empty")
        
        if not self.chat_id or not self.chat_id.strip():
            errors.append("Chat ID cannot be empty")
        
        if not self.embedding:
            errors.append("Embedding vector cannot be empty")
        elif not isinstance(self.embedding, list):
            errors.append("Embedding must be a list/array")
        elif len(self.embedding) == 0:
            errors.append("Embedding vector cannot be empty")
        
        if not self.embedding_model or not self.embedding_model.strip():
            errors.append("Embedding model name cannot be empty")
        
        if len(self.embedding_model) > 200:
            errors.append(f"Embedding model name cannot exceed 200 characters, got {len(self.embedding_model)}")
        
        return errors
    
    def is_valid(self) -> bool:
        """
        Check if the embedding is valid.
        
        Returns:
            True if valid, False otherwise
        """
        return len(self.validate()) == 0
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vector.
        
        Returns:
            Dimension of the embedding vector, or 0 if invalid
        """
        if isinstance(self.embedding, list) and len(self.embedding) > 0:
            return len(self.embedding)
        return 0











