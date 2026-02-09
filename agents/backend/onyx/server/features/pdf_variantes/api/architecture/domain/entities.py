"""
Domain Layer - Entities
Business entities with business logic
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4


@dataclass
class DocumentEntity:
    """Document domain entity"""
    id: str
    user_id: str
    filename: str
    file_path: str
    file_size: int
    content_type: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[dict] = None
    
    def is_owned_by(self, user_id: str) -> bool:
        """Check if document is owned by user"""
        return self.user_id == user_id
    
    def can_be_accessed_by(self, user_id: str) -> bool:
        """Check if document can be accessed"""
        return self.is_owned_by(user_id)
    
    def mark_as_processed(self):
        """Mark document as processed"""
        self.status = "processed"
        self.updated_at = datetime.utcnow()


@dataclass
class VariantEntity:
    """Variant domain entity"""
    id: str
    document_id: str
    variant_type: str
    content: str
    similarity_score: float
    status: str
    created_at: datetime
    metadata: Optional[dict] = None
    
    def is_valid(self) -> bool:
        """Validate variant"""
        return (
            self.document_id is not None and
            self.variant_type is not None and
            len(self.content) > 0 and
            0.0 <= self.similarity_score <= 1.0
        )


@dataclass
class TopicEntity:
    """Topic domain entity"""
    id: str
    document_id: str
    topic: str
    relevance_score: float
    category: str
    created_at: datetime
    
    def is_relevant(self, min_score: float = 0.5) -> bool:
        """Check if topic is relevant"""
        return self.relevance_score >= min_score






