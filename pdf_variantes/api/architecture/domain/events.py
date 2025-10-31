"""
Domain Layer - Domain Events
Events that represent important business occurrences
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


class DomainEvent:
    """Base domain event"""
    
    def __init__(self):
        self.event_id = str(uuid4())
        self.occurred_at = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.__class__.__name__,
            "occurred_at": self.occurred_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class DocumentUploadedEvent(DomainEvent):
    """Event raised when document is uploaded"""
    document_id: str
    user_id: str
    filename: str
    file_size: int
    
    def __post_init__(self):
        super().__init__()
        self.metadata.update({
            "document_id": self.document_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "file_size": self.file_size
        })


@dataclass
class DocumentProcessedEvent(DomainEvent):
    """Event raised when document processing completes"""
    document_id: str
    user_id: str
    processing_time: float
    success: bool
    
    def __post_init__(self):
        super().__init__()
        self.metadata.update({
            "document_id": self.document_id,
            "user_id": self.user_id,
            "processing_time": self.processing_time,
            "success": self.success
        })


@dataclass
class VariantsGeneratedEvent(DomainEvent):
    """Event raised when variants are generated"""
    document_id: str
    user_id: str
    variant_count: int
    generation_time: float
    
    def __post_init__(self):
        super().__init__()
        self.metadata.update({
            "document_id": self.document_id,
            "user_id": self.user_id,
            "variant_count": self.variant_count,
            "generation_time": self.generation_time
        })


@dataclass
class TopicsExtractedEvent(DomainEvent):
    """Event raised when topics are extracted"""
    document_id: str
    user_id: str
    topic_count: int
    extraction_time: float
    
    def __post_init__(self):
        super().__init__()
        self.metadata.update({
            "document_id": self.document_id,
            "user_id": self.user_id,
            "topic_count": self.topic_count,
            "extraction_time": self.extraction_time
        })


@dataclass
class DocumentDeletedEvent(DomainEvent):
    """Event raised when document is deleted"""
    document_id: str
    user_id: str
    
    def __post_init__(self):
        super().__init__()
        self.metadata.update({
            "document_id": self.document_id,
            "user_id": self.user_id
        })


# Event handler interface
class EventHandler(ABC):
    """Base event handler"""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle domain event"""
        pass


# Event bus interface
class EventBus(ABC):
    """Event bus for publishing and subscribing to events"""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish domain event"""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        event_type: type,
        handler: EventHandler
    ) -> None:
        """Subscribe to domain events"""
        pass






