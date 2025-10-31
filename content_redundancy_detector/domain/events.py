"""
Domain Events - Event-driven architecture support
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime
import uuid


@dataclass
class DomainEvent:
    """
    Base class for domain events
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    aggregate_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "aggregate_id": self.aggregate_id,
            "metadata": self.metadata
        }


@dataclass
class AnalysisCompletedEvent(DomainEvent):
    """
    Event emitted when analysis is completed
    """
    content_hash: str = ""
    redundancy_score: float = 0.0
    word_count: int = 0
    
    def __post_init__(self):
        if not self.event_type:
            self.event_type = "analysis.completed"


@dataclass
class SimilarityCompletedEvent(DomainEvent):
    """
    Event emitted when similarity check is completed
    """
    similarity_score: float = 0.0
    is_similar: bool = False
    
    def __post_init__(self):
        if not self.event_type:
            self.event_type = "similarity.completed"


@dataclass
class QualityAssessedEvent(DomainEvent):
    """
    Event emitted when quality assessment is completed
    """
    quality_score: float = 0.0
    needs_improvement: bool = False
    
    def __post_init__(self):
        if not self.event_type:
            self.event_type = "quality.assessed"


@dataclass
class BatchProcessingCompletedEvent(DomainEvent):
    """
    Event emitted when batch processing is completed
    """
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    
    def __post_init__(self):
        if not self.event_type:
            self.event_type = "batch.processing.completed"






