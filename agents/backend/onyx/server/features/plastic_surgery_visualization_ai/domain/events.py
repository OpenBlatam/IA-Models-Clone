"""Domain events."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from api.schemas.visualization import SurgeryType


@dataclass
class DomainEvent:
    """Base class for domain events."""
    occurred_at: datetime
    event_type: str
    
    def __post_init__(self):
        if not self.occurred_at:
            self.occurred_at = datetime.utcnow()


@dataclass
class VisualizationCreatedEvent(DomainEvent):
    """Event raised when a visualization is created."""
    visualization_id: str
    surgery_type: SurgeryType
    intensity: float
    processing_time: float
    
    def __init__(
        self,
        visualization_id: str,
        surgery_type: SurgeryType,
        intensity: float,
        processing_time: float,
        occurred_at: Optional[datetime] = None
    ):
        super().__init__(
            occurred_at=occurred_at or datetime.utcnow(),
            event_type="visualization.created"
        )
        self.visualization_id = visualization_id
        self.surgery_type = surgery_type
        self.intensity = intensity
        self.processing_time = processing_time


@dataclass
class VisualizationRetrievedEvent(DomainEvent):
    """Event raised when a visualization is retrieved."""
    visualization_id: str
    
    def __init__(
        self,
        visualization_id: str,
        occurred_at: Optional[datetime] = None
    ):
        super().__init__(
            occurred_at=occurred_at or datetime.utcnow(),
            event_type="visualization.retrieved"
        )
        self.visualization_id = visualization_id


@dataclass
class ComparisonCreatedEvent(DomainEvent):
    """Event raised when a comparison is created."""
    comparison_id: str
    visualization_id: str
    layout: str
    
    def __init__(
        self,
        comparison_id: str,
        visualization_id: str,
        layout: str,
        occurred_at: Optional[datetime] = None
    ):
        super().__init__(
            occurred_at=occurred_at or datetime.utcnow(),
            event_type="comparison.created"
        )
        self.comparison_id = comparison_id
        self.visualization_id = visualization_id
        self.layout = layout


@dataclass
class BatchProcessingCompletedEvent(DomainEvent):
    """Event raised when batch processing is completed."""
    total: int
    processed: int
    failed: int
    processing_time: float
    
    def __init__(
        self,
        total: int,
        processed: int,
        failed: int,
        processing_time: float,
        occurred_at: Optional[datetime] = None
    ):
        super().__init__(
            occurred_at=occurred_at or datetime.utcnow(),
            event_type="batch.processing.completed"
        )
        self.total = total
        self.processed = processed
        self.failed = failed
        self.processing_time = processing_time

