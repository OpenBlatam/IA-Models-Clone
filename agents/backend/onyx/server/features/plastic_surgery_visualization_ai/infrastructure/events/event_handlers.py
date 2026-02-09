"""Event handlers for domain events."""

from domain.events import (
    VisualizationCreatedEvent,
    VisualizationRetrievedEvent,
    ComparisonCreatedEvent,
    BatchProcessingCompletedEvent
)
from core.interfaces import IMetricsCollector
from utils.logger import get_logger

logger = get_logger(__name__)


class MetricsEventHandler:
    """Event handler that updates metrics."""
    
    def __init__(self, metrics_collector: IMetricsCollector):
        self.metrics_collector = metrics_collector
    
    async def handle_visualization_created(self, event: VisualizationCreatedEvent) -> None:
        """Handle visualization created event."""
        self.metrics_collector.increment("events.visualization.created")
        self.metrics_collector.increment(f"events.visualization.{event.surgery_type.value}")
        self.metrics_collector.record_timing(
            "events.visualization.processing_time",
            event.processing_time
        )
        logger.info(f"Visualization created: {event.visualization_id}")
    
    async def handle_visualization_retrieved(self, event: VisualizationRetrievedEvent) -> None:
        """Handle visualization retrieved event."""
        self.metrics_collector.increment("events.visualization.retrieved")
        logger.debug(f"Visualization retrieved: {event.visualization_id}")
    
    async def handle_comparison_created(self, event: ComparisonCreatedEvent) -> None:
        """Handle comparison created event."""
        self.metrics_collector.increment("events.comparison.created")
        logger.info(f"Comparison created: {event.comparison_id}")
    
    async def handle_batch_completed(self, event: BatchProcessingCompletedEvent) -> None:
        """Handle batch processing completed event."""
        self.metrics_collector.increment("events.batch.completed")
        self.metrics_collector.record_timing(
            "events.batch.processing_time",
            event.processing_time
        )
        logger.info(
            f"Batch processing completed: {event.processed}/{event.total} processed, "
            f"{event.failed} failed"
        )


class LoggingEventHandler:
    """Event handler that logs events."""
    
    async def handle_visualization_created(self, event: VisualizationCreatedEvent) -> None:
        """Handle visualization created event."""
        logger.info(
            f"Visualization created: {event.visualization_id}, "
            f"type: {event.surgery_type}, intensity: {event.intensity}"
        )
    
    async def handle_visualization_retrieved(self, event: VisualizationRetrievedEvent) -> None:
        """Handle visualization retrieved event."""
        logger.debug(f"Visualization retrieved: {event.visualization_id}")
    
    async def handle_comparison_created(self, event: ComparisonCreatedEvent) -> None:
        """Handle comparison created event."""
        logger.info(
            f"Comparison created: {event.comparison_id}, "
            f"layout: {event.layout}"
        )
    
    async def handle_batch_completed(self, event: BatchProcessingCompletedEvent) -> None:
        """Handle batch processing completed event."""
        logger.info(
            f"Batch processing completed: {event.processed}/{event.total} processed, "
            f"{event.failed} failed in {event.processing_time:.2f}s"
        )

