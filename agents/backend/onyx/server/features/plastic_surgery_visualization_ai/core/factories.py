"""Factory functions for creating service instances."""

from functools import lru_cache
from typing import Optional

from services.visualization_service import VisualizationService
from utils.cache_advanced import Cache
from core.services.ai_processor import AIProcessor
from core.services.image_processor import ImageProcessor
from core.constants import DEFAULT_CACHE_TTL_HOURS
from infrastructure.repositories import FileStorageRepository, FileCacheRepository
from infrastructure.adapters import MetricsCollectorAdapter
from infrastructure.events import SimpleEventPublisher, NullEventPublisher
from core.interfaces import (
    IImageProcessor,
    IAProcessor,
    IStorageRepository,
    ICacheRepository,
    IMetricsCollector,
    IEventPublisher
)


# Global service instances
_visualization_service: Optional[VisualizationService] = None


@lru_cache(maxsize=1)
def create_cache() -> Cache:
    """Create cache instance (singleton)."""
    return Cache(ttl_hours=DEFAULT_CACHE_TTL_HOURS)


@lru_cache(maxsize=1)
def create_image_processor() -> IImageProcessor:
    """Create image processor instance (singleton)."""
    return ImageProcessor()


@lru_cache(maxsize=1)
def create_ai_processor() -> IAProcessor:
    """Create AI processor instance (singleton)."""
    return AIProcessor()


@lru_cache(maxsize=1)
def create_storage_repository() -> IStorageRepository:
    """Create storage repository instance (singleton)."""
    return FileStorageRepository()


@lru_cache(maxsize=1)
def create_cache_repository() -> ICacheRepository:
    """Create cache repository instance (singleton)."""
    return FileCacheRepository(create_cache())


@lru_cache(maxsize=1)
def create_metrics_collector() -> IMetricsCollector:
    """Create metrics collector instance (singleton)."""
    return MetricsCollectorAdapter()


@lru_cache(maxsize=1)
def create_event_publisher() -> IEventPublisher:
    """Create event publisher instance (singleton)."""
    from config.settings import settings
    
    # Use SimpleEventPublisher if events are enabled, otherwise NullEventPublisher
    if getattr(settings, 'enable_events', True):
        publisher = SimpleEventPublisher()
        # Setup default event handlers
        _setup_event_handlers(publisher)
        return publisher
    else:
        return NullEventPublisher()


def _setup_event_handlers(publisher: SimpleEventPublisher) -> None:
    """Setup default event handlers."""
    from infrastructure.events import MetricsEventHandler, LoggingEventHandler
    from core.factories import create_metrics_collector
    
    metrics_handler = MetricsEventHandler(create_metrics_collector())
    logging_handler = LoggingEventHandler()
    
    # Subscribe handlers to events (using sync version for setup)
    publisher.subscribe_sync("visualization.created", metrics_handler.handle_visualization_created)
    publisher.subscribe_sync("visualization.created", logging_handler.handle_visualization_created)
    publisher.subscribe_sync("visualization.retrieved", metrics_handler.handle_visualization_retrieved)
    publisher.subscribe_sync("visualization.retrieved", logging_handler.handle_visualization_retrieved)
    publisher.subscribe_sync("comparison.created", metrics_handler.handle_comparison_created)
    publisher.subscribe_sync("comparison.created", logging_handler.handle_comparison_created)
    publisher.subscribe_sync("batch.processing.completed", metrics_handler.handle_batch_completed)
    publisher.subscribe_sync("batch.processing.completed", logging_handler.handle_batch_completed)


def create_visualization_service() -> VisualizationService:
    """
    Create visualization service instance (singleton).
    
    Returns:
        VisualizationService instance
    """
    global _visualization_service
    if _visualization_service is None:
        _visualization_service = VisualizationService(
            image_processor=create_image_processor(),
            ai_processor=create_ai_processor(),
            storage_repository=create_storage_repository(),
            cache_repository=create_cache_repository(),
            metrics_collector=create_metrics_collector()
        )
    return _visualization_service
