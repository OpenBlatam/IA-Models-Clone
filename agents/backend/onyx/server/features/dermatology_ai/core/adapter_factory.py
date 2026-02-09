"""
Adapter factory for creating infrastructure adapters
Extracted from composition_root.py for better organization
"""

from typing import Dict, Any, Optional
import logging

from .infrastructure.adapters import (
    ImageProcessorAdapter,
    CacheAdapter,
    EventPublisherAdapter,
    IDatabaseAdapter,
)
from .infrastructure.adapters.fallback_adapters import (
    FallbackDatabaseAdapter,
    NoOpCacheAdapter,
    FallbackImageProcessor,
    NoOpEventPublisher,
)

logger = logging.getLogger(__name__)


class AdapterFactory:
    """Factory for creating infrastructure adapters"""
    
    @staticmethod
    async def create_database_adapter(config: Dict[str, Any]) -> IDatabaseAdapter:
        """Create database adapter"""
        try:
            from utils.database_abstraction import get_database_adapter
        except ImportError:
            logger.warning("Database abstraction not available, using fallback")
            return FallbackDatabaseAdapter()
        
        db_type = config.get("database_type", "sqlite")
        db_config = config.get("database_config", {})
        adapter = get_database_adapter(db_type, **db_config)
        await adapter.connect()
        return adapter
    
    @staticmethod
    async def create_cache_adapter(config: Dict[str, Any]) -> CacheAdapter:
        """Create cache adapter"""
        try:
            from utils.cache import get_cache_manager
            cache_manager = get_cache_manager()
            await cache_manager.initialize()
            return CacheAdapter(cache_manager)
        except Exception as e:
            logger.warning(f"Cache not available: {e}, using no-op cache")
            return NoOpCacheAdapter()
    
    @staticmethod
    async def create_image_processor_adapter(config: Dict[str, Any]) -> ImageProcessorAdapter:
        """Create image processor adapter"""
        try:
            from services.image_processor import ImageProcessor
            processor = ImageProcessor()
            return ImageProcessorAdapter(processor)
        except Exception as e:
            logger.warning(f"Image processor not available: {e}, using fallback")
            return FallbackImageProcessor()
    
    @staticmethod
    async def create_event_publisher_adapter(config: Dict[str, Any]) -> EventPublisherAdapter:
        """Create event publisher adapter"""
        try:
            from utils.message_broker import get_message_broker
            broker_type = config.get("message_broker_type", "memory")
            broker_config = config.get("message_broker_config", {})
            broker = get_message_broker(broker_type, **broker_config)
            await broker.connect()
            return EventPublisherAdapter(broker)
        except Exception as e:
            logger.warning(f"Event publisher not available: {e}, using no-op publisher")
            return NoOpEventPublisher()







