"""
Processor Factory - Factory for creating and managing processor instances
========================================================================

Provides factory methods and singleton management for continuous processors.
"""

import logging
from typing import Optional

from ..config.bul_config import BULConfig
from .continuous_processor import ContinuousProcessor
from .processor_config import ProcessorConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """Factory for creating processor instances."""
    
    _global_instance: Optional[ContinuousProcessor] = None
    
    @classmethod
    def create(
        cls,
        bul_config: Optional[BULConfig] = None,
        processor_config: Optional[ProcessorConfig] = None
    ) -> ContinuousProcessor:
        """
        Create a new processor instance.
        
        Args:
            bul_config: BUL configuration
            processor_config: Processor configuration
            
        Returns:
            New ContinuousProcessor instance
        """
        return ContinuousProcessor(
            config=bul_config,
            processor_config=processor_config or DEFAULT_CONFIG
        )
    
    @classmethod
    def get_global(cls) -> ContinuousProcessor:
        """
        Get or create the global processor instance (singleton).
        
        Returns:
            Global ContinuousProcessor instance
        """
        if cls._global_instance is None:
            cls._global_instance = cls.create()
            logger.info("Global processor instance created")
        return cls._global_instance
    
    @classmethod
    def reset_global(cls) -> None:
        """Reset the global processor instance (useful for testing)."""
        cls._global_instance = None
        logger.info("Global processor instance reset")


def get_global_processor() -> ContinuousProcessor:
    """Get the global continuous processor instance."""
    return ProcessorFactory.get_global()


async def start_global_processor() -> None:
    """Start the global continuous processor."""
    processor = get_global_processor()
    await processor.start()


def stop_global_processor() -> None:
    """Stop the global continuous processor."""
    processor = get_global_processor()
    processor.stop()






