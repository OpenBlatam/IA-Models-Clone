"""
Fallback adapters for when primary adapters are not available
Extracted from composition_root.py for better organization
"""

from typing import Dict, Any, Optional
import logging

from ...domain.interfaces import (
    IImageProcessor,
    IEventPublisher,
    ICacheService,
)
from .database_adapter import IDatabaseAdapter

logger = logging.getLogger(__name__)


class FallbackDatabaseAdapter(IDatabaseAdapter):
    """Fallback database adapter when database abstraction is not available"""
    
    async def connect(self) -> None:
        """No-op connection"""
        pass
    
    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        """No-op insert"""
        logger.warning("Using fallback database adapter - insert operation ignored")
        return ""
    
    async def get(self, table: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """No-op get"""
        return None
    
    async def query(
        self,
        table: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> list:
        """No-op query"""
        return []
    
    async def update(
        self,
        table: str,
        key: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> bool:
        """No-op update"""
        return False
    
    async def delete(self, table: str, key: Dict[str, Any]) -> bool:
        """No-op delete"""
        return False


class NoOpCacheAdapter(ICacheService):
    """No-op cache adapter when cache is not available"""
    
    async def get(self, key: str) -> Optional[Any]:
        """No-op get"""
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """No-op set"""
        return True
    
    async def delete(self, key: str) -> bool:
        """No-op delete"""
        return True


class FallbackImageProcessor(IImageProcessor):
    """Fallback image processor when image processor is not available"""
    
    async def process(self, image_data: bytes) -> Dict[str, Any]:
        """No-op process"""
        logger.warning("Using fallback image processor - processing ignored")
        return {}
    
    async def validate(self, image_data: bytes) -> bool:
        """No-op validate"""
        return True


class NoOpEventPublisher(IEventPublisher):
    """No-op event publisher when event publisher is not available"""
    
    async def publish(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """No-op publish"""
        return True

