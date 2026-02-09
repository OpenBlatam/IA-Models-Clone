"""Cache repository implementation."""

from typing import Optional

from core.interfaces import ICacheRepository
from utils.cache_advanced import Cache
from utils.logger import get_logger

logger = get_logger(__name__)


class FileCacheRepository(ICacheRepository):
    """File-based cache repository."""
    
    def __init__(self, cache: Cache):
        self.cache = cache
    
    async def get(self, key: str) -> Optional[dict]:
        """Get value from cache."""
        return await self.cache.get(key)
    
    async def set(
        self,
        key: str,
        value: dict,
        ttl_hours: Optional[float] = None
    ) -> None:
        """Set value in cache."""
        await self.cache.set(key, value)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        # File cache doesn't support individual deletion easily
        # This is a limitation of the current implementation
        logger.warning("Individual cache deletion not fully supported")
        return False
    
    async def clear(self) -> None:
        """Clear all cache."""
        # File cache doesn't support clearing easily
        logger.warning("Cache clearing not fully supported")

