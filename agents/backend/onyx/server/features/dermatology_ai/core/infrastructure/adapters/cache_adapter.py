from typing import Any, Optional

from ...domain.interfaces import ICacheService


class CacheAdapter(ICacheService):
    
    def __init__(self, cache_manager):
        self.cache = cache_manager
    
    async def get(self, key: str) -> Optional[Any]:
        return await self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return await self.cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        return await self.cache.delete(key)















