"""
PDF Variantes Cache Service
Cache service wrapper for main API
"""

import logging
from typing import Any, Dict, Optional
from ..utils.config import Settings
from ..utils.cache_helpers import CacheService as BaseCacheService

logger = logging.getLogger(__name__)

class CacheService:
    """Cache service for PDF Variantes API"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_service = BaseCacheService(settings)
    
    async def initialize(self):
        """Initialize cache service"""
        try:
            await self.cache_service.initialize()
            logger.info("Cache Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cache Service: {e}")
            # Don't raise - cache is optional
            pass
    
    async def cleanup(self):
        """Cleanup cache service"""
        try:
            if hasattr(self.cache_service, 'cleanup'):
                await self.cache_service.cleanup()
            elif hasattr(self.cache_service, 'close'):
                await self.cache_service.close()
        except Exception as e:
            logger.error(f"Error cleaning up Cache Service: {e}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        # Use the cache_manager directly
        if hasattr(self.cache_service, 'cache_manager'):
            return await self.cache_service.cache_manager.get(key, default)
        return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        # Use the cache_manager directly
        if hasattr(self.cache_service, 'cache_manager'):
            return await self.cache_service.cache_manager.set(key, value, ttl)
        return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        # Use the cache_manager directly
        if hasattr(self.cache_service, 'cache_manager'):
            return await self.cache_service.cache_manager.delete(key)
        return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        # Use the cache_manager directly
        if hasattr(self.cache_service, 'cache_manager'):
            return await self.cache_service.cache_manager.clear()
        return False
