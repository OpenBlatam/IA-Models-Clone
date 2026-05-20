"""
L3 Cache Layer - Disk cache
"""

import logging
import os
import tempfile
from typing import Optional, Any, Dict
from ..constants import L3_DEFAULT_SIZE_LIMIT

try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

logger = logging.getLogger(__name__)


class L3Cache:
    """L3 disk cache layer"""
    
    def __init__(self, cache_dir: Optional[str] = None, size_limit: int = L3_DEFAULT_SIZE_LIMIT):
        """Initialize L3 cache
        
        Args:
            cache_dir: Cache directory path (defaults to temp directory)
            size_limit: Maximum cache size in bytes
        """
        self.enabled = DISKCACHE_AVAILABLE
        self.cache: Optional[diskcache.Cache] = None
        
        if self.enabled:
            try:
                if cache_dir is None:
                    cache_dir = os.path.join(tempfile.gettempdir(), "multi_model_cache")
                os.makedirs(cache_dir, exist_ok=True)
                self.cache = diskcache.Cache(cache_dir, size_limit=size_limit)
                logger.info(f"L3 disk cache enabled at {cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize L3 disk cache: {e}")
                self.enabled = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.enabled or not self.cache:
            return None
        
        try:
            return self.cache.get(key)
        except Exception as e:
            logger.debug(f"L3 cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in disk cache
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.cache:
            return False
        
        try:
            self.cache.set(key, value, expire=expire)
            return True
        except Exception as e:
            logger.debug(f"L3 cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from disk cache
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled or not self.cache:
            return False
        
        try:
            return self.cache.delete(key)
        except Exception as e:
            logger.debug(f"L3 cache delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all entries
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.cache:
            return False
        
        try:
            self.cache.clear()
            return True
        except Exception as e:
            logger.debug(f"L3 cache clear error: {e}")
            return False
    
    def close(self):
        """Close disk cache"""
        if self.enabled and self.cache:
            try:
                self.cache.close()
            except Exception as e:
                logger.warning(f"Error closing L3 cache: {e}")

