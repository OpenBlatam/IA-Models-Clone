"""
Cache Manager
=============

Advanced caching system with multiple backends.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Union
import json
import pickle
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Advanced cache manager with multiple backends.
    
    Features:
    - Memory cache
    - Redis cache
    - File cache
    - TTL management
    - Serialization
    """
    
    def __init__(self):
        self.memory_cache = {}
        self.redis_client = None
        self.file_cache_path = "./cache"
        
    async def initialize(self):
        """Initialize cache manager."""
        logger.info("Initializing Cache Manager...")
        
        try:
            # Initialize file cache directory
            import os
            os.makedirs(self.file_cache_path, exist_ok=True)
            
            logger.info("Cache Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cache Manager: {str(e)}")
            raise
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        backend: str = "memory"
    ) -> bool:
        """Set cache value."""
        try:
            if backend == "memory":
                return await self._set_memory(key, value, ttl)
            elif backend == "file":
                return await self._set_file(key, value, ttl)
            else:
                raise ValueError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {str(e)}")
            return False
    
    async def get(
        self, 
        key: str, 
        backend: str = "memory",
        default: Any = None
    ) -> Any:
        """Get cache value."""
        try:
            if backend == "memory":
                return await self._get_memory(key, default)
            elif backend == "file":
                return await self._get_file(key, default)
            else:
                raise ValueError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {str(e)}")
            return default
    
    async def delete(self, key: str, backend: str = "memory") -> bool:
        """Delete cache key."""
        try:
            if backend == "memory":
                return await self._delete_memory(key)
            elif backend == "file":
                return await self._delete_file(key)
            else:
                raise ValueError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str, backend: str = "memory") -> bool:
        """Check if key exists."""
        try:
            if backend == "memory":
                return key in self.memory_cache
            elif backend == "file":
                import os
                cache_file = f"{self.file_cache_path}/{self._hash_key(key)}.cache"
                return os.path.exists(cache_file)
            else:
                raise ValueError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {str(e)}")
            return False
    
    async def clear(self, backend: str = "memory") -> bool:
        """Clear cache."""
        try:
            if backend == "memory":
                self.memory_cache.clear()
                return True
            elif backend == "file":
                import os
                import glob
                cache_files = glob.glob(f"{self.file_cache_path}/*.cache")
                for file in cache_files:
                    os.remove(file)
                return True
            else:
                raise ValueError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False
    
    async def _set_memory(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set memory cache value."""
        try:
            cache_data = {
                'value': value,
                'expires_at': datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
            }
            
            self.memory_cache[key] = cache_data
            return True
            
        except Exception as e:
            logger.error(f"Failed to set memory cache: {str(e)}")
            return False
    
    async def _get_memory(self, key: str, default: Any = None) -> Any:
        """Get memory cache value."""
        try:
            if key not in self.memory_cache:
                return default
            
            cache_data = self.memory_cache[key]
            
            # Check expiration
            if cache_data['expires_at'] and datetime.utcnow() > cache_data['expires_at']:
                del self.memory_cache[key]
                return default
            
            return cache_data['value']
            
        except Exception as e:
            logger.error(f"Failed to get memory cache: {str(e)}")
            return default
    
    async def _delete_memory(self, key: str) -> bool:
        """Delete memory cache key."""
        try:
            if key in self.memory_cache:
                del self.memory_cache[key]
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory cache: {str(e)}")
            return False
    
    async def _set_file(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set file cache value."""
        try:
            cache_data = {
                'value': value,
                'expires_at': datetime.utcnow() + timedelta(seconds=ttl) if ttl else None,
                'created_at': datetime.utcnow()
            }
            
            cache_file = f"{self.file_cache_path}/{self._hash_key(key)}.cache"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set file cache: {str(e)}")
            return False
    
    async def _get_file(self, key: str, default: Any = None) -> Any:
        """Get file cache value."""
        try:
            cache_file = f"{self.file_cache_path}/{self._hash_key(key)}.cache"
            
            import os
            if not os.path.exists(cache_file):
                return default
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check expiration
            if cache_data['expires_at'] and datetime.utcnow() > cache_data['expires_at']:
                os.remove(cache_file)
                return default
            
            return cache_data['value']
            
        except Exception as e:
            logger.error(f"Failed to get file cache: {str(e)}")
            return default
    
    async def _delete_file(self, key: str) -> bool:
        """Delete file cache key."""
        try:
            cache_file = f"{self.file_cache_path}/{self._hash_key(key)}.cache"
            
            import os
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file cache: {str(e)}")
            return False
    
    def _hash_key(self, key: str) -> str:
        """Hash key for file storage."""
        return hashlib.md5(key.encode()).hexdigest()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            memory_count = len(self.memory_cache)
            
            import os
            import glob
            file_count = len(glob.glob(f"{self.file_cache_path}/*.cache"))
            
            return {
                'memory_cache_size': memory_count,
                'file_cache_size': file_count,
                'total_size': memory_count + file_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup cache manager."""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            logger.info("Cache Manager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Cache Manager: {str(e)}")











