"""
Cache Manager for Imagen Video Enhancer AI
==========================================

Intelligent caching system for processed results.
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of processed results.
    
    Features:
    - Hash-based cache keys
    - TTL (Time To Live) for cache entries
    - Automatic cache cleanup
    - Cache statistics
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, default_ttl_hours: int = 24):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
            default_ttl_hours: Default TTL in hours
        """
        self.cache_dir = cache_dir or Path("enhancer_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }
    
    def _generate_key(
        self,
        file_path: str,
        service_type: str,
        **kwargs
    ) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            file_path: Path to file
            service_type: Type of service
            **kwargs: Additional parameters
            
        Returns:
            Cache key (hash)
        """
        # Get file modification time for cache invalidation
        file_path_obj = Path(file_path)
        file_mtime = 0
        if file_path_obj.exists():
            try:
                file_mtime = file_path_obj.stat().st_mtime
            except Exception:
                pass
        
        # Create hash from parameters
        key_data = {
            "file_path": str(file_path_obj.resolve()),
            "file_mtime": file_mtime,
            "service_type": service_type,
            **kwargs
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache entry."""
        return self.cache_dir / f"{key}.json"
    
    async def get(
        self,
        file_path: str,
        service_type: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result.
        
        Args:
            file_path: Path to file
            service_type: Type of service
            **kwargs: Additional parameters
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(file_path, service_type, **kwargs)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self._stats["misses"] += 1
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # Check expiration
            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            if datetime.now() > expires_at:
                logger.debug(f"Cache expired: {key}")
                cache_path.unlink()
                self._stats["misses"] += 1
                return None
            
            # Check file modification time
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                current_mtime = file_path_obj.stat().st_mtime
                cached_mtime = datetime.fromisoformat(cache_data.get("file_mtime", "1970-01-01")).timestamp()
                if current_mtime > cached_mtime:
                    logger.debug(f"File modified, cache invalid: {key}")
                    cache_path.unlink()
                    self._stats["misses"] += 1
                    return None
            
            self._stats["hits"] += 1
            logger.debug(f"Cache hit: {key}")
            return cache_data["result"]
            
        except Exception as e:
            logger.warning(f"Error reading cache {key}: {e}")
            self._stats["misses"] += 1
            return None
    
    async def set(
        self,
        file_path: str,
        service_type: str,
        result: Dict[str, Any],
        ttl: Optional[timedelta] = None,
        **kwargs
    ):
        """
        Cache a result.
        
        Args:
            file_path: Path to file
            service_type: Type of service
            result: Result to cache
            ttl: Time to live (defaults to default_ttl)
            **kwargs: Additional parameters
        """
        key = self._generate_key(file_path, service_type, **kwargs)
        cache_path = self._get_cache_path(key)
        
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + ttl
        
        # Get file modification time
        file_path_obj = Path(file_path)
        file_mtime = datetime.now()
        if file_path_obj.exists():
            try:
                file_mtime = datetime.fromtimestamp(file_path_obj.stat().st_mtime)
            except Exception:
                pass
        
        cache_data = {
            "key": key,
            "file_path": str(file_path_obj.resolve()),
            "service_type": service_type,
            "result": result,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "file_mtime": file_mtime.isoformat(),
            **kwargs
        }
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self._stats["sets"] += 1
            logger.debug(f"Cached result: {key}")
            
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned = 0
        now = datetime.now()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                expires_at = datetime.fromisoformat(cache_data["expires_at"])
                if now > expires_at:
                    cache_file.unlink()
                    cleaned += 1
                    self._stats["evictions"] += 1
                    
            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired cache entries")
        
        return cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }
    
    async def clear(self):
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_file}: {e}")
        
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }
        
        logger.info("Cache cleared")




