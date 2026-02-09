"""
Cache Service for Social Video Transcriber AI
Implements caching for transcriptions to avoid re-processing
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class CacheService:
    """Service for caching transcription results"""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.temp_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=24)  # Cache for 24 hours
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
    
    def _generate_cache_key(self, url: str, options: Optional[Dict] = None) -> str:
        """Generate a unique cache key for a URL and options"""
        key_data = {
            "url": url,
            "options": options or {},
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry"""
        return self.cache_dir / f"{cache_key}.json"
    
    async def get(self, url: str, options: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached transcription result
        
        Args:
            url: Video URL
            options: Transcription options
            
        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._generate_cache_key(url, options)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            if datetime.fromisoformat(entry["expires_at"]) > datetime.utcnow():
                logger.debug(f"Cache hit (memory): {cache_key}")
                return entry["data"]
            else:
                del self._memory_cache[cache_key]
        
        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                def _read_cache():
                    with open(cache_path, "r") as f:
                        return json.load(f)
                
                loop = asyncio.get_event_loop()
                entry = await loop.run_in_executor(None, _read_cache)
                
                if datetime.fromisoformat(entry["expires_at"]) > datetime.utcnow():
                    logger.debug(f"Cache hit (file): {cache_key}")
                    # Store in memory cache for faster access
                    self._memory_cache[cache_key] = entry
                    return entry["data"]
                else:
                    # Expired, delete file
                    cache_path.unlink()
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
        
        logger.debug(f"Cache miss: {cache_key}")
        return None
    
    async def set(
        self,
        url: str,
        data: Dict[str, Any],
        options: Optional[Dict] = None,
        ttl: Optional[timedelta] = None,
    ):
        """
        Store transcription result in cache
        
        Args:
            url: Video URL
            data: Transcription result data
            options: Transcription options
            ttl: Time to live (default: 24 hours)
        """
        cache_key = self._generate_cache_key(url, options)
        ttl = ttl or self.cache_ttl
        
        entry = {
            "url": url,
            "options": options,
            "data": data,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + ttl).isoformat(),
        }
        
        # Store in memory cache
        self._memory_cache[cache_key] = entry
        
        # Store in file cache
        cache_path = self._get_cache_path(cache_key)
        
        def _write_cache():
            with open(cache_path, "w") as f:
                json.dump(entry, f)
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _write_cache)
            logger.debug(f"Cached: {cache_key}")
        except Exception as e:
            logger.warning(f"Error writing cache: {e}")
    
    async def invalidate(self, url: str, options: Optional[Dict] = None):
        """Invalidate cache entry"""
        cache_key = self._generate_cache_key(url, options)
        
        # Remove from memory
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        # Remove from file
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()
        
        logger.debug(f"Invalidated: {cache_key}")
    
    async def clear_expired(self):
        """Clear all expired cache entries"""
        cleared = 0
        
        # Clear memory cache
        expired_keys = [
            k for k, v in self._memory_cache.items()
            if datetime.fromisoformat(v["expires_at"]) <= datetime.utcnow()
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            cleared += 1
        
        # Clear file cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    entry = json.load(f)
                if datetime.fromisoformat(entry["expires_at"]) <= datetime.utcnow():
                    cache_file.unlink()
                    cleared += 1
            except Exception:
                pass
        
        logger.info(f"Cleared {cleared} expired cache entries")
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_entries = len(self._memory_cache)
        file_entries = len(list(self.cache_dir.glob("*.json")))
        
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.json")
        )
        
        return {
            "memory_entries": memory_entries,
            "file_entries": file_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }


_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get cache service singleton"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service












