"""
Result Cache for Autonomous SAM3 Agent
======================================

Caching system to avoid reprocessing identical requests.
"""

import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for a task result."""
    image_path: str
    text_prompt: str
    result: Dict[str, Any]
    timestamp: float
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class ResultCache:
    """Cache for task results."""
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_size: int = 1000,
        ttl_hours: int = 24
    ):
        """
        Initialize result cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cache entries
            ttl_hours: Time to live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        
        # In-memory cache
        self.cache: Dict[str, CacheEntry] = {}
        
        logger.info(f"Initialized ResultCache (max_size={max_size}, ttl={ttl_hours}h)")
    
    def _generate_key(self, image_path: str, text_prompt: str) -> str:
        """
        Generate cache key from image path and prompt.
        
        Args:
            image_path: Path to image
            text_prompt: Text prompt
            
        Returns:
            Cache key (hash)
        """
        # Use file modification time and size for image
        try:
            path = Path(image_path)
            if path.exists():
                stat = path.stat()
                image_hash = f"{path}_{stat.st_mtime}_{stat.st_size}"
            else:
                image_hash = image_path
        except Exception:
            image_hash = image_path
        
        # Combine image and prompt
        combined = f"{image_hash}:{text_prompt}"
        
        # Generate hash
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, image_path: str, text_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result.
        
        Args:
            image_path: Path to image
            text_prompt: Text prompt
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(image_path, text_prompt)
        
        if key not in self.cache:
            # Try to load from disk
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                    entry = CacheEntry(**data)
                    self.cache[key] = entry
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key}: {e}")
                    return None
            else:
                return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry.is_expired():
            logger.debug(f"Cache entry {key} expired")
            del self.cache[key]
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                cache_file.unlink()
            return None
        
        logger.debug(f"Cache hit for {key}")
        return entry.result
    
    def set(
        self,
        image_path: str,
        text_prompt: str,
        result: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Cache a result.
        
        Args:
            image_path: Path to image
            text_prompt: Text prompt
            result: Result to cache
            ttl_seconds: Time to live in seconds (uses default if None)
        """
        key = self._generate_key(image_path, text_prompt)
        ttl = ttl_seconds or self.ttl_seconds
        
        expires_at = time.time() + ttl if ttl > 0 else None
        
        entry = CacheEntry(
            image_path=image_path,
            text_prompt=text_prompt,
            result=result,
            timestamp=time.time(),
            expires_at=expires_at,
        )
        
        # Store in memory
        self.cache[key] = entry
        
        # Enforce max size
        if len(self.cache) > self.max_size:
            # Remove oldest entries
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].timestamp
            )
            to_remove = len(self.cache) - self.max_size
            for old_key, _ in sorted_entries[:to_remove]:
                del self.cache[old_key]
                cache_file = self.cache_dir / f"{old_key}.json"
                if cache_file.exists():
                    cache_file.unlink()
        
        # Save to disk
        try:
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, "w") as f:
                json.dump(asdict(entry), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache entry {key}: {e}")
        
        logger.debug(f"Cached result for {key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        
        # Remove cache files
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                cache_file.unlink()
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        expired_count = sum(
            1 for entry in self.cache.values()
            if entry.is_expired()
        )
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "expired_entries": expired_count,
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl_seconds / 3600,
        }
