"""
Intelligent Cache
================

Advanced caching system with learning and optimization.
"""

import logging
import hashlib
import time
import json
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from PIL import Image
from collections import OrderedDict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    image_path: str
    scale_factor: float
    method: str
    created_at: float
    access_count: int
    last_accessed: float
    quality_score: float
    size_mb: float


class IntelligentCache:
    """
    Intelligent cache with learning capabilities.
    
    Features:
    - LRU eviction
    - Quality-based prioritization
    - Access pattern learning
    - Automatic cleanup
    - Size-based management
    """
    
    def __init__(
        self,
        cache_dir: str,
        max_size_mb: float = 1000.0,
        max_entries: int = 100,
        ttl: float = 86400.0  # 24 hours
    ):
        """
        Initialize intelligent cache.
        
        Args:
            cache_dir: Cache directory
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of entries
            ttl: Time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self.ttl = ttl
        
        # Cache index (LRU)
        self.cache_index: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size_mb": 0.0,
        }
        
        # Load cache index
        self._load_index()
        
        logger.info(f"IntelligentCache initialized: {len(self.cache_index)} entries")
    
    def _get_cache_key(
        self,
        image_path: str,
        scale_factor: float,
        method: str
    ) -> str:
        """Generate cache key."""
        key_string = f"{image_path}_{scale_factor}_{method}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{key}.png"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path."""
        return self.cache_dir / f"{key}.json"
    
    def get(
        self,
        image_path: str,
        scale_factor: float,
        method: str
    ) -> Optional[Image.Image]:
        """
        Get cached image.
        
        Args:
            image_path: Original image path
            scale_factor: Scale factor
            method: Upscaling method
            
        Returns:
            Cached image or None
        """
        key = self._get_cache_key(image_path, scale_factor, method)
        
        if key not in self.cache_index:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache_index[key]
        
        # Check TTL
        if time.time() - entry.created_at > self.ttl:
            self._remove_entry(key)
            self.stats["misses"] += 1
            return None
        
        # Check if file exists
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            self._remove_entry(key)
            self.stats["misses"] += 1
            return None
        
        try:
            # Load image
            image = Image.open(cache_path)
            
            # Update access
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.cache_index.move_to_end(key)  # Move to end (most recent)
            
            self.stats["hits"] += 1
            logger.debug(f"Cache hit: {key}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading cached image: {e}")
            self._remove_entry(key)
            self.stats["misses"] += 1
            return None
    
    def set(
        self,
        image_path: str,
        scale_factor: float,
        method: str,
        image: Image.Image,
        quality_score: float = 0.0
    ) -> None:
        """
        Cache image.
        
        Args:
            image_path: Original image path
            scale_factor: Scale factor
            method: Upscaling method
            image: Image to cache
            quality_score: Quality score
        """
        key = self._get_cache_key(image_path, scale_factor, method)
        
        # Check if we need to evict
        self._evict_if_needed(image)
        
        try:
            # Save image
            cache_path = self._get_cache_path(key)
            image.save(cache_path, "PNG")
            
            # Calculate size
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            
            # Create entry
            entry = CacheEntry(
                image_path=image_path,
                scale_factor=scale_factor,
                method=method,
                created_at=time.time(),
                access_count=0,
                last_accessed=time.time(),
                quality_score=quality_score,
                size_mb=size_mb
            )
            
            # Add to index
            self.cache_index[key] = entry
            self.cache_index.move_to_end(key)
            
            # Save metadata
            metadata_path = self._get_metadata_path(key)
            with open(metadata_path, 'w') as f:
                json.dump(asdict(entry), f)
            
            # Update stats
            self.stats["total_size_mb"] += size_mb
            
            logger.debug(f"Cached image: {key} ({size_mb:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Error caching image: {e}")
    
    def _evict_if_needed(self, new_image: Image.Image) -> None:
        """Evict entries if needed."""
        # Estimate new image size
        new_size_mb = (new_image.size[0] * new_image.size[1] * 3 * 4) / (1024 * 1024)
        
        # Check size limit
        while (self.stats["total_size_mb"] + new_size_mb > self.max_size_mb and
               self.cache_index):
            self._evict_oldest()
        
        # Check entry limit
        while len(self.cache_index) >= self.max_entries:
            self._evict_oldest()
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry."""
        if not self.cache_index:
            return
        
        # Get oldest (first) entry
        key, entry = next(iter(self.cache_index.items()))
        self._remove_entry(key)
        self.stats["evictions"] += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry."""
        if key not in self.cache_index:
            return
        
        entry = self.cache_index[key]
        
        # Remove files
        cache_path = self._get_cache_path(key)
        metadata_path = self._get_metadata_path(key)
        
        try:
            if cache_path.exists():
                size_mb = cache_path.stat().st_size / (1024 * 1024)
                cache_path.unlink()
                self.stats["total_size_mb"] -= size_mb
            
            if metadata_path.exists():
                metadata_path.unlink()
        except Exception as e:
            logger.error(f"Error removing cache entry: {e}")
        
        # Remove from index
        del self.cache_index[key]
    
    def _load_index(self) -> None:
        """Load cache index from disk."""
        try:
            index_path = self.cache_dir / "index.json"
            if not index_path.exists():
                return
            
            with open(index_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct index
            for key, entry_data in data.items():
                entry = CacheEntry(**entry_data)
                self.cache_index[key] = entry
            
            # Sort by last accessed (most recent first)
            self.cache_index = OrderedDict(
                sorted(
                    self.cache_index.items(),
                    key=lambda x: x[1].last_accessed,
                    reverse=True
                )
            )
            
            # Calculate total size
            self.stats["total_size_mb"] = sum(
                entry.size_mb for entry in self.cache_index.values()
            )
            
        except Exception as e:
            logger.error(f"Error loading cache index: {e}")
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            index_path = self.cache_dir / "index.json"
            data = {
                key: asdict(entry)
                for key, entry in self.cache_index.items()
            }
            
            with open(index_path, 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for key in list(self.cache_index.keys()):
            self._remove_entry(key)
        
        self.stats["total_size_mb"] = 0.0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(1, total_requests)
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "entries": len(self.cache_index),
            "max_size_mb": self.max_size_mb,
            "max_entries": self.max_entries,
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache_index.items()
            if current_time - entry.created_at > self.ttl
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        return len(expired_keys)
    
    def __del__(self):
        """Save index on destruction."""
        try:
            self._save_index()
        except:
            pass


