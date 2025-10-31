#!/usr/bin/env python3
"""
Intelligent Cache Manager for Enhanced HeyGen AI
Handles caching for models, audio, video, and API responses with smart expiration.
"""

import os
import json
import hashlib
import time
import asyncio
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import pickle
import gzip
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import psutil

logger = structlog.get_logger()

class CachePriority(Enum):
    """Cache priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CacheType(Enum):
    """Types of cached data."""
    MODEL = "model"
    AUDIO = "audio"
    VIDEO = "video"
    API_RESPONSE = "api_response"
    METADATA = "metadata"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    cache_type: CacheType
    priority: CachePriority
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    expires_at: Optional[float]
    tags: List[str]
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def should_evict(self, max_age: float, max_access_count: int) -> bool:
        """Check if entry should be evicted based on age and access patterns."""
        age = time.time() - self.created_at
        return (age > max_age and self.access_count < max_access_count) or self.is_expired()

class IntelligentCacheManager:
    """Intelligent cache manager with automatic optimization."""
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        max_memory_mb: int = 1024,
        max_disk_gb: int = 10,
        enable_compression: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_gb * 1024 * 1024 * 1024
        self.enable_compression = enable_compression
        
        # Cache storage
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.disk_cache_dir = self.cache_dir / "disk"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "decompressions": 0
        }
        
        # Initialize cache
        self._initialize_cache()
        
        # Start background maintenance
        asyncio.create_task(self._background_maintenance())
    
    def _initialize_cache(self):
        """Initialize cache directories and load metadata."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Reconstruct memory cache for high-priority items
                    for key, entry_data in metadata.items():
                        if entry_data["priority"] in ["high", "critical"]:
                            self._load_entry_to_memory(key, entry_data)
            
            logger.info("Cache manager initialized", 
                       cache_dir=str(self.cache_dir),
                       max_memory_mb=self.max_memory_bytes // (1024 * 1024))
            
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
    
    def _generate_cache_key(self, data: Any, prefix: str = "") -> str:
        """Generate a unique cache key."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        hash_content = f"{prefix}:{content}"
        return hashlib.sha256(hash_content.encode()).hexdigest()
    
    def _get_entry_size(self, data: Any) -> int:
        """Estimate the size of cached data in bytes."""
        if isinstance(data, (str, bytes)):
            return len(data)
        elif isinstance(data, dict):
            return len(json.dumps(data))
        elif isinstance(data, list):
            return sum(self._get_entry_size(item) for item in data)
        else:
            return len(pickle.dumps(data))
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if compression is enabled."""
        if not self.enable_compression:
            return data
        
        try:
            compressed = gzip.compress(data)
            if len(compressed) < len(data):
                self.stats["compressions"] += 1
                return compressed
            return data
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if it was compressed."""
        try:
            if data.startswith(b'\x1f\x8b'):  # Gzip magic number
                decompressed = gzip.decompress(data)
                self.stats["decompressions"] += 1
                return decompressed
            return data
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return data
    
    async def get(
        self, 
        key: str, 
        cache_type: CacheType = CacheType.API_RESPONSE
    ) -> Optional[Any]:
        """Get data from cache."""
        try:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired():
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    self.stats["hits"] += 1
                    return entry.data
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
            
            # Check disk cache
            disk_entry = await self._load_from_disk(key)
            if disk_entry and not disk_entry.is_expired():
                # Move to memory if high priority
                if disk_entry.priority in [CachePriority.HIGH, CachePriority.CRITICAL]:
                    self._add_to_memory_cache(disk_entry)
                
                disk_entry.last_accessed = time.time()
                disk_entry.access_count += 1
                self.stats["hits"] += 1
                return disk_entry.data
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        data: Any,
        cache_type: CacheType = CacheType.API_RESPONSE,
        priority: CachePriority = CachePriority.MEDIUM,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        force_disk: bool = False
    ) -> bool:
        """Set data in cache."""
        try:
            # Create cache entry
            expires_at = None
            if ttl_seconds:
                expires_at = time.time() + ttl_seconds
            
            entry = CacheEntry(
                key=key,
                data=data,
                cache_type=cache_type,
                priority=priority,
                size_bytes=self._get_entry_size(data),
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                expires_at=expires_at,
                tags=tags or []
            )
            
            # Store based on priority and size
            if (priority in [CachePriority.HIGH, CachePriority.CRITICAL] and 
                not force_disk and 
                entry.size_bytes < self.max_memory_bytes // 4):
                # High priority goes to memory
                self._add_to_memory_cache(entry)
            else:
                # Store on disk
                await self._store_on_disk(entry)
            
            # Update metadata
            await self._update_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    def _add_to_memory_cache(self, entry: CacheEntry):
        """Add entry to memory cache with eviction if needed."""
        # Check if we need to evict entries
        current_memory = sum(e.size_bytes for e in self.memory_cache.values())
        
        if current_memory + entry.size_bytes > self.max_memory_bytes:
            self._evict_memory_entries(entry.size_bytes)
        
        self.memory_cache[entry.key] = entry
    
    def _evict_memory_entries(self, needed_bytes: int):
        """Evict entries from memory cache to free space."""
        # Sort by priority and access patterns
        entries = list(self.memory_cache.values())
        entries.sort(key=lambda e: (
            CachePriority.CRITICAL.value != e.priority.value,
            e.access_count,
            e.last_accessed
        ))
        
        freed_bytes = 0
        for entry in entries:
            if freed_bytes >= needed_bytes:
                break
            
            del self.memory_cache[entry.key]
            freed_bytes += entry.size_bytes
            self.stats["evictions"] += 1
            
            logger.debug(f"Evicted entry {entry.key} from memory cache")
    
    async def _store_on_disk(self, entry: CacheEntry):
        """Store entry on disk."""
        try:
            # Serialize data
            if isinstance(entry.data, (str, bytes)):
                data_bytes = entry.data.encode() if isinstance(entry.data, str) else entry.data
            else:
                data_bytes = pickle.dumps(entry.data)
            
            # Compress if beneficial
            data_bytes = self._compress_data(data_bytes)
            
            # Save to disk
            file_path = self.disk_cache_dir / f"{entry.key}.cache"
            with open(file_path, 'wb') as f:
                f.write(data_bytes)
            
            # Save metadata
            await self._update_metadata()
            
        except Exception as e:
            logger.error(f"Failed to store entry {entry.key} on disk: {e}")
    
    async def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load entry from disk."""
        try:
            file_path = self.disk_cache_dir / f"{key}.cache"
            if not file_path.exists():
                return None
            
            # Load metadata
            metadata = await self._load_metadata()
            if key not in metadata:
                return None
            
            entry_data = metadata[key]
            
            # Load data
            with open(file_path, 'rb') as f:
                data_bytes = f.read()
            
            # Decompress if needed
            data_bytes = self._decompress_data(data_bytes)
            
            # Deserialize data
            if entry_data["cache_type"] == CacheType.API_RESPONSE:
                data = json.loads(data_bytes.decode())
            else:
                data = pickle.loads(data_bytes)
            
            # Reconstruct entry
            entry = CacheEntry(
                key=key,
                data=data,
                cache_type=CacheType(entry_data["cache_type"]),
                priority=CachePriority(entry_data["priority"]),
                size_bytes=entry_data["size_bytes"],
                created_at=entry_data["created_at"],
                last_accessed=entry_data["last_accessed"],
                access_count=entry_data["access_count"],
                expires_at=entry_data["expires_at"],
                tags=entry_data["tags"]
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to load entry {key} from disk: {e}")
            return None
    
    async def _update_metadata(self):
        """Update cache metadata file."""
        try:
            # Combine memory and disk metadata
            metadata = {}
            
            # Memory cache entries
            for entry in self.memory_cache.values():
                metadata[entry.key] = asdict(entry)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
    
    async def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag."""
        invalidated = 0
        
        # Check memory cache
        keys_to_remove = []
        for key, entry in self.memory_cache.items():
            if tag in entry.tags:
                keys_to_remove.append(key)
                invalidated += 1
        
        for key in keys_to_remove:
            del self.memory_cache[key]
        
        # Check disk cache
        metadata = await self._load_metadata()
        keys_to_remove = []
        for key, entry_data in metadata.items():
            if tag in entry_data.get("tags", []):
                keys_to_remove.append(key)
                invalidated += 1
        
        for key in keys_to_remove:
            file_path = self.disk_cache_dir / f"{key}.cache"
            if file_path.exists():
                file_path.unlink()
        
        if invalidated > 0:
            await self._update_metadata()
        
        return invalidated
    
    async def clear_expired(self) -> int:
        """Clear all expired entries."""
        cleared = 0
        
        # Clear expired from memory
        keys_to_remove = []
        for key, entry in self.memory_cache.items():
            if entry.is_expired():
                keys_to_remove.append(key)
                cleared += 1
        
        for key in keys_to_remove:
            del self.memory_cache[key]
        
        # Clear expired from disk
        metadata = await self._load_metadata()
        keys_to_remove = []
        for key, entry_data in metadata.items():
            if entry_data.get("expires_at") and time.time() > entry_data["expires_at"]:
                keys_to_remove.append(key)
                cleared += 1
        
        for key in keys_to_remove:
            file_path = self.disk_cache_dir / f"{key}.cache"
            if file_path.exists():
                file_path.unlink()
        
        if cleared > 0:
            await self._update_metadata()
        
        return cleared
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_memory = sum(e.size_bytes for e in self.memory_cache.values())
        current_disk = sum(f.stat().st_size for f in self.disk_cache_dir.glob("*.cache"))
        
        return {
            **self.stats,
            "memory_usage_bytes": current_memory,
            "memory_usage_mb": current_memory / (1024 * 1024),
            "disk_usage_bytes": current_disk,
            "disk_usage_mb": current_disk / (1024 * 1024),
            "memory_entries": len(self.memory_cache),
            "disk_entries": len(list(self.disk_cache_dir.glob("*.cache"))),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "max_disk_gb": self.max_disk_bytes / (1024 * 1024 * 1024)
        }
    
    async def _background_maintenance(self):
        """Background maintenance task."""
        while True:
            try:
                # Clear expired entries
                expired_count = await self.clear_expired()
                if expired_count > 0:
                    logger.debug(f"Cleared {expired_count} expired cache entries")
                
                # Update metadata
                await self._update_metadata()
                
                # Log stats periodically
                if time.time() % 300 < 60:  # Every 5 minutes
                    stats = await self.get_stats()
                    logger.info("Cache statistics", **stats)
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Background maintenance failed: {e}")
                await asyncio.sleep(60)
    
    async def shutdown(self):
        """Shutdown cache manager."""
        try:
            await self._update_metadata()
            logger.info("Cache manager shutdown complete")
        except Exception as e:
            logger.error(f"Cache shutdown failed: {e}")

# Global cache instance
cache_manager: Optional[IntelligentCacheManager] = None

def get_cache_manager() -> IntelligentCacheManager:
    """Get global cache manager instance."""
    global cache_manager
    if cache_manager is None:
        cache_manager = IntelligentCacheManager()
    return cache_manager

async def shutdown_cache_manager():
    """Shutdown global cache manager."""
    global cache_manager
    if cache_manager:
        await cache_manager.shutdown()
        cache_manager = None

