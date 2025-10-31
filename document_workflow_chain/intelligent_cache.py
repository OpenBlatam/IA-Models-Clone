"""
Intelligent Cache System for Workflow Chain Engine
=================================================

This module implements an intelligent caching system that optimizes performance
by storing and reusing processed document chunks, analysis results, and model responses.

Features:
- Smart document chunk caching
- Analysis result caching
- Model response caching
- Cache invalidation strategies
- Performance optimization
- Memory management
"""

import hashlib
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
import asyncio
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: int = 3600  # 1 hour default
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate size of cached value in bytes"""
        try:
            if isinstance(self.value, str):
                return len(self.value.encode('utf-8'))
            elif isinstance(self.value, (dict, list)):
                return len(json.dumps(self.value).encode('utf-8'))
            else:
                return len(pickle.dumps(self.value))
        except Exception:
            return 1024  # Default size estimate
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > (self.created_at + timedelta(seconds=self.ttl_seconds))
    
    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class IntelligentCache:
    """Intelligent caching system with multiple strategies"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        # Cache statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'total_size_bytes': 0,
            'average_access_time': 0.0
        }
    
    def _generate_key(self, content: str, operation: str, **kwargs) -> str:
        """Generate cache key from content and operation"""
        # Create a hash of the content and operation parameters
        key_data = {
            'content_hash': hashlib.md5(content.encode('utf-8')).hexdigest(),
            'operation': operation,
            'params': sorted(kwargs.items())
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired():
                # Remove expired entry
                self._remove_entry(key)
                self.miss_count += 1
                self.stats['cache_misses'] += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.touch()
            
            self.hit_count += 1
            self.stats['cache_hits'] += 1
            self.stats['average_access_time'] = (
                (self.stats['average_access_time'] * (self.stats['total_requests'] - 1) + 
                 (time.time() - start_time)) / self.stats['total_requests']
            )
            
            return entry.value
        
        self.miss_count += 1
        self.stats['cache_misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            tags = tags or []
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl,
                tags=tags
            )
            
            # Check if we need to evict entries
            while (self.current_size_bytes + entry.size_bytes > self.max_size_bytes and 
                   len(self.cache) > 0):
                self._evict_least_recently_used()
            
            # Add or update entry
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size_bytes -= old_entry.size_bytes
            
            self.cache[key] = entry
            self.current_size_bytes += entry.size_bytes
            self.stats['total_size_bytes'] = self.current_size_bytes
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache entry: {str(e)}")
            return False
    
    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[key]
    
    def _evict_least_recently_used(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        # Remove the first (oldest) entry
        key, entry = self.cache.popitem(last=False)
        self.current_size_bytes -= entry.size_bytes
        self.eviction_count += 1
        self.stats['evictions'] += 1
        
        logger.debug(f"Evicted cache entry: {key}")
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate entries with specific tags"""
        invalidated = 0
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_entry(key)
            invalidated += 1
        
        logger.info(f"Invalidated {invalidated} cache entries with tags: {tags}")
        return invalidated
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.current_size_bytes = 0
        self.stats['total_size_bytes'] = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (self.hit_count / max(self.hit_count + self.miss_count, 1)) * 100
        
        return {
            'total_entries': len(self.cache),
            'total_size_mb': self.current_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'hit_rate_percent': hit_rate,
            'total_requests': self.stats['total_requests'],
            'cache_hits': self.hit_count,
            'cache_misses': self.miss_count,
            'evictions': self.eviction_count,
            'average_access_time_ms': self.stats['average_access_time'] * 1000,
            'utilization_percent': (self.current_size_bytes / self.max_size_bytes) * 100
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)


class DocumentChunkCache:
    """Specialized cache for document chunks"""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.chunk_ttl = 7200  # 2 hours for chunks
    
    def get_chunk(self, content: str, chunk_index: int) -> Optional[Any]:
        """Get cached document chunk"""
        key = self._generate_chunk_key(content, chunk_index)
        return self.cache.get(key)
    
    def set_chunk(self, content: str, chunk_index: int, chunk_data: Any) -> bool:
        """Cache document chunk"""
        key = self._generate_chunk_key(content, chunk_index)
        return self.cache.set(key, chunk_data, ttl=self.chunk_ttl, tags=['chunk'])
    
    def _generate_chunk_key(self, content: str, chunk_index: int) -> str:
        """Generate key for document chunk"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"chunk:{content_hash}:{chunk_index}"


class AnalysisResultCache:
    """Specialized cache for analysis results"""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.analysis_ttl = 3600  # 1 hour for analysis results
    
    def get_analysis(self, content: str, analysis_type: str) -> Optional[Any]:
        """Get cached analysis result"""
        key = self._generate_analysis_key(content, analysis_type)
        return self.cache.get(key)
    
    def set_analysis(self, content: str, analysis_type: str, result: Any) -> bool:
        """Cache analysis result"""
        key = self._generate_analysis_key(content, analysis_type)
        return self.cache.set(key, result, ttl=self.analysis_ttl, tags=['analysis', analysis_type])
    
    def _generate_analysis_key(self, content: str, analysis_type: str) -> str:
        """Generate key for analysis result"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"analysis:{analysis_type}:{content_hash}"


class ModelResponseCache:
    """Specialized cache for model responses"""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.model_ttl = 1800  # 30 minutes for model responses
    
    def get_response(self, prompt: str, model: str) -> Optional[Any]:
        """Get cached model response"""
        key = self._generate_response_key(prompt, model)
        return self.cache.get(key)
    
    def set_response(self, prompt: str, model: str, response: Any) -> bool:
        """Cache model response"""
        key = self._generate_response_key(prompt, model)
        return self.cache.set(key, response, ttl=self.model_ttl, tags=['model_response', model])
    
    def _generate_response_key(self, prompt: str, model: str) -> str:
        """Generate key for model response"""
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        return f"model:{model}:{prompt_hash}"


class CacheManager:
    """Central cache manager for the workflow chain engine"""
    
    def __init__(self, max_size_mb: int = 200):
        self.main_cache = IntelligentCache(max_size_mb=max_size_mb)
        self.chunk_cache = DocumentChunkCache(self.main_cache)
        self.analysis_cache = AnalysisResultCache(self.main_cache)
        self.model_cache = ModelResponseCache(self.main_cache)
        
        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                    self.main_cache.cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in cache cleanup task: {str(e)}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def get_chunk(self, content: str, chunk_index: int) -> Optional[Any]:
        """Get cached document chunk"""
        return self.chunk_cache.get_chunk(content, chunk_index)
    
    def set_chunk(self, content: str, chunk_index: int, chunk_data: Any) -> bool:
        """Cache document chunk"""
        return self.chunk_cache.set_chunk(content, chunk_index, chunk_data)
    
    def get_analysis(self, content: str, analysis_type: str) -> Optional[Any]:
        """Get cached analysis result"""
        return self.analysis_cache.get_analysis(content, analysis_type)
    
    def set_analysis(self, content: str, analysis_type: str, result: Any) -> bool:
        """Cache analysis result"""
        return self.analysis_cache.set_analysis(content, analysis_type, result)
    
    def get_model_response(self, prompt: str, model: str) -> Optional[Any]:
        """Get cached model response"""
        return self.model_cache.get_response(prompt, model)
    
    def set_model_response(self, prompt: str, model: str, response: Any) -> bool:
        """Cache model response"""
        return self.model_cache.set_response(prompt, model, response)
    
    def invalidate_content(self, content: str):
        """Invalidate all cache entries for specific content"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Find and remove all entries related to this content
        keys_to_remove = []
        for key in self.main_cache.cache.keys():
            if content_hash in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.main_cache._remove_entry(key)
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries for content")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        base_stats = self.main_cache.get_stats()
        
        # Add specialized cache stats
        chunk_count = sum(1 for key in self.main_cache.cache.keys() if key.startswith('chunk:'))
        analysis_count = sum(1 for key in self.main_cache.cache.keys() if key.startswith('analysis:'))
        model_count = sum(1 for key in self.main_cache.cache.keys() if key.startswith('model:'))
        
        return {
            **base_stats,
            'chunk_entries': chunk_count,
            'analysis_entries': analysis_count,
            'model_response_entries': model_count,
            'cache_efficiency': {
                'chunk_hit_rate': self._calculate_specialized_hit_rate('chunk:'),
                'analysis_hit_rate': self._calculate_specialized_hit_rate('analysis:'),
                'model_hit_rate': self._calculate_specialized_hit_rate('model:')
            }
        }
    
    def _calculate_specialized_hit_rate(self, prefix: str) -> float:
        """Calculate hit rate for specialized cache types"""
        # This is a simplified calculation - in a real implementation,
        # you'd track hits/misses per cache type separately
        return self.main_cache.get_stats()['hit_rate_percent']
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache by removing least valuable entries"""
        optimization_results = {
            'entries_removed': 0,
            'space_freed_mb': 0,
            'optimization_time': 0
        }
        
        start_time = time.time()
        
        # Remove entries with low access count and old age
        keys_to_remove = []
        current_time = datetime.now()
        
        for key, entry in self.main_cache.cache.items():
            age_hours = (current_time - entry.created_at).total_seconds() / 3600
            access_rate = entry.access_count / max(age_hours, 0.1)
            
            # Remove entries that are old and rarely accessed
            if age_hours > 2 and access_rate < 0.1:
                keys_to_remove.append(key)
        
        # Remove selected entries
        for key in keys_to_remove:
            entry = self.main_cache.cache[key]
            optimization_results['space_freed_mb'] += entry.size_bytes / (1024 * 1024)
            self.main_cache._remove_entry(key)
            optimization_results['entries_removed'] += 1
        
        optimization_results['optimization_time'] = time.time() - start_time
        
        logger.info(f"Cache optimization completed: {optimization_results}")
        return optimization_results
    
    def shutdown(self):
        """Shutdown cache manager and cleanup tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Save cache statistics
        stats = self.get_comprehensive_stats()
        logger.info(f"Cache manager shutdown. Final stats: {stats}")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(max_size_mb: int = 200) -> CacheManager:
    """Get or create global cache manager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(max_size_mb=max_size_mb)
    return _cache_manager


def clear_global_cache():
    """Clear the global cache"""
    global _cache_manager
    if _cache_manager:
        _cache_manager.main_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    global _cache_manager
    if _cache_manager:
        return _cache_manager.get_comprehensive_stats()
    return {'error': 'Cache manager not initialized'}



























