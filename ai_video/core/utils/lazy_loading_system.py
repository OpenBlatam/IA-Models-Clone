from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Iterator, AsyncIterator, Callable
from typing_extensions import Protocol
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import defaultdict, deque
import statistics
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, computed_field
from datetime import datetime, timedelta
            import psutil
from typing import Any, List, Dict, Optional
"""
🚀 LAZY LOADING SYSTEM - AI VIDEO SYSTEM
========================================

Advanced lazy loading techniques for large datasets and substantial API responses
in the AI Video system.

Features:
- Streaming data loading
- Pagination with cursor-based navigation
- Memory-efficient data processing
- Background prefetching
- Chunked data delivery
- Virtual scrolling support
- Cache-aware lazy loading
- Performance monitoring
"""



logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# ============================================================================
# 1. LAZY LOADING CONFIGURATION
# ============================================================================

class LoadingStrategy(str, Enum):
    """Lazy loading strategies."""
    STREAMING = "streaming"           # Stream data as it's requested
    PAGINATION = "pagination"         # Load data in pages
    CHUNKED = "chunked"               # Load data in chunks
    VIRTUAL_SCROLLING = "virtual"     # Virtual scrolling for UI
    BACKGROUND = "background"         # Background prefetching
    HYBRID = "hybrid"                 # Combination of strategies

class CachePolicy(str, Enum):
    """Cache policies for lazy loading."""
    NONE = "none"                     # No caching
    LRU = "lru"                       # Least Recently Used
    LFU = "lfu"                       # Least Frequently Used
    TTL = "ttl"                       # Time To Live
    ADAPTIVE = "adaptive"             # Adaptive caching

@dataclass
class LazyLoadingConfig:
    """Configuration for lazy loading system."""
    strategy: LoadingStrategy = LoadingStrategy.HYBRID
    chunk_size: int = 100
    page_size: int = 50
    prefetch_size: int = 200
    cache_policy: CachePolicy = CachePolicy.LRU
    cache_size: int = 1000
    cache_ttl: int = 3600
    max_memory_mb: int = 512
    enable_monitoring: bool = True
    enable_background_prefetch: bool = True
    background_workers: int = 2
    stream_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0

# ============================================================================
# 2. LAZY LOADING INTERFACES
# ============================================================================

class DataProvider(Protocol[T]):
    """Protocol for data providers."""
    
    async def get_data(self, offset: int, limit: int) -> List[T]:
        """Get data from offset with limit."""
        ...
    
    async def get_total_count(self) -> int:
        """Get total count of available data."""
        ...
    
    async def get_data_by_ids(self, ids: List[str]) -> List[T]:
        """Get data by specific IDs."""
        ...

class StreamProvider(Protocol[T]):
    """Protocol for streaming data providers."""
    
    async def stream_data(self, batch_size: int = 100) -> AsyncIterator[List[T]]:
        """Stream data in batches."""
        ...
    
    async def stream_filtered_data(self, filter_func: Callable[[T], bool], batch_size: int = 100) -> AsyncIterator[List[T]]:
        """Stream filtered data."""
        ...

# ============================================================================
# 3. LAZY LOADING CACHE
# ============================================================================

class LazyLoadingCache:
    """Cache for lazy loading with multiple policies."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.expiry_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, data_type: str, offset: int, limit: int, **kwargs) -> str:
        """Generate cache key."""
        params = f"{data_type}:{offset}:{limit}"
        if kwargs:
            params += ":" + ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return params
    
    async def get(self, data_type: str, offset: int, limit: int, **kwargs) -> Optional[List[T]]:
        """Get data from cache."""
        key = self._generate_key(data_type, offset, limit, **kwargs)
        
        async with self._lock:
            if key in self.cache:
                cache_entry = self.cache[key]
                
                # Check TTL
                if self.config.cache_policy == CachePolicy.TTL:
                    if time.time() > self.expiry_times.get(key, 0):
                        await self._remove_entry(key)
                        return None
                
                # Update access tracking
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                
                return cache_entry["data"]
        
        return None
    
    async def set(self, data_type: str, offset: int, limit: int, data: List[T], **kwargs):
        """Store data in cache."""
        key = self._generate_key(data_type, offset, limit, **kwargs)
        
        async with self._lock:
            # Check cache size
            if len(self.cache) >= self.config.cache_size:
                await self._evict_entries()
            
            # Store data
            self.cache[key] = {
                "data": data,
                "timestamp": time.time(),
                "size": len(data)
            }
            
            # Set expiry time for TTL policy
            if self.config.cache_policy == CachePolicy.TTL:
                self.expiry_times[key] = time.time() + self.config.cache_ttl
            
            # Initialize access tracking
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
    
    async def _evict_entries(self) -> Any:
        """Evict entries based on cache policy."""
        if self.config.cache_policy == CachePolicy.LRU:
            # Evict least recently used
            if self.access_times:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                await self._remove_entry(oldest_key)
        
        elif self.config.cache_policy == CachePolicy.LFU:
            # Evict least frequently used
            if self.access_counts:
                least_used_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
                await self._remove_entry(least_used_key)
        
        elif self.config.cache_policy == CachePolicy.TTL:
            # Evict expired entries
            current_time = time.time()
            expired_keys = [k for k, expiry in self.expiry_times.items() if current_time > expiry]
            for key in expired_keys:
                await self._remove_entry(key)
    
    async def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.access_counts:
            del self.access_counts[key]
        if key in self.expiry_times:
            del self.expiry_times[key]
    
    async def clear(self) -> Any:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.expiry_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry["size"] for entry in self.cache.values())
        
        return {
            "size": len(self.cache),
            "max_size": self.config.cache_size,
            "total_size_bytes": total_size,
            "policy": self.config.cache_policy.value,
            "ttl": self.config.cache_ttl if self.config.cache_policy == CachePolicy.TTL else None
        }

# ============================================================================
# 4. STREAMING LAZY LOADER
# ============================================================================

class StreamingLazyLoader(Generic[T]):
    """Streaming lazy loader for large datasets."""
    
    def __init__(self, provider: StreamProvider[T], config: LazyLoadingConfig):
        
    """__init__ function."""
self.provider = provider
        self.config = config
        self.cache = LazyLoadingCache(config)
        self.active_streams: Dict[str, asyncio.Task] = {}
        self.stream_buffers: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def create_stream(self, stream_id: str, filter_func: Optional[Callable[[T], bool]] = None) -> str:
        """Create a new data stream."""
        async with self._lock:
            if stream_id in self.active_streams:
                return stream_id
            
            # Start streaming task
            if filter_func:
                task = asyncio.create_task(
                    self._stream_filtered_data(stream_id, filter_func)
                )
            else:
                task = asyncio.create_task(
                    self._stream_data(stream_id)
                )
            
            self.active_streams[stream_id] = task
            return stream_id
    
    async def _stream_data(self, stream_id: str):
        """Stream data to buffer."""
        try:
            async for batch in self.provider.stream_data(self.config.chunk_size):
                self.stream_buffers[stream_id].extend(batch)
                
                # Limit buffer size
                while len(self.stream_buffers[stream_id]) > self.config.prefetch_size:
                    self.stream_buffers[stream_id].popleft()
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error streaming data for {stream_id}: {e}")
        finally:
            # Clean up
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def _stream_filtered_data(self, stream_id: str, filter_func: Callable[[T], bool]):
        """Stream filtered data to buffer."""
        try:
            async for batch in self.provider.stream_filtered_data(filter_func, self.config.chunk_size):
                filtered_batch = [item for item in batch if filter_func(item)]
                self.stream_buffers[stream_id].extend(filtered_batch)
                
                # Limit buffer size
                while len(self.stream_buffers[stream_id]) > self.config.prefetch_size:
                    self.stream_buffers[stream_id].popleft()
                
                await asyncio.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error streaming filtered data for {stream_id}: {e}")
        finally:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def get_data(self, stream_id: str, offset: int, limit: int) -> List[T]:
        """Get data from stream."""
        # Check cache first
        cached = await self.cache.get("stream", offset, limit, stream_id=stream_id)
        if cached:
            return cached
        
        # Get from buffer
        buffer = self.stream_buffers.get(stream_id, deque())
        start_idx = offset
        end_idx = offset + limit
        
        if start_idx >= len(buffer):
            return []
        
        result = list(buffer)[start_idx:end_idx]
        
        # Cache result
        await self.cache.set("stream", offset, limit, result, stream_id=stream_id)
        
        return result
    
    async def close_stream(self, stream_id: str):
        """Close a data stream."""
        async with self._lock:
            if stream_id in self.active_streams:
                self.active_streams[stream_id].cancel()
                del self.active_streams[stream_id]
            
            if stream_id in self.stream_buffers:
                del self.stream_buffers[stream_id]
    
    async def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Get stream status."""
        buffer = self.stream_buffers.get(stream_id, deque())
        is_active = stream_id in self.active_streams
        
        return {
            "stream_id": stream_id,
            "is_active": is_active,
            "buffer_size": len(buffer),
            "total_buffered": len(buffer)
        }

# ============================================================================
# 5. PAGINATION LAZY LOADER
# ============================================================================

class PaginationLazyLoader(Generic[T]):
    """Pagination-based lazy loader."""
    
    def __init__(self, provider: DataProvider[T], config: LazyLoadingConfig):
        
    """__init__ function."""
self.provider = provider
        self.config = config
        self.cache = LazyLoadingCache(config)
        self.total_count: Optional[int] = None
        self._total_count_lock = asyncio.Lock()
    
    async def get_page(self, page: int, page_size: Optional[int] = None) -> Dict[str, Any]:
        """Get a specific page of data."""
        page_size = page_size or self.config.page_size
        offset = page * page_size
        
        # Check cache
        cached = await self.cache.get("page", page, page_size)
        if cached:
            return {
                "data": cached,
                "page": page,
                "page_size": page_size,
                "total_count": await self.get_total_count(),
                "cached": True
            }
        
        # Load data
        data = await self.provider.get_data(offset, page_size)
        
        # Cache result
        await self.cache.set("page", page, page_size, data)
        
        return {
            "data": data,
            "page": page,
            "page_size": page_size,
            "total_count": await self.get_total_count(),
            "cached": False
        }
    
    async def get_total_count(self) -> int:
        """Get total count with caching."""
        if self.total_count is not None:
            return self.total_count
        
        async with self._total_count_lock:
            if self.total_count is None:
                self.total_count = await self.provider.get_total_count()
        
        return self.total_count
    
    async def get_pages_info(self) -> Dict[str, Any]:
        """Get pagination information."""
        total_count = await self.get_total_count()
        total_pages = (total_count + self.config.page_size - 1) // self.config.page_size
        
        return {
            "total_count": total_count,
            "page_size": self.config.page_size,
            "total_pages": total_pages,
            "has_next": total_pages > 0,
            "has_previous": False
        }
    
    async def get_data_by_ids(self, ids: List[str]) -> List[T]:
        """Get data by specific IDs."""
        # Check cache first
        cache_key = f"ids:{','.join(sorted(ids))}"
        cached = await self.cache.get("ids", 0, len(ids), ids=cache_key)
        if cached:
            return cached
        
        # Load data
        data = await self.provider.get_data_by_ids(ids)
        
        # Cache result
        await self.cache.set("ids", 0, len(ids), data, ids=cache_key)
        
        return data

# ============================================================================
# 6. CHUNKED LAZY LOADER
# ============================================================================

class ChunkedLazyLoader(Generic[T]):
    """Chunked lazy loader for memory-efficient processing."""
    
    def __init__(self, provider: DataProvider[T], config: LazyLoadingConfig):
        
    """__init__ function."""
self.provider = provider
        self.config = config
        self.cache = LazyLoadingCache(config)
        self.chunk_metadata: Dict[int, Dict[str, Any]] = {}
        self._metadata_lock = asyncio.Lock()
    
    async def get_chunk(self, chunk_id: int) -> Dict[str, Any]:
        """Get a specific chunk of data."""
        # Check cache
        cached = await self.cache.get("chunk", chunk_id, self.config.chunk_size)
        if cached:
            return {
                "data": cached,
                "chunk_id": chunk_id,
                "chunk_size": len(cached),
                "cached": True
            }
        
        # Calculate offset
        offset = chunk_id * self.config.chunk_size
        
        # Load data
        data = await self.provider.get_data(offset, self.config.chunk_size)
        
        # Cache result
        await self.cache.set("chunk", chunk_id, self.config.chunk_size, data)
        
        # Update metadata
        async with self._metadata_lock:
            self.chunk_metadata[chunk_id] = {
                "offset": offset,
                "size": len(data),
                "loaded_at": time.time()
            }
        
        return {
            "data": data,
            "chunk_id": chunk_id,
            "chunk_size": len(data),
            "cached": False
        }
    
    async def get_chunk_range(self, start_chunk: int, end_chunk: int) -> List[Dict[str, Any]]:
        """Get a range of chunks."""
        tasks = [
            self.get_chunk(chunk_id)
            for chunk_id in range(start_chunk, end_chunk + 1)
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_chunk_metadata(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a chunk."""
        async with self._metadata_lock:
            return self.chunk_metadata.get(chunk_id)
    
    async def get_all_chunk_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Get metadata for all chunks."""
        async with self._metadata_lock:
            return self.chunk_metadata.copy()

# ============================================================================
# 7. VIRTUAL SCROLLING LAZY LOADER
# ============================================================================

class VirtualScrollingLazyLoader(Generic[T]):
    """Virtual scrolling lazy loader for UI components."""
    
    def __init__(self, provider: DataProvider[T], config: LazyLoadingConfig):
        
    """__init__ function."""
self.provider = provider
        self.config = config
        self.cache = LazyLoadingCache(config)
        self.viewport_size = 20  # Number of items visible in viewport
        self.buffer_size = 50    # Number of items to keep in buffer
    
    async def get_viewport_data(self, scroll_position: int, viewport_size: Optional[int] = None) -> Dict[str, Any]:
        """Get data for virtual scrolling viewport."""
        viewport_size = viewport_size or self.viewport_size
        
        # Calculate visible range
        start_index = scroll_position
        end_index = scroll_position + viewport_size
        
        # Calculate buffer range
        buffer_start = max(0, start_index - self.buffer_size // 2)
        buffer_end = end_index + self.buffer_size // 2
        
        # Load buffer data
        buffer_data = await self.provider.get_data(buffer_start, buffer_end - buffer_start)
        
        # Extract visible data
        visible_data = buffer_data[start_index - buffer_start:end_index - buffer_start]
        
        return {
            "visible_data": visible_data,
            "buffer_data": buffer_data,
            "scroll_position": scroll_position,
            "viewport_size": viewport_size,
            "buffer_start": buffer_start,
            "buffer_end": buffer_end,
            "total_count": await self.provider.get_total_count()
        }
    
    async def prefetch_viewport(self, current_position: int, direction: str = "both"):
        """Prefetch data for viewport."""
        viewport_size = self.viewport_size
        buffer_size = self.buffer_size
        
        if direction == "forward":
            # Prefetch forward
            start_index = current_position + viewport_size
            end_index = start_index + buffer_size
        elif direction == "backward":
            # Prefetch backward
            end_index = current_position
            start_index = max(0, end_index - buffer_size)
        else:
            # Prefetch both directions
            start_index = max(0, current_position - buffer_size // 2)
            end_index = current_position + viewport_size + buffer_size // 2
        
        # Load data in background
        asyncio.create_task(self._prefetch_data(start_index, end_index))
    
    async def _prefetch_data(self, start_index: int, end_index: int):
        """Prefetch data in background."""
        try:
            data = await self.provider.get_data(start_index, end_index - start_index)
            
            # Cache prefetched data
            for i, item in enumerate(data):
                cache_key = start_index + i
                await self.cache.set("virtual", cache_key, 1, [item])
        
        except Exception as e:
            logger.error(f"Error prefetching data: {e}")

# ============================================================================
# 8. BACKGROUND PREFETCHER
# ============================================================================

class BackgroundPrefetcher(Generic[T]):
    """Background data prefetcher."""
    
    def __init__(self, provider: DataProvider[T], config: LazyLoadingConfig):
        
    """__init__ function."""
self.provider = provider
        self.config = config
        self.cache = LazyLoadingCache(config)
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.is_running = False
    
    async def start(self) -> Any:
        """Start background prefetching."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.config.background_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {self.config.background_workers} background prefetch workers")
    
    async def stop(self) -> Any:
        """Stop background prefetching."""
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Stopped background prefetch workers")
    
    async def prefetch(self, offset: int, limit: int, priority: int = 1):
        """Add prefetch request to queue."""
        await self.prefetch_queue.put({
            "offset": offset,
            "limit": limit,
            "priority": priority,
            "timestamp": time.time()
        })
    
    async def _worker(self, worker_id: str):
        """Background worker for prefetching."""
        while self.is_running:
            try:
                # Get prefetch request
                request = await asyncio.wait_for(
                    self.prefetch_queue.get(),
                    timeout=1.0
                )
                
                # Load data
                data = await self.provider.get_data(
                    request["offset"],
                    request["limit"]
                )
                
                # Cache data
                await self.cache.set(
                    "prefetch",
                    request["offset"],
                    request["limit"],
                    data
                )
                
                logger.debug(f"Worker {worker_id} prefetched {len(data)} items")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(self.config.retry_delay)

# ============================================================================
# 9. HYBRID LAZY LOADER
# ============================================================================

class HybridLazyLoader(Generic[T]):
    """Hybrid lazy loader combining multiple strategies."""
    
    def __init__(self, provider: DataProvider[T], config: LazyLoadingConfig):
        
    """__init__ function."""
self.provider = provider
        self.config = config
        self.cache = LazyLoadingCache(config)
        
        # Initialize different loaders
        self.pagination_loader = PaginationLazyLoader(provider, config)
        self.chunked_loader = ChunkedLazyLoader(provider, config)
        self.virtual_loader = VirtualScrollingLazyLoader(provider, config)
        self.background_prefetcher = BackgroundPrefetcher(provider, config)
        
        # Performance monitoring
        self.performance_stats = defaultdict(list)
    
    async def start(self) -> Any:
        """Start the hybrid loader."""
        await self.background_prefetcher.start()
    
    async def stop(self) -> Any:
        """Stop the hybrid loader."""
        await self.background_prefetcher.stop()
    
    async def get_data(self, strategy: LoadingStrategy, **kwargs) -> Dict[str, Any]:
        """Get data using specified strategy."""
        start_time = time.time()
        
        try:
            if strategy == LoadingStrategy.PAGINATION:
                page = kwargs.get("page", 0)
                page_size = kwargs.get("page_size", self.config.page_size)
                result = await self.pagination_loader.get_page(page, page_size)
            
            elif strategy == LoadingStrategy.CHUNKED:
                chunk_id = kwargs.get("chunk_id", 0)
                result = await self.chunked_loader.get_chunk(chunk_id)
            
            elif strategy == LoadingStrategy.VIRTUAL_SCROLLING:
                scroll_position = kwargs.get("scroll_position", 0)
                viewport_size = kwargs.get("viewport_size", 20)
                result = await self.virtual_loader.get_viewport_data(scroll_position, viewport_size)
            
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            # Record performance
            duration = time.time() - start_time
            self.performance_stats[strategy.value].append(duration)
            
            # Add performance info to result
            result["performance"] = {
                "strategy": strategy.value,
                "duration": duration,
                "cached": result.get("cached", False)
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in hybrid loader: {e}")
            raise
    
    async def prefetch_next(self, current_data: Dict[str, Any]):
        """Prefetch next data based on current data."""
        strategy = current_data.get("performance", {}).get("strategy")
        
        if strategy == LoadingStrategy.PAGINATION.value:
            page = current_data.get("page", 0)
            await self.background_prefetcher.prefetch(
                (page + 1) * self.config.page_size,
                self.config.page_size
            )
        
        elif strategy == LoadingStrategy.CHUNKED.value:
            chunk_id = current_data.get("chunk_id", 0)
            await self.background_prefetcher.prefetch(
                (chunk_id + 1) * self.config.chunk_size,
                self.config.chunk_size
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for strategy, durations in self.performance_stats.items():
            if durations:
                stats[strategy] = {
                    "total_requests": len(durations),
                    "avg_duration": statistics.mean(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations),
                    "throughput": len(durations) / sum(durations) if sum(durations) > 0 else 0
                }
        
        return stats

# ============================================================================
# 10. PERFORMANCE MONITORING
# ============================================================================

class LazyLoadingPerformanceMonitor:
    """Monitor lazy loading performance."""
    
    def __init__(self) -> Any:
        self.performance_data = defaultdict(list)
        self.alerts = []
    
    @asynccontextmanager
    async def monitor_loading(self, operation: str, data_size: int = 0):
        """Monitor lazy loading operation."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.performance_data[operation].append({
                "duration": duration,
                "memory_delta": memory_delta,
                "data_size": data_size,
                "timestamp": time.time()
            })
            
            # Keep only recent data
            if len(self.performance_data[operation]) > 1000:
                self.performance_data[operation] = self.performance_data[operation][-1000:]
            
            # Check for performance alerts
            if duration > 5.0:  # More than 5 seconds
                self.alerts.append({
                    "type": "slow_loading",
                    "operation": operation,
                    "duration": duration,
                    "timestamp": time.time()
                })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "operations": {},
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "timestamp": time.time()
        }
        
        for operation, data in self.performance_data.items():
            if data:
                durations = [item["duration"] for item in data]
                memory_deltas = [item["memory_delta"] for item in data]
                data_sizes = [item["data_size"] for item in data]
                
                report["operations"][operation] = {
                    "total_operations": len(data),
                    "avg_duration": statistics.mean(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations),
                    "avg_memory_delta": statistics.mean(memory_deltas),
                    "avg_data_size": statistics.mean(data_sizes) if data_sizes else 0,
                    "throughput": len(data) / sum(durations) if sum(durations) > 0 else 0
                }
        
        return report
    
    def clear_alerts(self) -> Any:
        """Clear performance alerts."""
        self.alerts.clear()

# ============================================================================
# 11. DECORATORS AND UTILITIES
# ============================================================================

def lazy_loaded(config: LazyLoadingConfig = None):
    """Decorator for lazy loading functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Create lazy loader if not provided
            if "lazy_loader" not in kwargs:
                # This would need to be implemented based on the function
                pass
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def stream_data(batch_size: int = 100):
    """Decorator for streaming data functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Implementation would depend on the function
            pass
        
        return wrapper
    return decorator

# ============================================================================
# 12. USAGE EXAMPLES
# ============================================================================

async def example_lazy_loading():
    """Example of using the lazy loading system."""
    
    # Create configuration
    config = LazyLoadingConfig(
        strategy=LoadingStrategy.HYBRID,
        chunk_size=100,
        page_size=50,
        prefetch_size=200,
        cache_policy=CachePolicy.LRU,
        cache_size=1000,
        enable_background_prefetch=True
    )
    
    # Example data provider (you would implement this)
    class ExampleDataProvider:
        async def get_data(self, offset: int, limit: int) -> List[Dict[str, Any]]:
            # Simulate data loading
            await asyncio.sleep(0.1)
            return [{"id": i, "data": f"item_{i}"} for i in range(offset, offset + limit)]
        
        async def get_total_count(self) -> int:
            return 10000
        
        async def get_data_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
            return [{"id": id, "data": f"item_{id}"} for id in ids]
    
    provider = ExampleDataProvider()
    
    # Create hybrid loader
    loader = HybridLazyLoader(provider, config)
    await loader.start()
    
    try:
        # Test pagination
        page_data = await loader.get_data(LoadingStrategy.PAGINATION, page=0, page_size=20)
        print(f"Pagination: {len(page_data['data'])} items")
        
        # Test chunked loading
        chunk_data = await loader.get_data(LoadingStrategy.CHUNKED, chunk_id=0)
        print(f"Chunked: {len(chunk_data['data'])} items")
        
        # Test virtual scrolling
        viewport_data = await loader.get_data(
            LoadingStrategy.VIRTUAL_SCROLLING,
            scroll_position=0,
            viewport_size=10
        )
        print(f"Virtual scrolling: {len(viewport_data['visible_data'])} visible items")
        
        # Get performance stats
        stats = loader.get_performance_stats()
        print(f"Performance stats: {stats}")
        
    finally:
        await loader.stop()

match __name__:
    case "__main__":
    asyncio.run(example_lazy_loading()) 