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
import logging
import weakref
import threading
from typing import Any, Optional, Dict, List, Callable, Awaitable, Set, Tuple, Union, AsyncGenerator, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import gc
import json
import hashlib
from contextlib import asynccontextmanager
import orjson
from pydantic import BaseModel, Field
import structlog
from typing import Any, List, Dict, Optional
"""
🚀 Advanced Lazy Loading for Large Datasets & API Responses
==========================================================

Comprehensive lazy loading system optimized for:
- Large dataset processing
- Substantial API responses
- Streaming data handling
- Memory-efficient operations
- Chunked processing
- Progressive loading
- Background prefetching
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')

class LoadingStrategy(Enum):
    """Loading strategies for different data types"""
    EAGER = "eager"           # Load all data immediately
    LAZY = "lazy"             # Load on demand
    STREAMING = "streaming"   # Stream data progressively
    CHUNKED = "chunked"       # Load in chunks
    PAGINATED = "paginated"   # Load with pagination
    BACKGROUND = "background" # Load in background
    HYBRID = "hybrid"         # Combination of strategies

class DataType(Enum):
    """Data types for optimization"""
    SMALL = "small"           # < 1KB
    MEDIUM = "medium"         # 1KB - 1MB
    LARGE = "large"           # 1MB - 100MB
    HUGE = "huge"             # > 100MB
    STREAMING = "streaming"   # Continuous data stream

class ChunkStrategy(Enum):
    """Chunking strategies"""
    FIXED_SIZE = "fixed_size"     # Fixed chunk size
    DYNAMIC_SIZE = "dynamic_size" # Dynamic chunk size based on content
    TIME_BASED = "time_based"     # Chunk based on time intervals
    CONTENT_BASED = "content_based" # Chunk based on content boundaries

@dataclass
class LazyLoadingConfig:
    """Configuration for lazy loading system"""
    # General settings
    default_strategy: LoadingStrategy = LoadingStrategy.LAZY
    enable_streaming: bool = True
    enable_chunking: bool = True
    enable_pagination: bool = True
    enable_background_loading: bool = True
    
    # Memory settings
    max_memory_mb: int = 1024
    chunk_size: int = 1024 * 1024  # 1MB default chunk
    max_chunks_in_memory: int = 10
    memory_cleanup_threshold: float = 0.8
    
    # Performance settings
    prefetch_enabled: bool = True
    prefetch_distance: int = 2
    background_workers: int = 4
    max_concurrent_loads: int = 10
    
    # Caching settings
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # Monitoring settings
    enable_metrics: bool = True
    log_slow_operations: bool = True
    slow_operation_threshold: float = 1.0

@dataclass
class LoadingMetrics:
    """Performance metrics for lazy loading"""
    total_operations: int = 0
    streaming_operations: int = 0
    chunked_operations: int = 0
    paginated_operations: int = 0
    background_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_load_time: float = 0.0
    average_load_time: float = 0.0
    memory_usage: int = 0
    errors: int = 0

@dataclass
class ChunkInfo:
    """Information about a data chunk"""
    chunk_id: str
    data: Any
    size: int
    index: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class DataChunker:
    """
    Intelligent data chunking system for large datasets.
    """
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.chunk_cache = {}
        self._lock = asyncio.Lock()
    
    async def chunk_data(
        self, 
        data: Any, 
        strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE,
        chunk_size: int = None
    ) -> List[ChunkInfo]:
        """Chunk data based on strategy."""
        chunk_size = chunk_size or self.config.chunk_size
        
        if strategy == ChunkStrategy.FIXED_SIZE:
            return await self._chunk_fixed_size(data, chunk_size)
        elif strategy == ChunkStrategy.DYNAMIC_SIZE:
            return await self._chunk_dynamic_size(data, chunk_size)
        elif strategy == ChunkStrategy.CONTENT_BASED:
            return await self._chunk_content_based(data)
        else:
            return await self._chunk_fixed_size(data, chunk_size)
    
    async def _chunk_fixed_size(self, data: Any, chunk_size: int) -> List[ChunkInfo]:
        """Chunk data into fixed-size pieces."""
        if isinstance(data, (str, bytes)):
            return await self._chunk_text_data(data, chunk_size)
        elif isinstance(data, list):
            return await self._chunk_list_data(data, chunk_size)
        elif isinstance(data, dict):
            return await self._chunk_dict_data(data, chunk_size)
        else:
            # For other types, serialize first
            serialized = orjson.dumps(data)
            return await self._chunk_text_data(serialized, chunk_size)
    
    async def _chunk_text_data(self, data: Union[str, bytes], chunk_size: int) -> List[ChunkInfo]:
        """Chunk text or binary data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        chunks = []
        total_size = len(data)
        total_chunks = (total_size + chunk_size - 1) // chunk_size
        
        for i in range(0, total_size, chunk_size):
            chunk_data = data[i:i + chunk_size]
            chunk_id = hashlib.md5(f"{i}_{total_size}".encode()).hexdigest()
            
            chunk = ChunkInfo(
                chunk_id=chunk_id,
                data=chunk_data,
                size=len(chunk_data),
                index=i // chunk_size,
                total_chunks=total_chunks,
                metadata={"type": "text", "encoding": "utf-8"}
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_list_data(self, data: List[Any], chunk_size: int) -> List[ChunkInfo]:
        """Chunk list data."""
        chunks = []
        total_items = len(data)
        items_per_chunk = max(1, chunk_size // 100)  # Estimate 100 bytes per item
        total_chunks = (total_items + items_per_chunk - 1) // items_per_chunk
        
        for i in range(0, total_items, items_per_chunk):
            chunk_data = data[i:i + items_per_chunk]
            chunk_id = hashlib.md5(f"list_{i}_{total_items}".encode()).hexdigest()
            
            chunk = ChunkInfo(
                chunk_id=chunk_id,
                data=chunk_data,
                size=len(chunk_data),
                index=i // items_per_chunk,
                total_chunks=total_chunks,
                metadata={"type": "list", "items_per_chunk": items_per_chunk}
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_dict_data(self, data: Dict[str, Any], chunk_size: int) -> List[ChunkInfo]:
        """Chunk dictionary data."""
        items = list(data.items())
        return await self._chunk_list_data(items, chunk_size)
    
    async def _chunk_dynamic_size(self, data: Any, max_chunk_size: int) -> List[ChunkInfo]:
        """Chunk data with dynamic size based on content."""
        # Implementation for dynamic chunking based on content analysis
        # This would analyze the data structure and create optimal chunks
        return await self._chunk_fixed_size(data, max_chunk_size)
    
    async def _chunk_content_based(self, data: Any) -> List[ChunkInfo]:
        """Chunk data based on content boundaries."""
        # Implementation for content-based chunking
        # This would chunk based on natural boundaries like paragraphs, objects, etc.
        return await self._chunk_fixed_size(data, self.config.chunk_size)

class StreamingDataLoader:
    """
    Streaming data loader for large datasets.
    """
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.stream_cache = {}
        self.active_streams = {}
        self._lock = asyncio.Lock()
    
    async def create_stream(
        self, 
        stream_id: str, 
        data_source: Callable,
        chunk_size: int = None
    ) -> str:
        """Create a new data stream."""
        chunk_size = chunk_size or self.config.chunk_size
        
        async with self._lock:
            if stream_id in self.active_streams:
                raise ValueError(f"Stream {stream_id} already exists")
            
            self.active_streams[stream_id] = {
                "data_source": data_source,
                "chunk_size": chunk_size,
                "created_at": time.time(),
                "total_chunks": 0,
                "consumed_chunks": 0
            }
        
        logger.info(f"Created stream: {stream_id}")
        return stream_id
    
    async def get_stream_chunk(self, stream_id: str, chunk_index: int = None) -> Optional[ChunkInfo]:
        """Get next chunk from stream."""
        async with self._lock:
            if stream_id not in self.active_streams:
                return None
            
            stream_info = self.active_streams[stream_id]
            
            if chunk_index is None:
                chunk_index = stream_info["consumed_chunks"]
            
            # Check cache first
            cache_key = f"{stream_id}_{chunk_index}"
            if cache_key in self.stream_cache:
                stream_info["consumed_chunks"] = chunk_index + 1
                return self.stream_cache[cache_key]
        
        # Load chunk from data source
        try:
            chunk = await self._load_stream_chunk(stream_id, chunk_index)
            if chunk:
                async with self._lock:
                    self.stream_cache[cache_key] = chunk
                    stream_info["consumed_chunks"] = chunk_index + 1
                    stream_info["total_chunks"] = max(stream_info["total_chunks"], chunk_index + 1)
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error loading stream chunk {chunk_index} from {stream_id}: {e}")
            return None
    
    async def _load_stream_chunk(self, stream_id: str, chunk_index: int) -> Optional[ChunkInfo]:
        """Load chunk from data source."""
        stream_info = self.active_streams[stream_id]
        data_source = stream_info["data_source"]
        chunk_size = stream_info["chunk_size"]
        
        # Call data source to get chunk
        if asyncio.iscoroutinefunction(data_source):
            data = await data_source(chunk_index, chunk_size)
        else:
            data = data_source(chunk_index, chunk_size)
        
        if data is None:
            return None
        
        chunk_id = hashlib.md5(f"{stream_id}_{chunk_index}".encode()).hexdigest()
        
        return ChunkInfo(
            chunk_id=chunk_id,
            data=data,
            size=len(data) if hasattr(data, '__len__') else 0,
            index=chunk_index,
            total_chunks=-1,  # Unknown for streams
            metadata={"type": "stream", "stream_id": stream_id}
        )
    
    async def close_stream(self, stream_id: str) -> bool:
        """Close a data stream."""
        async with self._lock:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
                
                # Clean up cache entries for this stream
                keys_to_remove = [k for k in self.stream_cache.keys() if k.startswith(f"{stream_id}_")]
                for key in keys_to_remove:
                    del self.stream_cache[key]
                
                logger.info(f"Closed stream: {stream_id}")
                return True
        
        return False
    
    async def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a stream."""
        async with self._lock:
            if stream_id in self.active_streams:
                info = self.active_streams[stream_id].copy()
                info["stream_id"] = stream_id
                return info
        
        return None

class PaginatedDataLoader:
    """
    Paginated data loader for large datasets.
    """
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.page_cache = {}
        self._lock = asyncio.Lock()
    
    async def load_page(
        self, 
        data_source: Callable,
        page: int,
        page_size: int,
        cache_key: str = None
    ) -> Dict[str, Any]:
        """Load a specific page of data."""
        cache_key = cache_key or f"page_{page}_{page_size}"
        
        # Check cache first
        async with self._lock:
            if cache_key in self.page_cache:
                cached_data = self.page_cache[cache_key]
                if time.time() - cached_data["timestamp"] < self.config.cache_ttl:
                    return cached_data["data"]
                else:
                    del self.page_cache[cache_key]
        
        # Load page from data source
        try:
            if asyncio.iscoroutinefunction(data_source):
                page_data = await data_source(page, page_size)
            else:
                page_data = data_source(page, page_size)
            
            result = {
                "data": page_data,
                "page": page,
                "page_size": page_size,
                "has_next": len(page_data) == page_size,
                "has_previous": page > 0,
                "total_pages": -1,  # Unknown without total count
                "timestamp": time.time()
            }
            
            # Cache result
            async with self._lock:
                if len(self.page_cache) >= self.config.cache_size:
                    # Remove oldest entry
                    oldest_key = min(
                        self.page_cache.keys(),
                        key=lambda k: self.page_cache[k]["timestamp"]
                    )
                    del self.page_cache[oldest_key]
                
                self.page_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading page {page}: {e}")
            return {
                "data": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 0,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def load_pages(
        self,
        data_source: Callable,
        start_page: int,
        end_page: int,
        page_size: int
    ) -> List[Dict[str, Any]]:
        """Load multiple pages concurrently."""
        tasks = [
            self.load_page(data_source, page, page_size)
            for page in range(start_page, end_page + 1)
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)

class BackgroundLoader:
    """
    Background data loader for prefetching and caching.
    """
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.background_tasks = {}
        self.prefetch_queue = deque()
        self.worker_semaphore = asyncio.Semaphore(config.background_workers)
        self._running = False
        self._lock = asyncio.Lock()
    
    async def start(self) -> Any:
        """Start background loader."""
        if self._running:
            return
        
        self._running = True
        asyncio.create_task(self._worker_loop())
        logger.info("Background loader started")
    
    async def stop(self) -> Any:
        """Stop background loader."""
        self._running = False
        
        # Cancel all background tasks
        async with self._lock:
            for task in self.background_tasks.values():
                task.cancel()
            self.background_tasks.clear()
        
        logger.info("Background loader stopped")
    
    async async def prefetch_data(
        self, 
        data_id: str, 
        loader_func: Callable,
        priority: int = 1
    ) -> bool:
        """Schedule data for background prefetching."""
        if not self.config.prefetch_enabled:
            return False
        
        async with self._lock:
            if data_id in self.background_tasks:
                return False  # Already scheduled
            
            self.prefetch_queue.append((priority, data_id, loader_func))
            self.prefetch_queue = deque(sorted(self.prefetch_queue, key=lambda x: x[0], reverse=True))
        
        return True
    
    async def _worker_loop(self) -> Any:
        """Background worker loop."""
        while self._running:
            try:
                # Get next prefetch task
                async with self._lock:
                    if not self.prefetch_queue:
                        await asyncio.sleep(0.1)
                        continue
                    
                    priority, data_id, loader_func = self.prefetch_queue.popleft()
                
                # Execute prefetch task
                async with self.worker_semaphore:
                    task = asyncio.create_task(self._execute_prefetch(data_id, loader_func))
                    self.background_tasks[data_id] = task
                
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                
            except Exception as e:
                logger.error(f"Background worker error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_prefetch(self, data_id: str, loader_func: Callable):
        """Execute a prefetch task."""
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(loader_func):
                result = await loader_func()
            else:
                result = loader_func()
            
            execution_time = time.time() - start_time
            logger.debug(f"Prefetched {data_id} in {execution_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Prefetch error for {data_id}: {e}")
        finally:
            async with self._lock:
                self.background_tasks.pop(data_id, None)

class AdvancedLazyLoader:
    """
    Advanced lazy loading system for large datasets and API responses.
    """
    
    def __init__(self, config: LazyLoadingConfig = None):
        
    """__init__ function."""
self.config = config or LazyLoadingConfig()
        self.metrics = LoadingMetrics()
        self.chunker = DataChunker(self.config)
        self.streaming_loader = StreamingDataLoader(self.config)
        self.paginated_loader = PaginatedDataLoader(self.config)
        self.background_loader = BackgroundLoader(self.config)
        self.data_cache = {}
        self._lock = asyncio.Lock()
        self._running = False
    
    async def initialize(self) -> Any:
        """Initialize the lazy loading system."""
        await self.background_loader.start()
        self._running = True
        logger.info("Advanced lazy loader initialized")
    
    async def cleanup(self) -> Any:
        """Cleanup the lazy loading system."""
        self._running = False
        await self.background_loader.stop()
        
        # Clear caches
        async with self._lock:
            self.data_cache.clear()
            self.streaming_loader.stream_cache.clear()
            self.paginated_loader.page_cache.clear()
        
        logger.info("Advanced lazy loader cleaned up")
    
    async def load_data(
        self,
        data_id: str,
        loader_func: Callable,
        strategy: LoadingStrategy = None,
        **kwargs
    ) -> Any:
        """Load data using specified strategy."""
        strategy = strategy or self.config.default_strategy
        start_time = time.time()
        
        try:
            if strategy == LoadingStrategy.EAGER:
                result = await self._load_eager(data_id, loader_func, **kwargs)
            elif strategy == LoadingStrategy.LAZY:
                result = await self._load_lazy(data_id, loader_func, **kwargs)
            elif strategy == LoadingStrategy.STREAMING:
                result = await self._load_streaming(data_id, loader_func, **kwargs)
            elif strategy == LoadingStrategy.CHUNKED:
                result = await self._load_chunked(data_id, loader_func, **kwargs)
            elif strategy == LoadingStrategy.PAGINATED:
                result = await self._load_paginated(data_id, loader_func, **kwargs)
            elif strategy == LoadingStrategy.BACKGROUND:
                result = await self._load_background(data_id, loader_func, **kwargs)
            else:
                result = await self._load_hybrid(data_id, loader_func, **kwargs)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(strategy, execution_time)
            
            return result
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Error loading data {data_id}: {e}")
            raise
    
    async def _load_eager(self, data_id: str, loader_func: Callable, **kwargs) -> Any:
        """Load data eagerly (immediately)."""
        # Check cache first
        async with self._lock:
            if data_id in self.data_cache:
                cached_data = self.data_cache[data_id]
                if time.time() - cached_data["timestamp"] < self.config.cache_ttl:
                    self.metrics.cache_hits += 1
                    return cached_data["data"]
                else:
                    del self.data_cache[data_id]
        
        # Load data
        if asyncio.iscoroutinefunction(loader_func):
            data = await loader_func(**kwargs)
        else:
            data = loader_func(**kwargs)
        
        # Cache result
        async with self._lock:
            if len(self.data_cache) >= self.config.cache_size:
                # Remove oldest entry
                oldest_key = min(
                    self.data_cache.keys(),
                    key=lambda k: self.data_cache[k]["timestamp"]
                )
                del self.data_cache[oldest_key]
            
            self.data_cache[data_id] = {
                "data": data,
                "timestamp": time.time()
            }
        
        return data
    
    async def _load_lazy(self, data_id: str, loader_func: Callable, **kwargs) -> Any:
        """Load data lazily (on first access)."""
        # Create a lazy proxy object
        class LazyProxy:
            def __init__(self, loader, data_id, kwargs) -> Any:
                self.loader = loader
                self.data_id = data_id
                self.kwargs = kwargs
                self._data = None
                self._loaded = False
            
            async def __call__(self) -> Any:
                if not self._loaded:
                    self._data = await self.loader._load_eager(self.data_id, self.loader_func, **self.kwargs)
                    self._loaded = True
                return self._data
            
            def __getattr__(self, name) -> Optional[Dict[str, Any]]:
                if not self._loaded:
                    raise RuntimeError(f"Data {self.data_id} not loaded yet")
                return getattr(self._data, name)
        
        return LazyProxy(self, data_id, kwargs)
    
    async def _load_streaming(self, data_id: str, loader_func: Callable, **kwargs) -> str:
        """Load data as a stream."""
        return await self.streaming_loader.create_stream(data_id, loader_func, **kwargs)
    
    async def _load_chunked(self, data_id: str, loader_func: Callable, **kwargs) -> List[ChunkInfo]:
        """Load data in chunks."""
        # Load full data first
        if asyncio.iscoroutinefunction(loader_func):
            data = await loader_func(**kwargs)
        else:
            data = loader_func(**kwargs)
        
        # Chunk the data
        chunk_size = kwargs.get('chunk_size', self.config.chunk_size)
        strategy = kwargs.get('chunk_strategy', ChunkStrategy.FIXED_SIZE)
        
        return await self.chunker.chunk_data(data, strategy, chunk_size)
    
    async def _load_paginated(self, data_id: str, loader_func: Callable, **kwargs) -> Dict[str, Any]:
        """Load data with pagination."""
        page = kwargs.get('page', 0)
        page_size = kwargs.get('page_size', 100)
        
        return await self.paginated_loader.load_page(loader_func, page, page_size, data_id)
    
    async def _load_background(self, data_id: str, loader_func: Callable, **kwargs) -> str:
        """Load data in background."""
        await self.background_loader.prefetch_data(data_id, loader_func)
        return data_id
    
    async def _load_hybrid(self, data_id: str, loader_func: Callable, **kwargs) -> Any:
        """Load data using hybrid strategy."""
        # Start background loading
        await self.background_loader.prefetch_data(data_id, loader_func)
        
        # Return a proxy that can be used immediately
        return await self._load_lazy(data_id, loader_func, **kwargs)
    
    def _update_metrics(self, strategy: LoadingStrategy, execution_time: float):
        """Update loading metrics."""
        self.metrics.total_operations += 1
        self.metrics.total_load_time += execution_time
        self.metrics.average_load_time = self.metrics.total_load_time / self.metrics.total_operations
        
        if strategy == LoadingStrategy.STREAMING:
            self.metrics.streaming_operations += 1
        elif strategy == LoadingStrategy.CHUNKED:
            self.metrics.chunked_operations += 1
        elif strategy == LoadingStrategy.PAGINATED:
            self.metrics.paginated_operations += 1
        elif strategy == LoadingStrategy.BACKGROUND:
            self.metrics.background_operations += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return {
            "total_operations": self.metrics.total_operations,
            "streaming_operations": self.metrics.streaming_operations,
            "chunked_operations": self.metrics.chunked_operations,
            "paginated_operations": self.metrics.paginated_operations,
            "background_operations": self.metrics.background_operations,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.total_operations - self.metrics.cache_hits,
            "cache_hit_rate": self.metrics.cache_hits / self.metrics.total_operations if self.metrics.total_operations > 0 else 0,
            "total_load_time": self.metrics.total_load_time,
            "average_load_time": self.metrics.average_load_time,
            "memory_usage": self.metrics.memory_usage,
            "errors": self.metrics.errors,
            "cache_sizes": {
                "data_cache": len(self.data_cache),
                "stream_cache": len(self.streaming_loader.stream_cache),
                "page_cache": len(self.paginated_loader.page_cache)
            }
        }

# Decorators for easy lazy loading
def lazy_load(strategy: LoadingStrategy = LoadingStrategy.LAZY, **kwargs):
    """Decorator for lazy loading functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **func_kwargs) -> Any:
            # Merge kwargs
            merged_kwargs = {**kwargs, **func_kwargs}
            
            # Create lazy loader if not provided
            lazy_loader = merged_kwargs.pop('lazy_loader', None)
            if lazy_loader is None:
                config = LazyLoadingConfig()
                lazy_loader = AdvancedLazyLoader(config)
                await lazy_loader.initialize()
            
            # Generate data ID
            data_id = hashlib.md5(f"{func.__name__}_{args}_{func_kwargs}".encode()).hexdigest()
            
            # Load data
            return await lazy_loader.load_data(data_id, func, strategy, **merged_kwargs)
        
        return wrapper
    return decorator

def streaming_load(chunk_size: int = None):
    """Decorator for streaming data loading."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            config = LazyLoadingConfig()
            lazy_loader = AdvancedLazyLoader(config)
            await lazy_loader.initialize()
            
            data_id = hashlib.md5(f"{func.__name__}_{args}_{kwargs}".encode()).hexdigest()
            
            return await lazy_loader.load_data(
                data_id, func, LoadingStrategy.STREAMING, chunk_size=chunk_size, **kwargs
            )
        
        return wrapper
    return decorator

def chunked_load(chunk_size: int = None, strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE):
    """Decorator for chunked data loading."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            config = LazyLoadingConfig()
            lazy_loader = AdvancedLazyLoader(config)
            await lazy_loader.initialize()
            
            data_id = hashlib.md5(f"{func.__name__}_{args}_{kwargs}".encode()).hexdigest()
            
            return await lazy_loader.load_data(
                data_id, func, LoadingStrategy.CHUNKED, 
                chunk_size=chunk_size, chunk_strategy=strategy, **kwargs
            )
        
        return wrapper
    return decorator

# Example usage
async def example_lazy_loading_usage():
    """Example usage of advanced lazy loading system."""
    
    # Create configuration
    config = LazyLoadingConfig(
        default_strategy=LoadingStrategy.LAZY,
        enable_streaming=True,
        enable_chunking=True,
        enable_pagination=True,
        enable_background_loading=True,
        max_memory_mb=512,
        chunk_size=1024 * 1024,  # 1MB chunks
        prefetch_enabled=True
    )
    
    # Initialize lazy loader
    lazy_loader = AdvancedLazyLoader(config)
    await lazy_loader.initialize()
    
    # Example data loader functions
    async def load_large_dataset():
        """Simulate loading a large dataset."""
        await asyncio.sleep(2)  # Simulate slow loading
        return [f"data_item_{i}" for i in range(10000)]
    
    async def stream_data(chunk_index: int, chunk_size: int):
        """Simulate streaming data."""
        await asyncio.sleep(0.1)  # Simulate processing time
        start = chunk_index * chunk_size
        end = start + chunk_size
        return [f"stream_item_{i}" for i in range(start, min(end, 1000))]
    
    async def load_paginated_data(page: int, page_size: int):
        """Simulate paginated data loading."""
        await asyncio.sleep(0.5)  # Simulate database query
        start = page * page_size
        end = start + page_size
        return [f"page_item_{i}" for i in range(start, min(end, 1000))]
    
    try:
        # Test different loading strategies
        
        # 1. Lazy loading
        logger.info("Testing lazy loading...")
        lazy_data = await lazy_loader.load_data("lazy_test", load_large_dataset, LoadingStrategy.LAZY)
        logger.info(f"Lazy data loaded: {type(lazy_data)}")
        
        # 2. Streaming loading
        logger.info("Testing streaming loading...")
        stream_id = await lazy_loader.load_data("stream_test", stream_data, LoadingStrategy.STREAMING)
        logger.info(f"Stream created: {stream_id}")
        
        # Get stream chunks
        for i in range(5):
            chunk = await lazy_loader.streaming_loader.get_stream_chunk(stream_id, i)
            if chunk:
                logger.info(f"Stream chunk {i}: {len(chunk.data)} items")
        
        # 3. Chunked loading
        logger.info("Testing chunked loading...")
        chunks = await lazy_loader.load_data("chunk_test", load_large_dataset, LoadingStrategy.CHUNKED)
        logger.info(f"Data chunked into {len(chunks)} chunks")
        
        # 4. Paginated loading
        logger.info("Testing paginated loading...")
        page_data = await lazy_loader.load_data("page_test", load_paginated_data, LoadingStrategy.PAGINATED, page=0, page_size=100)
        logger.info(f"Page data: {len(page_data['data'])} items")
        
        # 5. Background loading
        logger.info("Testing background loading...")
        bg_data_id = await lazy_loader.load_data("bg_test", load_large_dataset, LoadingStrategy.BACKGROUND)
        logger.info(f"Background loading started: {bg_data_id}")
        
        # Get metrics
        metrics = lazy_loader.get_metrics()
        logger.info(f"Lazy loading metrics: {metrics}")
        
        # Cleanup
        await lazy_loader.streaming_loader.close_stream(stream_id)
        await lazy_loader.cleanup()
        
    except Exception as e:
        logger.error(f"Lazy loading error: {e}")
        await lazy_loader.cleanup()

match __name__:
    case "__main__":
    asyncio.run(example_lazy_loading_usage()) 