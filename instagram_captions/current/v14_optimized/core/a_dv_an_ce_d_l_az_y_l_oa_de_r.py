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
import weakref
import threading
import json
import gzip
import pickle
from typing import (
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
from functools import wraps
import gc
import psutil
import os
from pathlib import Path
import tempfile
import shutil
    import numba
    from numba import jit, njit
from fastapi import Response
from fastapi.responses import StreamingResponse
import aiofiles
from typing import Any, List, Dict, Optional
"""
Advanced Lazy Loader for Large Datasets and API Responses v14.0

Specialized lazy loading for:
- Large dataset streaming
- Substantial API response handling
- Memory-efficient pagination
- Chunked data loading
- Background prefetching
- Resource pooling for large objects
- Streaming response generation
"""

    Dict, Any, List, Optional, Callable, TypeVar, Generic, Union, Set,
    AsyncIterator, Iterator, Tuple, Protocol
)

# Performance libraries
try:
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs) -> Any:
        def decorator(func) -> Any:
            return func
        return decorator
    njit = jit

# FastAPI streaming

logger = logging.getLogger(__name__)

# Type variables for generic operations
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class DataSize(Enum):
    """Data size categories for optimization"""
    SMALL = "small"        # < 1MB
    MEDIUM = "medium"      # 1MB - 10MB
    LARGE = "large"        # 10MB - 100MB
    HUGE = "huge"          # > 100MB


class LoadStrategy(Enum):
    """Loading strategies for different data sizes"""
    IMMEDIATE = "immediate"      # Load all at once
    STREAMING = "streaming"      # Stream data in chunks
    PAGINATED = "paginated"      # Load page by page
    BACKGROUND = "background"    # Load in background
    ON_DEMAND = "on_demand"      # Load when accessed


class CompressionType(Enum):
    """Compression types for data storage"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class ChunkConfig:
    """Configuration for chunked loading"""
    chunk_size: int = 1000
    max_chunks_in_memory: int = 10
    prefetch_chunks: int = 3
    compression_threshold: int = 1024  # Compress chunks larger than 1KB
    compression_type: CompressionType = CompressionType.GZIP


@dataclass
class StreamingConfig:
    """Configuration for streaming responses"""
    buffer_size: int = 8192
    chunk_timeout: float = 30.0
    max_concurrent_streams: int = 100
    enable_compression: bool = True
    compression_level: int = 6


@dataclass
class PaginationConfig:
    """Configuration for pagination"""
    default_page_size: int = 50
    max_page_size: int = 1000
    enable_cursor_pagination: bool = True
    cursor_field: str = "id"
    sort_field: str = "created_at"
    sort_direction: str = "desc"


@dataclass
class LargeDataConfig:
    """Configuration for large dataset handling"""
    # Memory management
    max_memory_mb: int = 2048
    memory_threshold: float = 0.75
    enable_disk_cache: bool = True
    disk_cache_dir: str = "/tmp/lazy_loader_cache"
    
    # Chunking
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)
    streaming_config: StreamingConfig = field(default_factory=StreamingConfig)
    pagination_config: PaginationConfig = field(default_factory=PaginationConfig)
    
    # Performance
    enable_monitoring: bool = True
    enable_metrics: bool = True
    load_timeout: float = 60.0
    retry_attempts: int = 3
    
    # Background processing
    enable_background_loading: bool = True
    max_background_tasks: int = 20
    background_batch_size: int = 100


@dataclass
class DataChunk:
    """Represents a chunk of data"""
    id: str
    data: Any
    size: int
    compressed: bool = False
    compression_type: CompressionType = CompressionType.NONE
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class PageInfo:
    """Pagination information"""
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool
    cursor: Optional[str] = None
    next_cursor: Optional[str] = None
    previous_cursor: Optional[str] = None


class DataLoader(Protocol[T]):
    """Protocol for data loading functions"""
    async def __call__(self, offset: int = 0, limit: int = None, **kwargs) -> List[T]:
        ...


class ChunkedDataLoader(Protocol[T]):
    """Protocol for chunked data loading"""
    async def __call__(self, chunk_id: str, **kwargs) -> DataChunk:
        ...


class AdvancedLazyLoader(Generic[T]):
    """Advanced lazy loader for large datasets and API responses"""
    
    def __init__(self, config: LargeDataConfig):
        
    """__init__ function."""
self.config = config
        self.chunks: Dict[str, DataChunk] = {}
        self.pages: Dict[str, List[T]] = {}
        self.streaming_sessions: Dict[str, asyncio.Task] = {}
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Thread safety
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        
        # Disk cache
        if config.enable_disk_cache:
            self._setup_disk_cache()
        
        # Statistics
        self.stats = {
            "total_chunks_loaded": 0,
            "total_pages_loaded": 0,
            "total_streams": 0,
            "cache_hits": 0,
            "disk_cache_hits": 0,
            "compression_savings": 0,
            "memory_usage_mb": 0.0,
            "avg_chunk_load_time": 0.0,
            "avg_page_load_time": 0.0
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _setup_disk_cache(self) -> Any:
        """Setup disk cache directory"""
        self.disk_cache_dir = Path(self.config.disk_cache_dir)
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Disk cache initialized at {self.disk_cache_dir}")
    
    def _start_background_tasks(self) -> Any:
        """Start background maintenance tasks"""
        if self.config.enable_monitoring:
            asyncio.create_task(self._monitor_memory())
        if self.config.enable_background_loading:
            asyncio.create_task(self._background_cleanup())
    
    async def load_chunked_data(
        self,
        loader_func: ChunkedDataLoader[T],
        chunk_ids: List[str],
        strategy: LoadStrategy = LoadStrategy.STREAMING
    ) -> AsyncIterator[DataChunk]:
        """Load data in chunks with streaming"""
        
        if strategy == LoadStrategy.STREAMING:
            async for chunk in self._stream_chunks(loader_func, chunk_ids):
                yield chunk
        elif strategy == LoadStrategy.BACKGROUND:
            async for chunk in self._background_load_chunks(loader_func, chunk_ids):
                yield chunk
        else:
            # Load all chunks immediately
            for chunk_id in chunk_ids:
                chunk = await self._load_chunk(loader_func, chunk_id)
                yield chunk
    
    async def _stream_chunks(
        self,
        loader_func: ChunkedDataLoader[T],
        chunk_ids: List[str]
    ) -> AsyncIterator[DataChunk]:
        """Stream chunks with prefetching"""
        
        # Start prefetching
        prefetch_tasks = []
        for i, chunk_id in enumerate(chunk_ids[:self.config.chunk_config.prefetch_chunks]):
            task = asyncio.create_task(self._load_chunk(loader_func, chunk_id))
            prefetch_tasks.append((chunk_id, task))
        
        # Stream chunks
        for i, chunk_id in enumerate(chunk_ids):
            # Wait for current chunk
            if i < len(prefetch_tasks):
                chunk = await prefetch_tasks[i][1]
            else:
                chunk = await self._load_chunk(loader_func, chunk_id)
            
            yield chunk
            
            # Start prefetching next chunk
            next_index = i + self.config.chunk_config.prefetch_chunks
            if next_index < len(chunk_ids):
                next_chunk_id = chunk_ids[next_index]
                task = asyncio.create_task(self._load_chunk(loader_func, next_chunk_id))
                prefetch_tasks.append((next_chunk_id, task))
    
    async def _load_chunk(
        self,
        loader_func: ChunkedDataLoader[T],
        chunk_id: str
    ) -> DataChunk:
        """Load a single chunk with caching"""
        
        # Check memory cache
        if chunk_id in self.chunks:
            self.chunks[chunk_id].access_count += 1
            self.chunks[chunk_id].accessed_at = time.time()
            self.stats["cache_hits"] += 1
            return self.chunks[chunk_id]
        
        # Check disk cache
        if self.config.enable_disk_cache:
            disk_chunk = await self._load_from_disk_cache(chunk_id)
            if disk_chunk:
                self.chunks[chunk_id] = disk_chunk
                self.stats["disk_cache_hits"] += 1
                return disk_chunk
        
        # Load from source
        start_time = time.time()
        chunk = await loader_func(chunk_id)
        
        # Compress if needed
        if chunk.size > self.config.chunk_config.compression_threshold:
            chunk = await self._compress_chunk(chunk)
        
        # Cache in memory
        self.chunks[chunk_id] = chunk
        
        # Cache to disk if enabled
        if self.config.enable_disk_cache:
            await self._save_to_disk_cache(chunk_id, chunk)
        
        # Update stats
        load_time = time.time() - start_time
        self.stats["total_chunks_loaded"] += 1
        self.stats["avg_chunk_load_time"] = (
            (self.stats["avg_chunk_load_time"] * (self.stats["total_chunks_loaded"] - 1) + 
             load_time) / self.stats["total_chunks_loaded"]
        )
        
        return chunk
    
    async def _compress_chunk(self, chunk: DataChunk) -> DataChunk:
        """Compress chunk data"""
        if chunk.compressed:
            return chunk
        
        if self.config.chunk_config.compression_type == CompressionType.GZIP:
            compressed_data = gzip.compress(pickle.dumps(chunk.data))
            savings = len(pickle.dumps(chunk.data)) - len(compressed_data)
            self.stats["compression_savings"] += savings
            
            return DataChunk(
                id=chunk.id,
                data=compressed_data,
                size=len(compressed_data),
                compressed=True,
                compression_type=CompressionType.GZIP,
                created_at=chunk.created_at,
                accessed_at=chunk.accessed_at,
                access_count=chunk.access_count
            )
        
        return chunk
    
    async def _decompress_chunk(self, chunk: DataChunk) -> DataChunk:
        """Decompress chunk data"""
        if not chunk.compressed:
            return chunk
        
        if chunk.compression_type == CompressionType.GZIP:
            decompressed_data = pickle.loads(gzip.decompress(chunk.data))
            
            return DataChunk(
                id=chunk.id,
                data=decompressed_data,
                size=len(pickle.dumps(decompressed_data)),
                compressed=False,
                compression_type=CompressionType.NONE,
                created_at=chunk.created_at,
                accessed_at=chunk.accessed_at,
                access_count=chunk.access_count
            )
        
        return chunk
    
    async def _save_to_disk_cache(self, chunk_id: str, chunk: DataChunk):
        """Save chunk to disk cache"""
        try:
            cache_file = self.disk_cache_dir / f"{chunk_id}.cache"
            async with aiofiles.open(cache_file, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                await f.write(pickle.dumps(chunk))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except Exception as e:
            logger.warning(f"Failed to save chunk {chunk_id} to disk cache: {e}")
    
    async def _load_from_disk_cache(self, chunk_id: str) -> Optional[DataChunk]:
        """Load chunk from disk cache"""
        try:
            cache_file = self.disk_cache_dir / f"{chunk_id}.cache"
            if cache_file.exists():
                async with aiofiles.open(cache_file, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Failed to load chunk {chunk_id} from disk cache: {e}")
        
        return None
    
    async def load_paginated_data(
        self,
        loader_func: DataLoader[T],
        page: int = 1,
        page_size: int = None,
        **kwargs
    ) -> Tuple[List[T], PageInfo]:
        """Load data with pagination"""
        
        page_size = page_size or self.config.pagination_config.default_page_size
        page_size = min(page_size, self.config.pagination_config.max_page_size)
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Generate cache key
        cache_key = f"page_{page}_{page_size}_{hash(str(kwargs))}"
        
        # Check cache
        if cache_key in self.pages:
            self.stats["cache_hits"] += 1
            return self.pages[cache_key], self._create_page_info(page, page_size, len(self.pages[cache_key]))
        
        # Load data
        start_time = time.time()
        data = await loader_func(offset=offset, limit=page_size, **kwargs)
        
        # Cache result
        self.pages[cache_key] = data
        
        # Update stats
        load_time = time.time() - start_time
        self.stats["total_pages_loaded"] += 1
        self.stats["avg_page_load_time"] = (
            (self.stats["avg_page_load_time"] * (self.stats["total_pages_loaded"] - 1) + 
             load_time) / self.stats["total_pages_loaded"]
        )
        
        return data, self._create_page_info(page, page_size, len(data))
    
    def _create_page_info(self, page: int, page_size: int, item_count: int) -> PageInfo:
        """Create pagination information"""
        return PageInfo(
            page=page,
            page_size=page_size,
            total_items=item_count,
            total_pages=(item_count + page_size - 1) // page_size,
            has_next=item_count == page_size,
            has_previous=page > 1
        )
    
    async def create_streaming_response(
        self,
        data_iterator: AsyncIterator[T],
        response_format: str = "json",
        enable_compression: bool = True
    ) -> StreamingResponse:
        """Create a streaming response for large datasets"""
        
        async def stream_generator():
            
    """stream_generator function."""
if response_format == "json":
                yield "["
                first = True
                async for item in data_iterator:
                    if not first:
                        yield ","
                    yield json.dumps(item)
                    first = False
                yield "]"
            elif response_format == "csv":
                # Add CSV header if needed
                yield "id,data,created_at\n"
                async for item in data_iterator:
                    yield f"{getattr(item, 'id', '')},{getattr(item, 'data', '')},{getattr(item, 'created_at', '')}\n"
            else:
                async for item in data_iterator:
                    yield str(item) + "\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type=f"application/{response_format}",
            headers={
                "Content-Encoding": "gzip" if enable_compression else "identity",
                "Cache-Control": "no-cache",
                "X-Streaming": "true"
            }
        )
    
    async def batch_process_large_dataset(
        self,
        dataset_loader: DataLoader[T],
        batch_size: int = 1000,
        processor_func: Callable[[List[T]], Any] = None
    ) -> AsyncIterator[Any]:
        """Process large datasets in batches"""
        
        offset = 0
        while True:
            # Load batch
            batch = await dataset_loader(offset=offset, limit=batch_size)
            
            if not batch:
                break
            
            # Process batch
            if processor_func:
                result = await processor_func(batch)
                yield result
            else:
                yield batch
            
            offset += batch_size
            
            # Check memory usage
            if self._get_memory_usage() > self.config.memory_threshold:
                await self._cleanup_memory()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
    
    async def _cleanup_memory(self) -> Any:
        """Clean up memory by removing least used chunks"""
        if len(self.chunks) <= self.config.chunk_config.max_chunks_in_memory:
            return
        
        # Sort chunks by access count and last access time
        sorted_chunks = sorted(
            self.chunks.items(),
            key=lambda x: (x[1].access_count, x[1].accessed_at)
        )
        
        # Remove least used chunks
        chunks_to_remove = len(sorted_chunks) - self.config.chunk_config.max_chunks_in_memory
        for i in range(chunks_to_remove):
            chunk_id, chunk = sorted_chunks[i]
            del self.chunks[chunk_id]
            logger.debug(f"Removed chunk {chunk_id} from memory")
    
    async def _monitor_memory(self) -> Any:
        """Monitor memory usage and cleanup when needed"""
        while True:
            try:
                memory_usage = self._get_memory_usage()
                self.stats["memory_usage_mb"] = memory_usage * 1024  # Convert to MB
                
                if memory_usage > self.config.memory_threshold:
                    await self._cleanup_memory()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _background_cleanup(self) -> Any:
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                # Clean up old chunks
                current_time = time.time()
                chunks_to_remove = []
                
                for chunk_id, chunk in self.chunks.items():
                    if (current_time - chunk.accessed_at) > 3600:  # 1 hour
                        chunks_to_remove.append(chunk_id)
                
                for chunk_id in chunks_to_remove:
                    del self.chunks[chunk_id]
                
                # Clean up old pages
                pages_to_remove = []
                for page_key in self.pages:
                    if len(self.pages[page_key]) == 0:
                        pages_to_remove.append(page_key)
                
                for page_key in pages_to_remove:
                    del self.pages[page_key]
                
                logger.debug(f"Background cleanup: removed {len(chunks_to_remove)} chunks, {len(pages_to_remove)} pages")
                
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics"""
        return {
            **self.stats,
            "active_chunks": len(self.chunks),
            "active_pages": len(self.pages),
            "active_streams": len(self.streaming_sessions),
            "background_tasks": len(self.background_tasks),
            "disk_cache_size_mb": await self._get_disk_cache_size()
        }
    
    async def _get_disk_cache_size(self) -> float:
        """Get disk cache size in MB"""
        if not self.config.enable_disk_cache:
            return 0.0
        
        try:
            total_size = 0
            for file_path in self.disk_cache_dir.rglob("*.cache"):
                total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    async def clear_cache(self) -> Any:
        """Clear all caches"""
        self.chunks.clear()
        self.pages.clear()
        
        if self.config.enable_disk_cache:
            try:
                shutil.rmtree(self.disk_cache_dir)
                self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")


# Utility functions for lazy loading
def lazy_load_large_dataset(loader_func: DataLoader[T]):
    """Decorator for lazy loading large datasets"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract pagination parameters
            page = kwargs.get('page', 1)
            page_size = kwargs.get('page_size', 50)
            
            # Load data lazily
            data, page_info = await loader_func(page=page, page_size=page_size, **kwargs)
            
            return func(*args, data=data, page_info=page_info, **kwargs)
        
        return wrapper
    return decorator


def stream_large_response(response_format: str = "json"):
    """Decorator for streaming large responses"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get data iterator
            data_iterator = await func(*args, **kwargs)
            
            # Create streaming response
            loader = AdvancedLazyLoader(LargeDataConfig())
            return await loader.create_streaming_response(data_iterator, response_format)
        
        return wrapper
    return decorator


@asynccontextmanager
async def large_dataset_context(loader: AdvancedLazyLoader[T]):
    """Context manager for large dataset operations"""
    try:
        yield loader
    finally:
        await loader.clear_cache()


# Factory function for creating lazy loaders
def create_lazy_loader(
    data_size: DataSize = DataSize.MEDIUM,
    enable_disk_cache: bool = True,
    max_memory_mb: int = 2048
) -> AdvancedLazyLoader:
    """Create a lazy loader with appropriate configuration"""
    
    config = LargeDataConfig(
        max_memory_mb=max_memory_mb,
        enable_disk_cache=enable_disk_cache
    )
    
    # Adjust configuration based on data size
    if data_size == DataSize.SMALL:
        config.chunk_config.chunk_size = 100
        config.chunk_config.max_chunks_in_memory = 50
    elif data_size == DataSize.MEDIUM:
        config.chunk_config.chunk_size = 1000
        config.chunk_config.max_chunks_in_memory = 20
    elif data_size == DataSize.LARGE:
        config.chunk_config.chunk_size = 5000
        config.chunk_config.max_chunks_in_memory = 10
        config.enable_disk_cache = True
    elif data_size == DataSize.HUGE:
        config.chunk_config.chunk_size = 10000
        config.chunk_config.max_chunks_in_memory = 5
        config.enable_disk_cache = True
        config.max_memory_mb = 4096
    
    return AdvancedLazyLoader(config) 