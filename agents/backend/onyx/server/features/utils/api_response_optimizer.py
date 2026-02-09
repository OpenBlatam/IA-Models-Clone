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
import json
import hashlib
from typing import Any, Optional, Dict, List, Callable, Awaitable, Union, AsyncGenerator, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import functools
import orjson
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import structlog
from .advanced_lazy_loading import (
from typing import Any, List, Dict, Optional
"""
🚀 API Response Optimizer
=========================

Advanced API response optimization with:
- Lazy loading for large responses
- Streaming responses
- Response chunking
- Progressive loading
- Memory optimization
- Background prefetching
- Response caching
"""



    AdvancedLazyLoader, LazyLoadingConfig, LoadingStrategy,
    StreamingDataLoader, PaginatedDataLoader, BackgroundLoader
)

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)

class ResponseOptimizationStrategy(Enum):
    """Response optimization strategies"""
    NONE = "none"                     # No optimization
    LAZY = "lazy"                     # Lazy loading
    STREAMING = "streaming"           # Streaming response
    CHUNKED = "chunked"               # Chunked response
    PAGINATED = "paginated"           # Paginated response
    PROGRESSIVE = "progressive"       # Progressive loading
    HYBRID = "hybrid"                 # Hybrid approach

class ResponseSize(Enum):
    """Response size categories"""
    SMALL = "small"       # < 1KB
    MEDIUM = "medium"     # 1KB - 1MB
    LARGE = "large"       # 1MB - 10MB
    HUGE = "huge"         # > 10MB

@dataclass
class ResponseOptimizationConfig:
    """Configuration for response optimization"""
    # General settings
    default_strategy: ResponseOptimizationStrategy = ResponseOptimizationStrategy.LAZY
    enable_streaming: bool = True
    enable_chunking: bool = True
    enable_pagination: bool = True
    enable_progressive_loading: bool = True
    
    # Size thresholds
    small_response_threshold: int = 1024        # 1KB
    medium_response_threshold: int = 1024 * 1024  # 1MB
    large_response_threshold: int = 10 * 1024 * 1024  # 10MB
    
    # Chunking settings
    default_chunk_size: int = 1024 * 1024       # 1MB
    max_chunks_per_response: int = 100
    chunk_timeout: float = 30.0
    
    # Streaming settings
    streaming_buffer_size: int = 8192
    streaming_timeout: float = 60.0
    enable_backpressure: bool = True
    
    # Pagination settings
    default_page_size: int = 100
    max_page_size: int = 1000
    enable_infinite_scroll: bool = True
    
    # Caching settings
    enable_response_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # Performance settings
    enable_compression: bool = True
    enable_background_prefetching: bool = True
    max_concurrent_responses: int = 50
    
    # Monitoring settings
    enable_metrics: bool = True
    log_slow_responses: bool = True
    slow_response_threshold: float = 2.0

@dataclass
class ResponseMetrics:
    """Response performance metrics"""
    total_responses: int = 0
    lazy_responses: int = 0
    streaming_responses: int = 0
    chunked_responses: int = 0
    paginated_responses: int = 0
    progressive_responses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    total_data_size: int = 0
    average_data_size: int = 0
    errors: int = 0

class ResponseSizeAnalyzer:
    """
    Analyzes response size and determines optimal strategy.
    """
    
    def __init__(self, config: ResponseOptimizationConfig):
        
    """__init__ function."""
self.config = config
    
    def analyze_response_size(self, data: Any) -> ResponseSize:
        """Analyze the size of response data."""
        if isinstance(data, (str, bytes)):
            size = len(data)
        elif isinstance(data, (list, tuple)):
            size = len(str(data))
        elif isinstance(data, dict):
            size = len(orjson.dumps(data))
        elif hasattr(data, '__len__'):
            size = len(data)
        else:
            size = len(str(data))
        
        if size < self.config.small_response_threshold:
            return ResponseSize.SMALL
        elif size < self.config.medium_response_threshold:
            return ResponseSize.MEDIUM
        elif size < self.config.large_response_threshold:
            return ResponseSize.LARGE
        else:
            return ResponseSize.HUGE
    
    def get_optimal_strategy(self, data: Any, user_preference: ResponseOptimizationStrategy = None) -> ResponseOptimizationStrategy:
        """Get optimal response strategy based on data size and user preference."""
        if user_preference and user_preference != ResponseOptimizationStrategy.NONE:
            return user_preference
        
        size = self.analyze_response_size(data)
        
        if size == ResponseSize.SMALL:
            return ResponseOptimizationStrategy.NONE
        elif size == ResponseSize.MEDIUM:
            return ResponseOptimizationStrategy.LAZY
        elif size == ResponseSize.LARGE:
            return ResponseOptimizationStrategy.CHUNKED
        else:  # HUGE
            return ResponseOptimizationStrategy.STREAMING

class StreamingResponseGenerator:
    """
    Generates streaming responses for large datasets.
    """
    
    def __init__(self, config: ResponseOptimizationConfig):
        
    """__init__ function."""
self.config = config
        self.active_streams = {}
        self._lock = asyncio.Lock()
    
    async def create_streaming_response(
        self,
        data_generator: Callable,
        response_id: str,
        chunk_size: int = None
    ) -> StreamingResponse:
        """Create a streaming response."""
        chunk_size = chunk_size or self.config.default_chunk_size
        
        async def stream_generator():
            
    """stream_generator function."""
try:
                if asyncio.iscoroutinefunction(data_generator):
                    async for chunk in data_generator():
                        yield self._format_chunk(chunk)
                else:
                    for chunk in data_generator():
                        yield self._format_chunk(chunk)
                        await asyncio.sleep(0.01)  # Prevent blocking
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield self._format_error_chunk(str(e))
        
        # Store stream info
        async with self._lock:
            self.active_streams[response_id] = {
                "created_at": time.time(),
                "chunk_size": chunk_size,
                "status": "active"
            }
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/json",
            headers={
                "X-Response-Type": "streaming",
                "X-Response-ID": response_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    def _format_chunk(self, chunk: Any) -> str:
        """Format a chunk for streaming."""
        if isinstance(chunk, (dict, list)):
            return f"data: {orjson.dumps(chunk).decode()}\n\n"
        else:
            return f"data: {json.dumps({'content': str(chunk)})}\n\n"
    
    def _format_error_chunk(self, error: str) -> str:
        """Format an error chunk."""
        return f"data: {json.dumps({'error': error})}\n\n"
    
    async def close_stream(self, response_id: str):
        """Close a streaming response."""
        async with self._lock:
            if response_id in self.active_streams:
                self.active_streams[response_id]["status"] = "closed"
                del self.active_streams[response_id]

class ChunkedResponseGenerator:
    """
    Generates chunked responses for large datasets.
    """
    
    def __init__(self, config: ResponseOptimizationConfig):
        
    """__init__ function."""
self.config = config
    
    async def create_chunked_response(
        self,
        data: Any,
        chunk_size: int = None
    ) -> Dict[str, Any]:
        """Create a chunked response."""
        chunk_size = chunk_size or self.config.default_chunk_size
        
        # Serialize data
        if isinstance(data, (dict, list)):
            serialized = orjson.dumps(data)
        else:
            serialized = str(data).encode('utf-8')
        
        # Split into chunks
        chunks = []
        total_size = len(serialized)
        total_chunks = (total_size + chunk_size - 1) // chunk_size
        
        for i in range(0, total_size, chunk_size):
            chunk_data = serialized[i:i + chunk_size]
            chunk_id = hashlib.md5(f"{i}_{total_size}".encode()).hexdigest()
            
            chunks.append({
                "chunk_id": chunk_id,
                "index": i // chunk_size,
                "data": chunk_data.decode('utf-8') if isinstance(chunk_data, bytes) else chunk_data,
                "size": len(chunk_data)
            })
        
        return {
            "type": "chunked",
            "total_chunks": total_chunks,
            "total_size": total_size,
            "chunk_size": chunk_size,
            "chunks": chunks,
            "timestamp": time.time()
        }

class PaginatedResponseGenerator:
    """
    Generates paginated responses for large datasets.
    """
    
    def __init__(self, config: ResponseOptimizationConfig):
        
    """__init__ function."""
self.config = config
    
    async def create_paginated_response(
        self,
        data: List[Any],
        page: int = 0,
        page_size: int = None,
        total_count: int = None
    ) -> Dict[str, Any]:
        """Create a paginated response."""
        page_size = page_size or self.config.default_page_size
        page_size = min(page_size, self.config.max_page_size)
        
        total_count = total_count or len(data)
        start_idx = page * page_size
        end_idx = start_idx + page_size
        
        page_data = data[start_idx:end_idx] if isinstance(data, list) else data
        
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "type": "paginated",
            "data": page_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages - 1,
                "has_previous": page > 0,
                "next_page": page + 1 if page < total_pages - 1 else None,
                "previous_page": page - 1 if page > 0 else None
            },
            "timestamp": time.time()
        }

class ProgressiveResponseGenerator:
    """
    Generates progressive responses that load data incrementally.
    """
    
    def __init__(self, config: ResponseOptimizationConfig):
        
    """__init__ function."""
self.config = config
    
    async def create_progressive_response(
        self,
        data_loader: Callable,
        stages: List[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create a progressive response."""
        stages = stages or ["initial", "enhanced", "complete"]
        
        for i, stage in enumerate(stages):
            try:
                # Load data for this stage
                if asyncio.iscoroutinefunction(data_loader):
                    stage_data = await data_loader(stage, i)
                else:
                    stage_data = data_loader(stage, i)
                
                yield {
                    "type": "progressive",
                    "stage": stage,
                    "stage_index": i,
                    "total_stages": len(stages),
                    "data": stage_data,
                    "is_complete": i == len(stages) - 1,
                    "timestamp": time.time()
                }
                
                # Add delay between stages for progressive loading effect
                if i < len(stages) - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Progressive loading error at stage {stage}: {e}")
                yield {
                    "type": "progressive",
                    "stage": stage,
                    "error": str(e),
                    "timestamp": time.time()
                }

class APIResponseOptimizer:
    """
    Main API response optimizer.
    """
    
    def __init__(self, config: ResponseOptimizationConfig = None):
        
    """__init__ function."""
self.config = config or ResponseOptimizationConfig()
        self.metrics = ResponseMetrics()
        self.size_analyzer = ResponseSizeAnalyzer(self.config)
        self.streaming_generator = StreamingResponseGenerator(self.config)
        self.chunked_generator = ChunkedResponseGenerator(self.config)
        self.paginated_generator = PaginatedResponseGenerator(self.config)
        self.progressive_generator = ProgressiveResponseGenerator(self.config)
        self.response_cache = {}
        self._lock = asyncio.Lock()
    
    async def optimize_response(
        self,
        data: Any,
        strategy: ResponseOptimizationStrategy = None,
        **kwargs
    ) -> Union[Response, Dict[str, Any]]:
        """Optimize API response based on data and strategy."""
        start_time = time.time()
        
        # Determine optimal strategy
        if strategy is None:
            strategy = self.size_analyzer.get_optimal_strategy(data)
        
        try:
            if strategy == ResponseOptimizationStrategy.NONE:
                result = await self._create_simple_response(data)
            elif strategy == ResponseOptimizationStrategy.LAZY:
                result = await self._create_lazy_response(data, **kwargs)
            elif strategy == ResponseOptimizationStrategy.STREAMING:
                result = await self._create_streaming_response(data, **kwargs)
            elif strategy == ResponseOptimizationStrategy.CHUNKED:
                result = await self._create_chunked_response(data, **kwargs)
            elif strategy == ResponseOptimizationStrategy.PAGINATED:
                result = await self._create_paginated_response(data, **kwargs)
            elif strategy == ResponseOptimizationStrategy.PROGRESSIVE:
                result = await self._create_progressive_response(data, **kwargs)
            else:
                result = await self._create_hybrid_response(data, **kwargs)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(strategy, execution_time, data)
            
            return result
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Response optimization error: {e}")
            raise
    
    async def _create_simple_response(self, data: Any) -> JSONResponse:
        """Create a simple JSON response."""
        return JSONResponse(content=data)
    
    async def _create_lazy_response(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Create a lazy loading response."""
        response_id = hashlib.md5(f"lazy_{time.time()}".encode()).hexdigest()
        
        return {
            "type": "lazy",
            "response_id": response_id,
            "status": "pending",
            "estimated_size": self.size_analyzer.analyze_response_size(data).value,
            "load_url": f"/api/responses/{response_id}/load",
            "timestamp": time.time()
        }
    
    async def _create_streaming_response(self, data: Any, **kwargs) -> StreamingResponse:
        """Create a streaming response."""
        response_id = hashlib.md5(f"stream_{time.time()}".encode()).hexdigest()
        
        async def data_generator():
            
    """data_generator function."""
if isinstance(data, (list, tuple)):
                for item in data:
                    yield item
                    await asyncio.sleep(0.01)  # Prevent blocking
            else:
                yield data
        
        return await self.streaming_generator.create_streaming_response(
            data_generator, response_id, kwargs.get('chunk_size')
        )
    
    async def _create_chunked_response(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Create a chunked response."""
        return await self.chunked_generator.create_chunked_response(
            data, kwargs.get('chunk_size')
        )
    
    async def _create_paginated_response(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Create a paginated response."""
        page = kwargs.get('page', 0)
        page_size = kwargs.get('page_size')
        total_count = kwargs.get('total_count')
        
        return await self.paginated_generator.create_paginated_response(
            data, page, page_size, total_count
        )
    
    async def _create_progressive_response(self, data: Any, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Create a progressive response."""
        async def data_loader(stage: str, stage_index: int):
            
    """data_loader function."""
match stage:
    case "initial":
                return data[:100] if isinstance(data, list) else data
            elmatch stage:
    case "enhanced":
                return data[:500] if isinstance(data, list) else data
            else:  # complete
                return data
        
        async for chunk in self.progressive_generator.create_progressive_response(data_loader):
            yield chunk
    
    async def _create_hybrid_response(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Create a hybrid response combining multiple strategies."""
        # Start with lazy loading
        lazy_response = await self._create_lazy_response(data, **kwargs)
        
        # Add streaming capability
        lazy_response["streaming_available"] = True
        lazy_response["stream_url"] = f"/api/responses/{lazy_response['response_id']}/stream"
        
        # Add chunking capability
        lazy_response["chunking_available"] = True
        lazy_response["chunk_url"] = f"/api/responses/{lazy_response['response_id']}/chunks"
        
        return lazy_response
    
    def _update_metrics(self, strategy: ResponseOptimizationStrategy, execution_time: float, data: Any):
        """Update response metrics."""
        self.metrics.total_responses += 1
        self.metrics.total_response_time += execution_time
        self.metrics.average_response_time = self.metrics.total_response_time / self.metrics.total_responses
        
        # Update strategy-specific metrics
        if strategy == ResponseOptimizationStrategy.LAZY:
            self.metrics.lazy_responses += 1
        elif strategy == ResponseOptimizationStrategy.STREAMING:
            self.metrics.streaming_responses += 1
        elif strategy == ResponseOptimizationStrategy.CHUNKED:
            self.metrics.chunked_responses += 1
        elif strategy == ResponseOptimizationStrategy.PAGINATED:
            self.metrics.paginated_responses += 1
        elif strategy == ResponseOptimizationStrategy.PROGRESSIVE:
            self.metrics.progressive_responses += 1
        
        # Update size metrics
        data_size = self.size_analyzer.analyze_response_size(data).value
        self.metrics.total_data_size += data_size
        self.metrics.average_data_size = self.metrics.total_data_size / self.metrics.total_responses
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get response optimization metrics."""
        return {
            "total_responses": self.metrics.total_responses,
            "lazy_responses": self.metrics.lazy_responses,
            "streaming_responses": self.metrics.streaming_responses,
            "chunked_responses": self.metrics.chunked_responses,
            "paginated_responses": self.metrics.paginated_responses,
            "progressive_responses": self.metrics.progressive_responses,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": self.metrics.cache_hits / self.metrics.total_responses if self.metrics.total_responses > 0 else 0,
            "total_response_time": self.metrics.total_response_time,
            "average_response_time": self.metrics.average_response_time,
            "total_data_size": self.metrics.total_data_size,
            "average_data_size": self.metrics.average_data_size,
            "errors": self.metrics.errors
        }

# FastAPI integration
def setup_response_optimization(app: FastAPI, optimizer: APIResponseOptimizer):
    """Setup response optimization for FastAPI app."""
    
    @app.middleware("http")
    async def response_optimization_middleware(request: Request, call_next):
        """Middleware for automatic response optimization."""
        response = await call_next(request)
        
        # Check if response needs optimization
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                # Get response body
                response_body = await response.body()
                response_data = orjson.loads(response_body)
                
                # Analyze response size
                size = optimizer.size_analyzer.analyze_response_size(response_data)
                
                # Add optimization headers
                response.headers["X-Response-Size"] = size.value
                response.headers["X-Response-Optimized"] = "true"
                
            except Exception as e:
                logger.error(f"Response optimization middleware error: {e}")
        
        return response
    
    # Add optimization endpoints
    @app.get("/api/optimization/metrics")
    async def get_optimization_metrics():
        """Get response optimization metrics."""
        return optimizer.get_metrics()
    
    @app.post("/api/optimization/clear-cache")
    async def clear_optimization_cache():
        """Clear response optimization cache."""
        async with optimizer._lock:
            optimizer.response_cache.clear()
        return {"message": "Optimization cache cleared successfully"}

# Decorators for easy response optimization
def optimize_response(strategy: ResponseOptimizationStrategy = None, **kwargs):
    """Decorator for response optimization."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **func_kwargs) -> Any:
            # Get optimizer from request state or create new one
            optimizer = func_kwargs.pop('optimizer', None)
            if optimizer is None:
                config = ResponseOptimizationConfig()
                optimizer = APIResponseOptimizer(config)
            
            # Call original function
            result = await func(*args, **func_kwargs)
            
            # Optimize response
            return await optimizer.optimize_response(result, strategy, **kwargs)
        
        return wrapper
    return decorator

def streaming_response(chunk_size: int = None):
    """Decorator for streaming responses."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            config = ResponseOptimizationConfig()
            optimizer = APIResponseOptimizer(config)
            
            result = await func(*args, **kwargs)
            
            return await optimizer.optimize_response(
                result, ResponseOptimizationStrategy.STREAMING, chunk_size=chunk_size
            )
        
        return wrapper
    return decorator

def chunked_response(chunk_size: int = None):
    """Decorator for chunked responses."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            config = ResponseOptimizationConfig()
            optimizer = APIResponseOptimizer(config)
            
            result = await func(*args, **kwargs)
            
            return await optimizer.optimize_response(
                result, ResponseOptimizationStrategy.CHUNKED, chunk_size=chunk_size
            )
        
        return wrapper
    return decorator

def paginated_response(page_size: int = None):
    """Decorator for paginated responses."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            config = ResponseOptimizationConfig()
            optimizer = APIResponseOptimizer(config)
            
            result = await func(*args, **kwargs)
            
            return await optimizer.optimize_response(
                result, ResponseOptimizationStrategy.PAGINATED, page_size=page_size
            )
        
        return wrapper
    return decorator

# Example usage
async def example_response_optimization():
    """Example usage of API response optimization."""
    
    # Create configuration
    config = ResponseOptimizationConfig(
        default_strategy=ResponseOptimizationStrategy.LAZY,
        enable_streaming=True,
        enable_chunking=True,
        enable_pagination=True,
        enable_progressive_loading=True,
        default_chunk_size=1024 * 1024,  # 1MB
        default_page_size=100
    )
    
    # Initialize optimizer
    optimizer = APIResponseOptimizer(config)
    
    # Example data generators
    def generate_large_dataset():
        """Generate a large dataset."""
        return [f"data_item_{i}" for i in range(10000)]
    
    async def generate_streaming_data():
        """Generate streaming data."""
        for i in range(1000):
            yield {"id": i, "content": f"stream_content_{i}"}
            await asyncio.sleep(0.01)
    
    try:
        # Test different optimization strategies
        
        # 1. Simple response (no optimization)
        logger.info("Testing simple response...")
        simple_data = generate_large_dataset()[:100]  # Small subset
        simple_response = await optimizer.optimize_response(simple_data, ResponseOptimizationStrategy.NONE)
        logger.info(f"Simple response type: {type(simple_response)}")
        
        # 2. Lazy response
        logger.info("Testing lazy response...")
        lazy_response = await optimizer.optimize_response(generate_large_dataset(), ResponseOptimizationStrategy.LAZY)
        logger.info(f"Lazy response: {lazy_response}")
        
        # 3. Streaming response
        logger.info("Testing streaming response...")
        streaming_response = await optimizer.optimize_response(
            generate_streaming_data(), ResponseOptimizationStrategy.STREAMING
        )
        logger.info(f"Streaming response type: {type(streaming_response)}")
        
        # 4. Chunked response
        logger.info("Testing chunked response...")
        chunked_response = await optimizer.optimize_response(
            generate_large_dataset(), ResponseOptimizationStrategy.CHUNKED
        )
        logger.info(f"Chunked response: {len(chunked_response['chunks'])} chunks")
        
        # 5. Paginated response
        logger.info("Testing paginated response...")
        paginated_response = await optimizer.optimize_response(
            generate_large_dataset(), ResponseOptimizationStrategy.PAGINATED, page=0, page_size=100
        )
        logger.info(f"Paginated response: {len(paginated_response['data'])} items")
        
        # 6. Progressive response
        logger.info("Testing progressive response...")
        progressive_generator = optimizer.optimize_response(
            generate_large_dataset(), ResponseOptimizationStrategy.PROGRESSIVE
        )
        async for stage in progressive_generator:
            logger.info(f"Progressive stage: {stage['stage']}, items: {len(stage['data'])}")
        
        # Get metrics
        metrics = optimizer.get_metrics()
        logger.info(f"Response optimization metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Response optimization error: {e}")

match __name__:
    case "__main__":
    asyncio.run(example_response_optimization()) 