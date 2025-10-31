"""
Advanced Features for Ultra-Adaptive K/V Cache Engine
Includes: streaming responses, request prioritization, prefetching, and more
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, AsyncIterator, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
from queue import PriorityQueue
import logging
from collections import deque

try:
    from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine
except ImportError:
    UltraAdaptiveKVCacheEngine = None

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class PrioritizedRequest:
    """Request with priority."""
    priority: RequestPriority
    request: Dict[str, Any]
    timestamp: float
    deadline: Optional[float] = None
    
    def __lt__(self, other):
        """Compare for priority queue."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        
        # If same priority, earlier deadline first
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        
        # Earlier timestamp first
        return self.timestamp < other.timestamp


class PriorityRequestQueue:
    """Priority queue for requests."""
    
    def __init__(self):
        self.queue = PriorityQueue()
        self.lock = asyncio.Lock()
    
    async def put(self, prioritized_request: PrioritizedRequest):
        """Add request to queue."""
        await self.lock.acquire()
        try:
            self.queue.put(prioritized_request)
        finally:
            self.lock.release()
    
    async def get(self) -> PrioritizedRequest:
        """Get next request from queue."""
        while True:
            await self.lock.acquire()
            try:
                if not self.queue.empty():
                    return self.queue.get()
            finally:
                self.lock.release()
            
            await asyncio.sleep(0.01)  # Small delay to avoid busy waiting
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()


class StreamingResponseHandler:
    """Handler for streaming responses."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine):
        self.engine = engine
    
    async def stream_response(self, request: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Stream response tokens as they are generated.
        
        Usage:
            async for token in handler.stream_response(request):
                yield token
        """
        # This is a simplified version - in practice, would integrate with engine's streaming
        result = await self.engine.process_request(request)
        
        if result['success']:
            response_text = result['response'].get('text', '')
            
            # Stream in chunks
            chunk_size = 10  # characters per chunk
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.01)  # Small delay for streaming effect


class RequestPrefetcher:
    """Intelligent request prefetching."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine):
        self.engine = engine
        self.prefetch_queue = asyncio.Queue()
        self.prefetch_cache = {}
        self.prefetch_enabled = True
    
    async def prefetch(self, request: Dict[str, Any]):
        """Prefetch request result."""
        if not self.prefetch_enabled:
            return
        
        # Check if already prefetched
        request_key = self._request_key(request)
        if request_key in self.prefetch_cache:
            return
        
        # Add to prefetch queue
        await self.prefetch_queue.put(request)
    
    async def get_prefetched(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get prefetched result if available."""
        request_key = self._request_key(request)
        return self.prefetch_cache.pop(request_key, None)
    
    def _request_key(self, request: Dict[str, Any]) -> str:
        """Generate key for request."""
        return f"{request.get('session_id', '')}:{hash(str(request.get('text', '')))}"
    
    async def _prefetch_worker(self):
        """Background worker for prefetching."""
        while self.prefetch_enabled:
            try:
                request = await asyncio.wait_for(self.prefetch_queue.get(), timeout=1.0)
                request_key = self._request_key(request)
                
                # Process request
                result = await self.engine.process_request(request)
                
                # Cache result
                self.prefetch_cache[request_key] = result
                
                # Limit cache size
                if len(self.prefetch_cache) > 100:
                    # Remove oldest
                    oldest_key = next(iter(self.prefetch_cache))
                    del self.prefetch_cache[oldest_key]
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in prefetch worker: {e}")
    
    def start(self):
        """Start prefetch worker."""
        if self.prefetch_enabled:
            asyncio.create_task(self._prefetch_worker())
    
    def stop(self):
        """Stop prefetch worker."""
        self.prefetch_enabled = False


class RequestBatchingOptimizer:
    """Optimize request batching based on various factors."""
    
    def __init__(self, max_batch_size: int = 20, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batch_queue = asyncio.Queue()
    
    async def add_request(self, request: Dict[str, Any]) -> asyncio.Future:
        """Add request and return future for result."""
        future = asyncio.Future()
        await self.batch_queue.put((request, future))
        return future
    
    async def process_batches(self, engine: UltraAdaptiveKVCacheEngine):
        """Process batches continuously."""
        batch = []
        futures = []
        deadline = time.time() + self.max_wait_time
        
        while True:
            try:
                # Try to get request with timeout
                timeout = max(0, deadline - time.time())
                if timeout <= 0 or len(batch) >= self.max_batch_size:
                    # Process batch
                    if batch:
                        results = await engine.process_batch(batch)
                        
                        # Set futures
                        for i, result in enumerate(results):
                            if i < len(futures) and not futures[i].done():
                                futures[i].set_result(result)
                        
                        batch = []
                        futures = []
                        deadline = time.time() + self.max_wait_time
                    
                    if batch:
                        continue
                
                # Wait for next request
                request, future = await asyncio.wait_for(
                    self.batch_queue.get(),
                    timeout=timeout
                )
                
                batch.append(request)
                futures.append(future)
                
                # Reset deadline if batch is full
                if len(batch) >= self.max_batch_size:
                    deadline = time.time()
            
            except asyncio.TimeoutError:
                # Timeout reached, process current batch
                if batch:
                    results = await engine.process_batch(batch)
                    
                    for i, result in enumerate(results):
                        if i < len(futures) and not futures[i].done():
                            futures[i].set_result(result)
                    
                    batch = []
                    futures = []
                    deadline = time.time() + self.max_wait_time
            
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # Set error on futures
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                batch = []
                futures = []


class RequestDeduplicator:
    """Deduplicate identical requests."""
    
    def __init__(self, ttl: float = 60.0):
        self.ttl = ttl
        self.cache = {}
        self.lock = asyncio.Lock()
    
    async def deduplicate(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if request is duplicate and return cached result if available.
        Returns None if not duplicate.
        """
        request_key = self._request_key(request)
        
        async with self.lock:
            if request_key in self.cache:
                entry = self.cache[request_key]
                
                # Check if still valid
                if time.time() - entry['timestamp'] < self.ttl:
                    return entry['result']
                else:
                    # Expired
                    del self.cache[request_key]
        
        return None
    
    async def cache_result(self, request: Dict[str, Any], result: Dict[str, Any]):
        """Cache result for deduplication."""
        request_key = self._request_key(request)
        
        async with self.lock:
            self.cache[request_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            # Cleanup expired entries
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time - entry['timestamp'] >= self.ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
    
    def _request_key(self, request: Dict[str, Any]) -> str:
        """Generate key for request."""
        return f"{request.get('session_id', '')}:{hash(str(request.get('text', '')))}"


class AdaptiveThrottler:
    """Adaptive throttling based on system load."""
    
    def __init__(self, initial_rate: float = 10.0, min_rate: float = 1.0, 
                 max_rate: float = 100.0):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.last_request_time = 0
        self.request_times = deque(maxlen=100)
        self.error_count = 0
    
    async def acquire(self):
        """Acquire permission to process request."""
        current_time = time.time()
        
        # Calculate minimum interval
        min_interval = 1.0 / self.current_rate
        
        # Check if we need to wait
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_times.append(self.last_request_time)
    
    def update_rate_based_on_errors(self, error_rate: float):
        """Update rate based on error rate."""
        if error_rate > 0.1:  # High error rate
            self.current_rate = max(self.min_rate, self.current_rate * 0.8)
        elif error_rate < 0.01:  # Low error rate
            self.current_rate = min(self.max_rate, self.current_rate * 1.2)
    
    def get_current_rate(self) -> float:
        """Get current rate limit."""
        return self.current_rate


class RequestValidator:
    """Advanced request validation."""
    
    @staticmethod
    def validate(request: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate request with detailed checks."""
        # Check required fields
        if 'text' not in request:
            return False, "Missing required field: 'text'"
        
        text = request.get('text', '')
        
        # Check text length
        if len(text) == 0:
            return False, "Text cannot be empty"
        
        if len(text) > 100000:  # 100KB limit
            return False, "Text exceeds maximum length (100KB)"
        
        # Check max_length
        max_length = request.get('max_length', 100)
        if not isinstance(max_length, int) or max_length < 1 or max_length > 10000:
            return False, "max_length must be between 1 and 10000"
        
        # Check temperature
        temperature = request.get('temperature', 1.0)
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            return False, "temperature must be between 0 and 2"
        
        # Check session_id format
        session_id = request.get('session_id')
        if session_id and not isinstance(session_id, str):
            return False, "session_id must be a string"
        
        return True, None
    
    @staticmethod
    def sanitize(request: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request to remove potentially harmful content."""
        sanitized = request.copy()
        
        # Truncate very long text
        if 'text' in sanitized and len(sanitized['text']) > 50000:
            sanitized['text'] = sanitized['text'][:50000] + "... [truncated]"
        
        # Ensure safe types
        if 'max_length' in sanitized:
            sanitized['max_length'] = int(sanitized['max_length'])
        
        if 'temperature' in sanitized:
            sanitized['temperature'] = float(sanitized['temperature'])
        
        return sanitized

