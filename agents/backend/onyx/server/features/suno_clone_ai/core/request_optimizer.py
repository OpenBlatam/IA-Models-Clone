"""
Request Processing Optimizations

Optimizations for:
- Request validation
- Request batching
- Request deduplication
- Request prioritization
- Request queuing
"""

import logging
import hashlib
import time
import json
from typing import Optional, Dict, Any, List, Callable
from collections import deque
import asyncio
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Request:
    """Request with priority."""
    data: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest())


class RequestDeduplicator:
    """Deduplicate duplicate requests."""
    
    def __init__(self, ttl: int = 60):
        """
        Initialize deduplicator.
        
        Args:
            ttl: Time to live for deduplication cache
        """
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.ttl = ttl
    
    def _make_key(self, request_data: Dict[str, Any]) -> str:
        """Create deduplication key."""
        # Sort and hash request data
        key_data = json.dumps(request_data, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def check_duplicate(
        self,
        request_data: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Check if request is duplicate.
        
        Args:
            request_data: Request data
            
        Returns:
            Cached result if duplicate, None otherwise
        """
        key = self._make_key(request_data)
        now = time.time()
        
        # Clean expired entries
        expired_keys = [
            k for k, (_, timestamp) in self.cache.items()
            if now - timestamp > self.ttl
        ]
        for k in expired_keys:
            del self.cache[k]
        
        # Check for duplicate
        if key in self.cache:
            result, _ = self.cache[key]
            logger.debug(f"Duplicate request detected: {key}")
            return result
        
        return None
    
    def cache_result(
        self,
        request_data: Dict[str, Any],
        result: Any
    ) -> None:
        """Cache request result."""
        key = self._make_key(request_data)
        self.cache[key] = (result, time.time())


class PriorityQueue:
    """Priority queue for requests."""
    
    def __init__(self):
        """Initialize priority queue."""
        self.queues = {
            Priority.URGENT: deque(),
            Priority.HIGH: deque(),
            Priority.NORMAL: deque(),
            Priority.LOW: deque()
        }
    
    def enqueue(self, request: Request) -> None:
        """Add request to queue."""
        self.queues[request.priority].append(request)
    
    def dequeue(self) -> Optional[Request]:
        """Get next request from queue."""
        for priority in [Priority.URGENT, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            if self.queues[priority]:
                return self.queues[priority].popleft()
        return None
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return all(len(q) == 0 for q in self.queues.values())
    
    def size(self) -> int:
        """Get total queue size."""
        return sum(len(q) for q in self.queues.values())


class RequestBatcher:
    """Batch similar requests together."""
    
    def __init__(self, batch_size: int = 10, timeout_ms: int = 100):
        """
        Initialize request batcher.
        
        Args:
            batch_size: Maximum batch size
            timeout_ms: Timeout in milliseconds
        """
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.current_batch: List[Request] = []
        self.batch_futures: List[asyncio.Future] = []
        self.last_batch_time = time.time()
    
    async def add_request(
        self,
        request: Request,
        processor_func: Callable
    ) -> Any:
        """
        Add request to batch.
        
        Args:
            request: Request to add
            processor_func: Function to process batch
            
        Returns:
            Result for this request
        """
        # Create future for this request
        future = asyncio.Future()
        self.current_batch.append(request)
        self.batch_futures.append(future)
        
        # Process batch if full
        if len(self.current_batch) >= self.batch_size:
            await self._process_batch(processor_func)
        
        # Wait for result
        return await future
    
    async def _process_batch(self, processor_func: Callable) -> None:
        """Process current batch."""
        if not self.current_batch:
            return
        
        batch = self.current_batch[:]
        futures = self.batch_futures[:]
        
        # Clear current batch
        self.current_batch.clear()
        self.batch_futures.clear()
        
        try:
            # Process batch
            if asyncio.iscoroutinefunction(processor_func):
                results = await processor_func([r.data for r in batch])
            else:
                results = processor_func([r.data for r in batch])
            
            # Resolve futures
            for i, future in enumerate(futures):
                if i < len(results):
                    future.set_result(results[i])
                else:
                    future.set_exception(IndexError("Result not available"))
                    
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                future.set_exception(e)
    
    async def flush(self, processor_func: Callable) -> None:
        """Flush remaining requests in batch."""
        if self.current_batch:
            await self._process_batch(processor_func)


class RequestValidator:
    """Fast request validation."""
    
    @staticmethod
    def validate_request(
        request_data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate request against schema.
        
        Args:
            request_data: Request data
            schema: Validation schema
            
        Returns:
            (is_valid, error_message)
        """
        # Required fields
        required = schema.get('required', [])
        for field in required:
            if field not in request_data:
                return False, f"Missing required field: {field}"
        
        # Type validation
        types = schema.get('types', {})
        for field, expected_type in types.items():
            if field in request_data:
                if not isinstance(request_data[field], expected_type):
                    return False, f"Field {field} must be {expected_type.__name__}"
        
        # Range validation
        ranges = schema.get('ranges', {})
        for field, (min_val, max_val) in ranges.items():
            if field in request_data:
                value = request_data[field]
                if not (min_val <= value <= max_val):
                    return False, f"Field {field} must be between {min_val} and {max_val}"
        
        return True, None


class RequestThrottler:
    """Throttle requests to prevent overload."""
    
    def __init__(self, max_requests_per_second: float = 10.0):
        """
        Initialize throttler.
        
        Args:
            max_requests_per_second: Maximum requests per second
        """
        self.max_rps = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
    
    async def throttle(self) -> None:
        """Throttle request if needed."""
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()

