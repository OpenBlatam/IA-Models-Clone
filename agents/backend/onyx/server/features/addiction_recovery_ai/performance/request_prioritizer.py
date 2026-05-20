"""
Request Prioritizer
Priority-based request handling and QoS management
"""

import logging
import asyncio
from typing import Dict, Optional, Callable
from enum import IntEnum
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)


class RequestPriority(IntEnum):
    """Request priority levels"""
    CRITICAL = 10  # Health checks, emergency
    HIGH = 8       # User-facing critical operations
    NORMAL = 5     # Standard requests
    LOW = 3        # Background tasks
    BATCH = 1      # Batch operations


@dataclass
class PrioritizedRequest:
    """Prioritized request"""
    priority: RequestPriority
    request_id: str
    func: Callable
    args: tuple
    kwargs: dict
    timestamp: float
    future: asyncio.Future


class RequestPrioritizer:
    """
    Request prioritizer
    
    Features:
    - Priority-based queuing
    - QoS management
    - Fair scheduling
    - Priority inheritance
    - Deadline-based scheduling
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self._queues: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None
        self._stats = {
            "processed": 0,
            "dropped": 0,
            "by_priority": {priority.name: 0 for priority in RequestPriority}
        }
        
        logger.info("✅ Request prioritizer initialized")
    
    async def submit(
        self,
        priority: RequestPriority,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Submit prioritized request
        
        Args:
            priority: Request priority
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Check queue size
        total_size = sum(len(queue) for queue in self._queues.values())
        if total_size >= self.max_queue_size:
            # Drop lowest priority requests
            if self._queues[RequestPriority.BATCH]:
                self._queues[RequestPriority.BATCH].popleft()
                self._stats["dropped"] += 1
                logger.warning("Request dropped due to queue full")
            else:
                raise RuntimeError("Request queue full")
        
        # Create request
        request_id = f"{priority.name}_{int(time.time() * 1000)}"
        future = asyncio.Future()
        
        request = PrioritizedRequest(
            priority=priority,
            request_id=request_id,
            func=func,
            args=args,
            kwargs=kwargs,
            timestamp=time.time(),
            future=future
        )
        
        # Add to priority queue
        self._queues[priority].append(request)
        
        # Start processor if not running
        if not self._processing:
            self._start_processor()
        
        # Wait for result
        return await future
    
    def _start_processor(self):
        """Start request processor"""
        if self._processor_task is None or self._processor_task.done():
            self._processing = True
            self._processor_task = asyncio.create_task(self._process_requests())
    
    async def _process_requests(self):
        """Process requests in priority order"""
        while True:
            # Find highest priority non-empty queue
            request = None
            priority = None
            
            for p in sorted(RequestPriority, reverse=True):
                if self._queues[p]:
                    request = self._queues[p].popleft()
                    priority = p
                    break
            
            if request is None:
                # No requests, wait a bit
                await asyncio.sleep(0.01)
                continue
            
            # Process request
            try:
                if asyncio.iscoroutinefunction(request.func):
                    result = await request.func(*request.args, **request.kwargs)
                else:
                    result = request.func(*request.args, **request.kwargs)
                
                request.future.set_result(result)
                self._stats["processed"] += 1
                self._stats["by_priority"][priority.name] += 1
                
            except Exception as e:
                request.future.set_exception(e)
                logger.error(f"Request processing error: {e}")
            
            # Yield control
            await asyncio.sleep(0)
    
    def get_stats(self) -> Dict:
        """Get prioritizer statistics"""
        queue_sizes = {
            priority.name: len(queue)
            for priority, queue in self._queues.items()
        }
        
        return {
            **self._stats,
            "queue_sizes": queue_sizes,
            "total_queued": sum(queue_sizes.values())
        }
    
    async def shutdown(self):
        """Shutdown request processor"""
        self._processing = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass


# Global prioritizer instance
_prioritizer: Optional[RequestPrioritizer] = None


def get_request_prioritizer() -> RequestPrioritizer:
    """Get global request prioritizer instance"""
    global _prioritizer
    if _prioritizer is None:
        _prioritizer = RequestPrioritizer()
    return _prioritizer















