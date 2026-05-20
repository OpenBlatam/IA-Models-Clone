"""
Throttle Manager for Color Grading AI
======================================

Request throttling with priority and queuing.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ThrottlePriority(Enum):
    """Throttle priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThrottleConfig:
    """Throttle configuration."""
    max_concurrent: int = 10
    max_queue_size: int = 100
    timeout: float = 60.0
    priority_enabled: bool = True


class ThrottleManager:
    """
    Request throttling manager.
    
    Features:
    - Concurrent request limiting
    - Priority-based queuing
    - Timeout handling
    - Statistics
    """
    
    def __init__(self, config: Optional[ThrottleConfig] = None):
        """
        Initialize throttle manager.
        
        Args:
            config: Optional throttle configuration
        """
        self.config = config or ThrottleConfig()
        self._active: Dict[str, asyncio.Task] = {}
        self._queue: List[Dict[str, Any]] = []
        self._stats: Dict[str, Any] = {
            "processed": 0,
            "queued": 0,
            "rejected": 0,
            "timeout": 0,
        }
        self._lock = asyncio.Lock()
    
    async def execute(
        self,
        func: Callable,
        key: str = "default",
        priority: ThrottlePriority = ThrottlePriority.NORMAL,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with throttling.
        
        Args:
            func: Function to execute
            key: Throttle key
            priority: Request priority
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            ThrottleRejectedError: If queue is full
            asyncio.TimeoutError: If timeout
        """
        async with self._lock:
            # Check if can execute immediately
            if len(self._active) < self.config.max_concurrent:
                task = asyncio.create_task(self._execute_task(func, key, *args, **kwargs))
                self._active[key] = task
                self._stats["processed"] += 1
                return await task
            
            # Check queue capacity
            if len(self._queue) >= self.config.max_queue_size:
                self._stats["rejected"] += 1
                raise ThrottleRejectedError(f"Queue full for key: {key}")
            
            # Add to queue
            queue_item = {
                "func": func,
                "key": key,
                "priority": priority,
                "args": args,
                "kwargs": kwargs,
                "event": asyncio.Event(),
                "result": None,
                "exception": None,
                "created_at": datetime.now(),
            }
            
            # Insert by priority
            self._insert_by_priority(queue_item)
            self._stats["queued"] += 1
        
        # Wait for execution
        try:
            await asyncio.wait_for(
                queue_item["event"].wait(),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            self._stats["timeout"] += 1
            raise
        
        # Check for exception
        if queue_item["exception"]:
            raise queue_item["exception"]
        
        return queue_item["result"]
    
    def _insert_by_priority(self, item: Dict[str, Any]):
        """Insert item in queue by priority."""
        if not self.config.priority_enabled:
            self._queue.append(item)
            return
        
        priority_value = item["priority"].value
        insert_index = len(self._queue)
        
        for i, queued_item in enumerate(self._queue):
            if queued_item["priority"].value < priority_value:
                insert_index = i
                break
        
        self._queue.insert(insert_index, item)
    
    async def _execute_task(
        self,
        func: Callable,
        key: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute task and process queue."""
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result
        finally:
            async with self._lock:
                if key in self._active:
                    del self._active[key]
                
                # Process next in queue
                if self._queue:
                    next_item = self._queue.pop(0)
                    task = asyncio.create_task(
                        self._execute_queued_item(next_item)
                    )
                    self._active[next_item["key"]] = task
    
    async def _execute_queued_item(self, item: Dict[str, Any]):
        """Execute queued item."""
        try:
            result = await self._execute_task(
                item["func"],
                item["key"],
                *item["args"],
                **item["kwargs"]
            )
            item["result"] = result
        except Exception as e:
            item["exception"] = e
        finally:
            item["event"].set()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throttle statistics."""
        return {
            **self._stats,
            "active": len(self._active),
            "queued": len(self._queue),
            "max_concurrent": self.config.max_concurrent,
            "max_queue_size": self.config.max_queue_size,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "processed": 0,
            "queued": 0,
            "rejected": 0,
            "timeout": 0,
        }


class ThrottleRejectedError(Exception):
    """Throttle queue is full."""
    pass




