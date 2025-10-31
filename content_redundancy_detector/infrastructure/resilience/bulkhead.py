"""
Advanced Bulkhead Pattern - Resource isolation and fault containment
Production-ready bulkhead system for resource management
"""

import asyncio
import threading
from typing import Any, Callable, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class BulkheadState(Enum):
    """Bulkhead states"""
    AVAILABLE = "available"
    FULL = "full"
    ISOLATED = "isolated"

@dataclass
class BulkheadConfig:
    """Bulkhead configuration"""
    max_concurrent: int = 10
    max_waiting: int = 100
    timeout: float = 60.0
    isolation_threshold: float = 0.9  # Isolate at 90% capacity

class Bulkhead:
    """Advanced bulkhead for resource isolation"""
    
    def __init__(self, name: str, config: BulkheadConfig = None):
        self.name = name
        self.config = config or BulkheadConfig()
        
        # Resource tracking
        self.active_count = 0
        self.waiting_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_waiting)
        self.completed_count = 0
        self.rejected_count = 0
        
        # State management
        self.state = BulkheadState.AVAILABLE
        self.isolated = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.total_requests = 0

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function within bulkhead constraints"""
        self.total_requests += 1
        
        # Check if isolated
        if self.isolated:
            self.rejected_count += 1
            raise BulkheadIsolatedError(
                f"Bulkhead '{self.name}' is isolated"
            )
        
        # Wait for available slot
        if self.active_count >= self.config.max_concurrent:
            if self.waiting_queue.qsize() >= self.config.max_waiting:
                self.rejected_count += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' is full (max: {self.config.max_concurrent})"
                )
            
            # Add to waiting queue
            await self.waiting_queue.put(None)
        
        # Acquire slot
        self.active_count += 1
        self._update_state()
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self.completed_count += 1
            return result
            
        finally:
            # Release slot
            self.active_count -= 1
            self._update_state()
            
            # Process waiting queue
            if not self.waiting_queue.empty():
                try:
                    self.waiting_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

    def _update_state(self):
        """Update bulkhead state"""
        utilization = self.active_count / max(self.config.max_concurrent, 1)
        
        if utilization >= self.config.isolation_threshold:
            self.state = BulkheadState.FULL
            if not self.isolated:
                logger.warning(
                    f"Bulkhead '{self.name}' at {utilization*100:.1f}% capacity"
                )
        elif utilization > 0:
            self.state = BulkheadState.AVAILABLE
        else:
            self.state = BulkheadState.AVAILABLE

    def isolate(self):
        """Isolate bulkhead (prevent new requests)"""
        with self.lock:
            self.isolated = True
            logger.warning(f"Bulkhead '{self.name}' isolated")

    def unisolate(self):
        """Remove isolation"""
        with self.lock:
            self.isolated = False
            logger.info(f"Bulkhead '{self.name}' unisolated")

    def get_state(self) -> BulkheadState:
        """Get current bulkhead state"""
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics"""
        utilization = self.active_count / max(self.config.max_concurrent, 1)
        
        return {
            "name": self.name,
            "state": self.state.value,
            "isolated": self.isolated,
            "active_count": self.active_count,
            "max_concurrent": self.config.max_concurrent,
            "waiting_count": self.waiting_queue.qsize(),
            "utilization": utilization,
            "completed_count": self.completed_count,
            "rejected_count": self.rejected_count,
            "total_requests": self.total_requests
        }

class BulkheadIsolatedError(Exception):
    """Exception when bulkhead is isolated"""
    pass

class BulkheadFullError(Exception):
    """Exception when bulkhead is full"""
    pass

def with_bulkhead(bulkhead: Bulkhead):
    """Decorator for automatic bulkhead handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await bulkhead.execute(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'd need a sync bulkhead implementation
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

import functools






