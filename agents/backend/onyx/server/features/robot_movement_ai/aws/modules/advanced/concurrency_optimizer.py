"""
Concurrency Optimizer
=====================

Advanced concurrency optimization.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyConfig:
    """Concurrency configuration."""
    max_workers: int = 10
    min_workers: int = 2
    target_utilization: float = 0.7
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    adjustment_interval: float = 30.0


class ConcurrencyOptimizer:
    """Advanced concurrency optimizer with auto-scaling."""
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        self.config = config or ConcurrencyConfig()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._current_workers = self.config.min_workers
        self._utilization_history: List[float] = []
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._monitoring = False
    
    def _create_semaphore(self):
        """Create semaphore with current worker count."""
        self._semaphore = asyncio.Semaphore(self._current_workers)
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with optimized concurrency."""
        if self._semaphore is None:
            self._create_semaphore()
        
        async with self._semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return await asyncio.to_thread(func, *args, **kwargs)
    
    def start_auto_scaling(self):
        """Start automatic worker scaling."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._create_semaphore()
        
        async def monitor():
            while self._monitoring:
                await asyncio.sleep(self.config.adjustment_interval)
                await self._adjust_workers()
        
        asyncio.create_task(monitor())
        logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop automatic worker scaling."""
        self._monitoring = False
        logger.info("Auto-scaling stopped")
    
    async def _adjust_workers(self):
        """Adjust worker count based on utilization."""
        if not self._utilization_history:
            return
        
        avg_utilization = sum(self._utilization_history[-10:]) / len(self._utilization_history[-10:])
        
        if avg_utilization > self.config.scale_up_threshold:
            # Scale up
            if self._current_workers < self.config.max_workers:
                self._current_workers = min(
                    self._current_workers + 1,
                    self.config.max_workers
                )
                self._create_semaphore()
                logger.info(f"Scaled up to {self._current_workers} workers")
        
        elif avg_utilization < self.config.scale_down_threshold:
            # Scale down
            if self._current_workers > self.config.min_workers:
                self._current_workers = max(
                    self._current_workers - 1,
                    self.config.min_workers
                )
                self._create_semaphore()
                logger.info(f"Scaled down to {self._current_workers} workers")
    
    def record_utilization(self, utilization: float):
        """Record utilization metric."""
        self._utilization_history.append(utilization)
        
        # Keep only recent history
        if len(self._utilization_history) > 100:
            self._utilization_history = self._utilization_history[-50:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get concurrency statistics."""
        avg_utilization = (
            sum(self._utilization_history[-10:]) / len(self._utilization_history[-10:])
            if self._utilization_history else 0.0
        )
        
        return {
            "current_workers": self._current_workers,
            "min_workers": self.config.min_workers,
            "max_workers": self.config.max_workers,
            "avg_utilization": avg_utilization,
            "queue_size": self._task_queue.qsize(),
            "monitoring": self._monitoring
        }















