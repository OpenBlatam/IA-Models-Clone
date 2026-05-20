"""
Auto Scaler
===========

Dynamic worker scaling based on queue load for 24/7 operation.
"""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass

if TYPE_CHECKING:
    from .parallel_executor import ParallelExecutor

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    min_workers: int = 2
    max_workers: int = 20
    scale_up_threshold: float = 0.8  # Scale up when queue is 80% full
    scale_down_threshold: float = 0.2  # Scale down when queue is 20% full
    cooldown_seconds: int = 30  # Wait between scaling operations
    queue_capacity: int = 100  # Assumed queue capacity for threshold calculation


class AutoScaler:
    """
    Dynamically scales workers based on queue load.
    
    Features:
    - Scale up when queue is filling up
    - Scale down when queue is mostly empty
    - Cooldown period between scaling operations
    - Respects min/max worker limits
    """
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        """
        Initialize auto scaler.
        
        Args:
            config: Scaling configuration
        """
        self.config = config or ScalingConfig()
        self._last_scale_time: Optional[datetime] = None
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"Initialized AutoScaler (min={self.config.min_workers}, "
            f"max={self.config.max_workers})"
        )
    
    async def start(self, executor: "ParallelExecutor"):
        """
        Start auto-scaling monitor.
        
        Args:
            executor: The parallel executor to scale
        """
        if self._running:
            logger.warning("AutoScaler is already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(executor)
        )
        logger.info("AutoScaler started")
    
    async def stop(self):
        """Stop auto-scaling monitor."""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AutoScaler stopped")
    
    async def _monitor_loop(self, executor: "ParallelExecutor"):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._evaluate_and_scale(executor)
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AutoScaler error: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def _evaluate_and_scale(self, executor: "ParallelExecutor"):
        """Evaluate queue load and scale if needed."""
        # Check cooldown
        if self._last_scale_time:
            elapsed = (datetime.now() - self._last_scale_time).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return
        
        # Get current state
        stats = executor.get_stats()
        queue_size = stats.get("queue_size", 0)
        active_workers = stats.get("active_workers", 0)
        
        # Calculate load ratio
        load_ratio = queue_size / self.config.queue_capacity
        
        # Scale up
        if load_ratio > self.config.scale_up_threshold:
            if active_workers < self.config.max_workers:
                new_workers = min(
                    active_workers + 2,  # Add 2 workers at a time
                    self.config.max_workers
                )
                workers_to_add = new_workers - active_workers
                if workers_to_add > 0:
                    await executor.add_workers(workers_to_add)
                    self._last_scale_time = datetime.now()
                    logger.info(
                        f"Scaled UP: {active_workers} -> {new_workers} workers "
                        f"(queue: {queue_size}, load: {load_ratio:.1%})"
                    )
        
        # Scale down
        elif load_ratio < self.config.scale_down_threshold:
            if active_workers > self.config.min_workers:
                new_workers = max(
                    active_workers - 1,  # Remove 1 worker at a time
                    self.config.min_workers
                )
                workers_to_remove = active_workers - new_workers
                if workers_to_remove > 0:
                    await executor.remove_workers(workers_to_remove)
                    self._last_scale_time = datetime.now()
                    logger.info(
                        f"Scaled DOWN: {active_workers} -> {new_workers} workers "
                        f"(queue: {queue_size}, load: {load_ratio:.1%})"
                    )
    
    def get_status(self) -> dict:
        """Get auto-scaler status."""
        return {
            "running": self._running,
            "config": {
                "min_workers": self.config.min_workers,
                "max_workers": self.config.max_workers,
                "scale_up_threshold": self.config.scale_up_threshold,
                "scale_down_threshold": self.config.scale_down_threshold,
                "cooldown_seconds": self.config.cooldown_seconds,
            },
            "last_scale_time": (
                self._last_scale_time.isoformat()
                if self._last_scale_time else None
            ),
        }
