"""
Performance Optimizer
Advanced performance optimizations
"""

from typing import Dict, Any, Optional
import logging
import asyncio
from functools import wraps
import time

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Advanced performance optimizations"""
    
    def __init__(self):
        self.optimizations_enabled = {
            "parallel_processing": True,
            "connection_pooling": True,
            "request_batching": True,
            "lazy_loading": True,
        }
    
    def enable_optimization(self, optimization: str):
        """Enable specific optimization"""
        if optimization in self.optimizations_enabled:
            self.optimizations_enabled[optimization] = True
            logger.info(f"Enabled optimization: {optimization}")
    
    def disable_optimization(self, optimization: str):
        """Disable specific optimization"""
        if optimization in self.optimizations_enabled:
            self.optimizations_enabled[optimization] = False
            logger.info(f"Disabled optimization: {optimization}")
    
    def optimize_image_generation_batch(
        self,
        image_tasks: list,
        max_concurrent: int = 5
    ) -> list:
        """
        Optimize batch image generation with concurrency control
        
        Args:
            image_tasks: List of image generation tasks
            max_concurrent: Maximum concurrent generations
            
        Returns:
            Results in same order as input
        """
        if not self.optimizations_enabled.get("parallel_processing", True):
            # Sequential processing
            return [task() for task in image_tasks]
        
        # Process in batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(task):
            async with semaphore:
                if asyncio.iscoroutinefunction(task):
                    return await task()
                else:
                    return task()
        
        async def process_all():
            tasks = [process_with_limit(task) for task in image_tasks]
            return await asyncio.gather(*tasks)
        
        # Run async processing
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(process_all())
    
    def batch_requests(self, requests: list, batch_size: int = 10) -> list:
        """
        Batch requests for efficient processing
        
        Args:
            requests: List of requests
            batch_size: Size of each batch
            
        Returns:
            Batched requests
        """
        if not self.optimizations_enabled.get("request_batching", True):
            return [[r] for r in requests]
        
        batches = []
        for i in range(0, len(requests), batch_size):
            batches.append(requests[i:i + batch_size])
        
        return batches
    
    def cache_key_optimization(self, data: Dict[str, Any]) -> str:
        """Optimize cache key generation"""
        import hashlib
        import json
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()


_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get performance optimizer instance (singleton)"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer

