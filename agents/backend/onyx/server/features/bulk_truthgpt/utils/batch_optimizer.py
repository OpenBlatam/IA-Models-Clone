"""
Batch Processing Optimizer
===========================

Intelligent batch processing with:
- Dynamic batch size optimization
- Adaptive batching
- Parallel processing
- Resource-aware scheduling
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Batch processing configuration."""
    min_batch_size: int = 1
    max_batch_size: int = 100
    initial_batch_size: int = 10
    target_latency_ms: float = 1000.0
    max_concurrent_batches: int = 5
    adaptive: bool = True

class BatchOptimizer:
    """
    Intelligent batch optimizer that:
    - Adapts batch size based on performance
    - Monitors resource usage
    - Optimizes for throughput vs latency
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.current_batch_size = self.config.initial_batch_size
        self.performance_history: List[Dict[str, Any]] = []
        self.active_batches = 0
        self.max_active_batches = self.config.max_concurrent_batches
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process a batch of items."""
        if not items:
            return []
        
        start_time = time.time()
        
        try:
            # Process batch
            if asyncio.iscoroutinefunction(processor):
                results = await processor(items, *args, **kwargs)
            else:
                results = processor(items, *args, **kwargs)
            
            latency = (time.time() - start_time) * 1000  # ms
            
            # Record performance
            self._record_performance(len(items), latency)
            
            # Adapt batch size if needed
            if self.config.adaptive:
                self._adapt_batch_size(latency)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def _record_performance(self, batch_size: int, latency_ms: float):
        """Record batch processing performance."""
        self.performance_history.append({
            "timestamp": datetime.now(),
            "batch_size": batch_size,
            "latency_ms": latency_ms,
            "throughput": batch_size / (latency_ms / 1000) if latency_ms > 0 else 0
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _adapt_batch_size(self, latency_ms: float):
        """Adapt batch size based on performance."""
        if not self.performance_history:
            return
        
        recent = self.performance_history[-10:]
        avg_latency = sum(p["latency_ms"] for p in recent) / len(recent)
        
        target = self.config.target_latency_ms
        
        # If latency is too high, reduce batch size
        if avg_latency > target * 1.2:
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            logger.debug(f"Reduced batch size to {self.current_batch_size} (latency: {avg_latency:.2f}ms)")
        
        # If latency is low and we have capacity, increase batch size
        elif avg_latency < target * 0.8:
            self.current_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            logger.debug(f"Increased batch size to {self.current_batch_size} (latency: {avg_latency:.2f}ms)")
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on current conditions."""
        if not self.config.adaptive:
            return self.current_batch_size
        
        # Check system resources
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Reduce batch size if resources are constrained
        if cpu_usage > 80 or memory.percent > 80:
            return max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.7)
            )
        
        return self.current_batch_size
    
    async def process_in_batches(
        self,
        items: List[Any],
        processor: Callable,
        batch_size: Optional[int] = None,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process items in optimized batches."""
        if not items:
            return []
        
        batch_size = batch_size or self.get_optimal_batch_size()
        all_results = []
        
        # Create batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(self.max_active_batches)
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await self.process_batch(batch, processor, *args, **kwargs)
        
        # Process all batches
        tasks = [process_with_semaphore(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
            else:
                all_results.extend(result)
        
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        if not self.performance_history:
            return {}
        
        recent = self.performance_history[-100:]
        
        return {
            "current_batch_size": self.current_batch_size,
            "total_batches": len(self.performance_history),
            "avg_latency_ms": sum(p["latency_ms"] for p in recent) / len(recent) if recent else 0,
            "avg_throughput": sum(p["throughput"] for p in recent) / len(recent) if recent else 0,
            "min_batch_size": self.config.min_batch_size,
            "max_batch_size": self.config.max_batch_size
        }
    
    def reset(self):
        """Reset optimizer state."""
        self.current_batch_size = self.config.initial_batch_size
        self.performance_history.clear()

# Global instance
batch_optimizer = BatchOptimizer()
































