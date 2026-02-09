"""
Batch Optimizer
===============

Advanced batch processing optimization.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of batch operation."""
    total_items: int
    successful: int
    failed: int
    total_time: float
    avg_time_per_item: float
    results: List[Dict[str, Any]]


class BatchOptimizer:
    """
    Advanced batch processing optimizer.
    
    Features:
    - Dynamic batch sizing
    - Load balancing
    - Error recovery
    - Progress tracking
    - Resource management
    """
    
    def __init__(
        self,
        initial_batch_size: int = 4,
        max_batch_size: int = 16,
        min_batch_size: int = 1,
        adaptive: bool = True
    ):
        """
        Initialize batch optimizer.
        
        Args:
            initial_batch_size: Initial batch size
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size
            adaptive: Enable adaptive batch sizing
        """
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.adaptive = adaptive
        
        self.current_batch_size = initial_batch_size
        self.batch_history = []
        
        logger.info(f"BatchOptimizer initialized (batch_size: {initial_batch_size})")
    
    async def process_batch_optimized(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_retries: int = 2
    ) -> BatchResult:
        """
        Process batch with optimization.
        
        Args:
            items: List of items to process
            process_func: Async function to process items
            progress_callback: Progress callback
            max_retries: Maximum retries for failed items
            
        Returns:
            BatchResult with statistics
        """
        start_time = time.time()
        total_items = len(items)
        successful = 0
        failed = 0
        results = []
        
        # Split into batches
        batches = self._create_batches(items)
        
        for batch_idx, batch in enumerate(batches):
            if progress_callback:
                progress_callback(batch_idx * self.current_batch_size, total_items)
            
            # Process batch
            batch_results = await self._process_batch(
                batch,
                process_func,
                max_retries
            )
            
            # Update statistics
            for result in batch_results:
                results.append(result)
                if result.get("success", False):
                    successful += 1
                else:
                    failed += 1
            
            # Adaptive adjustment
            if self.adaptive and len(batch_results) > 0:
                self._adjust_batch_size(batch_results)
        
        total_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(total_items, total_items)
        
        return BatchResult(
            total_items=total_items,
            successful=successful,
            failed=failed,
            total_time=total_time,
            avg_time_per_item=total_time / total_items if total_items > 0 else 0.0,
            results=results
        )
    
    def _create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Create optimized batches."""
        batches = []
        for i in range(0, len(items), self.current_batch_size):
            batch = items[i:i + self.current_batch_size]
            batches.append(batch)
        return batches
    
    async def _process_batch(
        self,
        batch: List[Any],
        process_func: Callable,
        max_retries: int
    ) -> List[Dict[str, Any]]:
        """Process a single batch."""
        tasks = []
        for item in batch:
            task = self._process_with_retry(item, process_func, max_retries)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert to result format
        formatted_results = []
        for result in results:
            if isinstance(result, Exception):
                formatted_results.append({
                    "success": False,
                    "error": str(result)
                })
            else:
                formatted_results.append(result)
        
        return formatted_results
    
    async def _process_with_retry(
        self,
        item: Any,
        process_func: Callable,
        max_retries: int
    ) -> Dict[str, Any]:
        """Process item with retry logic."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await process_func(item)
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1
                }
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
        
        return {
            "success": False,
            "error": str(last_error),
            "attempts": max_retries + 1
        }
    
    def _adjust_batch_size(self, batch_results: List[Dict[str, Any]]) -> None:
        """Adjust batch size based on results."""
        if not batch_results:
            return
        
        # Calculate success rate
        success_count = sum(1 for r in batch_results if r.get("success", False))
        success_rate = success_count / len(batch_results)
        
        # Record batch performance
        batch_time = sum(r.get("processing_time", 0.0) for r in batch_results)
        self.batch_history.append({
            "batch_size": self.current_batch_size,
            "success_rate": success_rate,
            "total_time": batch_time
        })
        
        # Adjust based on success rate
        if success_rate > 0.9 and self.current_batch_size < self.max_batch_size:
            # High success, can increase
            self.current_batch_size = min(
                self.max_batch_size,
                self.current_batch_size + 1
            )
        elif success_rate < 0.7 and self.current_batch_size > self.min_batch_size:
            # Low success, decrease
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size - 1
            )
    
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self.current_batch_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        if not self.batch_history:
            return {
                "current_batch_size": self.current_batch_size,
                "total_batches": 0,
                "avg_success_rate": 0.0
            }
        
        return {
            "current_batch_size": self.current_batch_size,
            "total_batches": len(self.batch_history),
            "avg_success_rate": sum(h["success_rate"] for h in self.batch_history) / len(self.batch_history),
            "avg_batch_time": sum(h["total_time"] for h in self.batch_history) / len(self.batch_history)
        }


