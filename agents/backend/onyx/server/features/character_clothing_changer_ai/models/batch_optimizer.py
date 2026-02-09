"""
Batch Optimizer for Flux2 Clothing Changer
==========================================

Advanced batch processing optimization.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategy."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"


@dataclass
class BatchResult:
    """Batch processing result."""
    batch_id: str
    items_processed: int
    success_count: int
    failure_count: int
    processing_time: float
    results: List[Any] = None
    errors: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.errors is None:
            self.errors = []


class BatchOptimizer:
    """Advanced batch processing optimizer."""
    
    def __init__(
        self,
        initial_batch_size: int = 4,
        max_batch_size: int = 32,
        min_batch_size: int = 1,
        strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
    ):
        """
        Initialize batch optimizer.
        
        Args:
            initial_batch_size: Initial batch size
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size
            strategy: Batch strategy
        """
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.strategy = strategy
        
        self.current_batch_size = initial_batch_size
        self.batch_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "processing_times": [],
            "throughput": [],
        }
    
    def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        batch_size: Optional[int] = None,
    ) -> BatchResult:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            processor: Processing function
            batch_size: Optional batch size
            
        Returns:
            Batch result
        """
        if batch_size is None:
            batch_size = self.current_batch_size
        
        batch_size = max(self.min_batch_size, min(batch_size, self.max_batch_size))
        
        batch_id = f"batch_{int(time.time() * 1000)}"
        start_time = time.time()
        
        results = []
        errors = []
        success_count = 0
        failure_count = 0
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                batch_results = processor(batch)
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
                success_count += len(batch)
            except Exception as e:
                error_info = {
                    "batch_index": i,
                    "batch_size": len(batch),
                    "error": str(e),
                }
                errors.append(error_info)
                failure_count += len(batch)
                logger.error(f"Batch processing error: {e}")
        
        processing_time = time.time() - start_time
        
        result = BatchResult(
            batch_id=batch_id,
            items_processed=len(items),
            success_count=success_count,
            failure_count=failure_count,
            processing_time=processing_time,
            results=results,
            errors=errors,
        )
        
        # Update metrics
        self._update_metrics(result)
        
        # Optimize batch size if adaptive
        if self.strategy == BatchStrategy.ADAPTIVE:
            self._optimize_batch_size(result)
        
        return result
    
    def _update_metrics(self, result: BatchResult) -> None:
        """
        Update performance metrics.
        
        Args:
            result: Batch result
        """
        self.batch_history.append({
            "batch_id": result.batch_id,
            "batch_size": self.current_batch_size,
            "processing_time": result.processing_time,
            "throughput": result.items_processed / result.processing_time if result.processing_time > 0 else 0,
            "success_rate": result.success_count / result.items_processed if result.items_processed > 0 else 0,
        })
        
        self.performance_metrics["processing_times"].append(result.processing_time)
        self.performance_metrics["throughput"].append(
            result.items_processed / result.processing_time if result.processing_time > 0 else 0
        )
        
        # Keep only last 1000 entries
        if len(self.batch_history) > 1000:
            self.batch_history = self.batch_history[-1000:]
            self.performance_metrics["processing_times"] = self.performance_metrics["processing_times"][-1000:]
            self.performance_metrics["throughput"] = self.performance_metrics["throughput"][-1000:]
    
    def _optimize_batch_size(self, result: BatchResult) -> None:
        """
        Optimize batch size based on performance.
        
        Args:
            result: Batch result
        """
        if len(self.batch_history) < 3:
            return
        
        # Get recent performance
        recent = self.batch_history[-5:]
        avg_throughput = sum(h["throughput"] for h in recent) / len(recent)
        avg_time = sum(h["processing_time"] for h in recent) / len(recent)
        
        # If throughput is increasing and time is reasonable, increase batch size
        if avg_throughput > 0 and avg_time < 10.0:
            if self.current_batch_size < self.max_batch_size:
                self.current_batch_size = min(
                    self.current_batch_size + 1,
                    self.max_batch_size
                )
                logger.debug(f"Increased batch size to {self.current_batch_size}")
        # If time is too high, decrease batch size
        elif avg_time > 30.0:
            if self.current_batch_size > self.min_batch_size:
                self.current_batch_size = max(
                    self.current_batch_size - 1,
                    self.min_batch_size
                )
                logger.debug(f"Decreased batch size to {self.current_batch_size}")
    
    def get_optimal_batch_size(self) -> int:
        """
        Get optimal batch size.
        
        Returns:
            Optimal batch size
        """
        if self.strategy == BatchStrategy.FIXED:
            return self.initial_batch_size
        elif self.strategy == BatchStrategy.DYNAMIC:
            return self.current_batch_size
        else:  # ADAPTIVE
            return self.current_batch_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch optimizer statistics."""
        if not self.batch_history:
            return {
                "current_batch_size": self.current_batch_size,
                "strategy": self.strategy.value,
                "total_batches": 0,
            }
        
        recent = self.batch_history[-10:]
        avg_throughput = sum(h["throughput"] for h in recent) / len(recent)
        avg_time = sum(h["processing_time"] for h in recent) / len(recent)
        avg_success_rate = sum(h["success_rate"] for h in recent) / len(recent)
        
        return {
            "current_batch_size": self.current_batch_size,
            "strategy": self.strategy.value,
            "total_batches": len(self.batch_history),
            "average_throughput": avg_throughput,
            "average_processing_time": avg_time,
            "average_success_rate": avg_success_rate,
        }


