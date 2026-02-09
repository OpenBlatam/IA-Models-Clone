"""
Performance Optimizer for Piel Mejorador AI SAM3
================================================

Advanced performance optimizations.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: float
    avg_response_time: float
    requests_per_second: float
    error_rate: float
    memory_usage_mb: float


class PerformanceOptimizer:
    """
    Optimizes performance based on metrics.
    
    Features:
    - Adaptive concurrency
    - Response time tracking
    - Automatic tuning
    - Performance recommendations
    """
    
    def __init__(self, initial_concurrency: int = 5):
        """
        Initialize performance optimizer.
        
        Args:
            initial_concurrency: Initial concurrency level
        """
        self.current_concurrency = initial_concurrency
        self.min_concurrency = 1
        self.max_concurrency = 20
        
        self._response_times: deque = deque(maxlen=100)
        self._error_count = 0
        self._success_count = 0
        self._last_optimization = time.time()
        self._optimization_interval = 60.0  # Optimize every 60 seconds
        
        self._stats = {
            "optimizations": 0,
            "concurrency_changes": 0,
        }
    
    def record_response(self, response_time: float, success: bool):
        """
        Record a response for optimization.
        
        Args:
            response_time: Response time in seconds
            success: Whether request was successful
        """
        self._response_times.append(response_time)
        
        if success:
            self._success_count += 1
        else:
            self._error_count += 1
    
    def get_optimal_concurrency(self) -> int:
        """
        Calculate optimal concurrency based on metrics.
        
        Returns:
            Optimal concurrency level
        """
        if not self._response_times:
            return self.current_concurrency
        
        avg_response_time = sum(self._response_times) / len(self._response_times)
        total_requests = self._success_count + self._error_count
        error_rate = self._error_count / total_requests if total_requests > 0 else 0
        
        # Adaptive algorithm
        if error_rate > 0.1:  # High error rate
            new_concurrency = max(
                self.min_concurrency,
                int(self.current_concurrency * 0.8)  # Reduce by 20%
            )
        elif avg_response_time < 1.0 and error_rate < 0.05:  # Good performance
            new_concurrency = min(
                self.max_concurrency,
                int(self.current_concurrency * 1.2)  # Increase by 20%
            )
        else:
            new_concurrency = self.current_concurrency
        
        # Clamp to bounds
        new_concurrency = max(self.min_concurrency, min(self.max_concurrency, new_concurrency))
        
        if new_concurrency != self.current_concurrency:
            logger.info(
                f"Optimizing concurrency: {self.current_concurrency} -> {new_concurrency} "
                f"(avg_time={avg_response_time:.2f}s, error_rate={error_rate:.1%})"
            )
            self.current_concurrency = new_concurrency
            self._stats["concurrency_changes"] += 1
        
        return new_concurrency
    
    async def optimize_periodically(self):
        """Optimize performance periodically."""
        while True:
            await asyncio.sleep(self._optimization_interval)
            
            optimal = self.get_optimal_concurrency()
            self._stats["optimizations"] += 1
            
            # Reset counters
            self._error_count = 0
            self._success_count = 0
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        avg_response_time = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times else 0
        )
        
        total = self._success_count + self._error_count
        error_rate = self._error_count / total if total > 0 else 0
        rps = len(self._response_times) / 60.0 if self._response_times else 0
        
        return PerformanceMetrics(
            timestamp=time.time(),
            avg_response_time=avg_response_time,
            requests_per_second=rps,
            error_rate=error_rate,
            memory_usage_mb=0  # Would get from memory optimizer
        )
    
    def get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if not self._response_times:
            return recommendations
        
        avg_time = sum(self._response_times) / len(self._response_times)
        total = self._success_count + self._error_count
        error_rate = self._error_count / total if total > 0 else 0
        
        if avg_time > 5.0:
            recommendations.append(
                f"Average response time is high ({avg_time:.2f}s). "
                "Consider reducing concurrency or optimizing processing."
            )
        
        if error_rate > 0.1:
            recommendations.append(
                f"Error rate is high ({error_rate:.1%}). "
                "Consider reducing concurrency or checking service health."
            )
        
        if self.current_concurrency >= self.max_concurrency:
            recommendations.append(
                "Concurrency is at maximum. Consider horizontal scaling."
            )
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            **self._stats,
            "current_concurrency": self.current_concurrency,
            "min_concurrency": self.min_concurrency,
            "max_concurrency": self.max_concurrency,
        }




