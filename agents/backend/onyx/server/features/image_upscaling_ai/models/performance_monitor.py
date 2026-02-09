"""
Performance Monitor for Upscaling
==================================

Monitor and optimize performance of upscaling operations.
"""

import logging
import time
import psutil
import os
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""
    operation_name: str
    duration: float
    memory_used_mb: float
    memory_peak_mb: float
    cpu_percent: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Monitor performance of upscaling operations.
    
    Features:
    - Memory tracking
    - CPU usage monitoring
    - Duration measurement
    - Resource optimization suggestions
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self.metrics_history = []
        self.process = psutil.Process(os.getpid()) if enabled else None
    
    @contextmanager
    def monitor(self, operation_name: str):
        """
        Context manager to monitor an operation.
        
        Args:
            operation_name: Name of the operation
            
        Yields:
            PerformanceMetrics object
        """
        if not self.enabled:
            yield None
            return
        
        # Initial state
        start_time = time.time()
        start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        start_cpu = self.process.cpu_percent()
        
        peak_memory = start_memory
        success = True
        error = None
        
        try:
            yield
            
        except Exception as e:
            success = False
            error = str(e)
            raise
        
        finally:
            # Final state
            end_time = time.time()
            end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            end_cpu = self.process.cpu_percent()
            
            duration = end_time - start_time
            memory_used = end_memory - start_memory
            peak_memory = max(peak_memory, end_memory)
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration=duration,
                memory_used_mb=memory_used,
                memory_peak_mb=peak_memory,
                cpu_percent=end_cpu,
                success=success,
                details={
                    "error": error,
                    "start_memory_mb": start_memory,
                    "end_memory_mb": end_memory,
                }
            )
            
            self.metrics_history.append(metrics)
            
            if not success:
                logger.warning(
                    f"Operation '{operation_name}' failed after {duration:.2f}s "
                    f"(memory: {memory_used:.1f}MB)"
                )
            else:
                logger.debug(
                    f"Operation '{operation_name}' completed in {duration:.2f}s "
                    f"(memory: {memory_used:.1f}MB, CPU: {end_cpu:.1f}%)"
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics_history:
            return {"message": "No metrics recorded"}
        
        successful = [m for m in self.metrics_history if m.success]
        failed = [m for m in self.metrics_history if not m.success]
        
        if successful:
            avg_duration = sum(m.duration for m in successful) / len(successful)
            avg_memory = sum(m.memory_used_mb for m in successful) / len(successful)
            max_memory = max(m.memory_peak_mb for m in successful)
            avg_cpu = sum(m.cpu_percent for m in successful) / len(successful)
        else:
            avg_duration = 0.0
            avg_memory = 0.0
            max_memory = 0.0
            avg_cpu = 0.0
        
        return {
            "total_operations": len(self.metrics_history),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.metrics_history) if self.metrics_history else 0.0,
            "average_duration": round(avg_duration, 3),
            "average_memory_mb": round(avg_memory, 2),
            "peak_memory_mb": round(max_memory, 2),
            "average_cpu_percent": round(avg_cpu, 2),
        }
    
    def get_suggestions(self) -> list[str]:
        """Get performance optimization suggestions."""
        suggestions = []
        
        if not self.metrics_history:
            return suggestions
        
        successful = [m for m in self.metrics_history if m.success]
        
        if not successful:
            return suggestions
        
        # Check memory usage
        avg_memory = sum(m.memory_used_mb for m in successful) / len(successful)
        if avg_memory > 1000:
            suggestions.append(
                f"High memory usage ({avg_memory:.1f}MB). Consider reducing batch size or image resolution."
            )
        
        # Check duration
        avg_duration = sum(m.duration for m in successful) / len(successful)
        if avg_duration > 10:
            suggestions.append(
                f"Slow operations ({avg_duration:.1f}s). Consider using faster quality mode or reducing scale factor."
            )
        
        # Check CPU usage
        avg_cpu = sum(m.cpu_percent for m in successful) / len(successful)
        if avg_cpu > 80:
            suggestions.append(
                f"High CPU usage ({avg_cpu:.1f}%). Consider reducing parallel workers."
            )
        
        return suggestions
    
    def reset(self) -> None:
        """Reset metrics history."""
        self.metrics_history = []
        logger.info("Performance metrics reset")


