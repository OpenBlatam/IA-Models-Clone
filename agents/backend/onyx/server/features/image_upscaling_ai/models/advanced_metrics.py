"""
Advanced Metrics System
=======================

Comprehensive metrics collection and analysis.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import statistics
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_id: str
    timestamp: float
    image_type: str
    scale_factor: float
    method: str
    processing_time: float
    quality_score: float
    memory_usage_mb: float
    cache_hit: bool
    success: bool
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_processing_time: float
    avg_processing_time: float
    avg_quality_score: float
    cache_hit_rate: float
    throughput: float  # Operations per second
    methods_usage: Dict[str, int]
    quality_distribution: Dict[str, int]
    error_rate: float


class AdvancedMetricsCollector:
    """
    Advanced metrics collection and analysis.
    
    Features:
    - Operation tracking
    - Performance analysis
    - Quality trends
    - Error tracking
    - System health monitoring
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        metrics_file: Optional[str] = None
    ):
        """
        Initialize metrics collector.
        
        Args:
            history_size: Size of metrics history
            metrics_file: File to save metrics
        """
        self.history_size = history_size
        self.metrics_file = metrics_file
        
        # Metrics storage
        self.operations: deque = deque(maxlen=history_size)
        self.operation_counter = 0
        
        # Aggregated metrics
        self.methods_usage = defaultdict(int)
        self.quality_buckets = defaultdict(int)
        self.error_types = defaultdict(int)
        
        logger.info("AdvancedMetricsCollector initialized")
    
    def record_operation(
        self,
        image_type: str,
        scale_factor: float,
        method: str,
        processing_time: float,
        quality_score: float,
        memory_usage_mb: float = 0.0,
        cache_hit: bool = False,
        success: bool = True,
        error: Optional[str] = None
    ) -> str:
        """
        Record an operation.
        
        Args:
            image_type: Type of image
            scale_factor: Scale factor
            method: Method used
            processing_time: Processing time
            quality_score: Quality score
            memory_usage_mb: Memory usage
            cache_hit: Whether cache was hit
            success: Whether operation succeeded
            error: Error message if failed
            
        Returns:
            Operation ID
        """
        self.operation_counter += 1
        operation_id = f"op_{self.operation_counter}_{int(time.time())}"
        
        metrics = OperationMetrics(
            operation_id=operation_id,
            timestamp=time.time(),
            image_type=image_type,
            scale_factor=scale_factor,
            method=method,
            processing_time=processing_time,
            quality_score=quality_score,
            memory_usage_mb=memory_usage_mb,
            cache_hit=cache_hit,
            success=success,
            error=error
        )
        
        self.operations.append(metrics)
        
        # Update aggregates
        if success:
            self.methods_usage[method] += 1
            self._update_quality_bucket(quality_score)
        else:
            if error:
                self.error_types[error] += 1
        
        return operation_id
    
    def _update_quality_bucket(self, quality: float) -> None:
        """Update quality distribution."""
        if quality >= 0.9:
            bucket = "excellent"
        elif quality >= 0.8:
            bucket = "very_good"
        elif quality >= 0.7:
            bucket = "good"
        elif quality >= 0.6:
            bucket = "fair"
        else:
            bucket = "poor"
        
        self.quality_buckets[bucket] += 1
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics."""
        if not self.operations:
            return SystemMetrics(
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                total_processing_time=0.0,
                avg_processing_time=0.0,
                avg_quality_score=0.0,
                cache_hit_rate=0.0,
                throughput=0.0,
                methods_usage={},
                quality_distribution={},
                error_rate=0.0
            )
        
        successful = [op for op in self.operations if op.success]
        failed = [op for op in self.operations if not op.success]
        
        total_time = sum(op.processing_time for op in self.operations)
        avg_time = total_time / len(self.operations) if self.operations else 0.0
        
        avg_quality = (
            statistics.mean([op.quality_score for op in successful])
            if successful else 0.0
        )
        
        cache_hits = sum(1 for op in self.operations if op.cache_hit)
        cache_hit_rate = cache_hits / len(self.operations) if self.operations else 0.0
        
        # Calculate throughput (operations per second)
        if len(self.operations) > 1:
            time_span = self.operations[-1].timestamp - self.operations[0].timestamp
            throughput = len(self.operations) / time_span if time_span > 0 else 0.0
        else:
            throughput = 0.0
        
        error_rate = len(failed) / len(self.operations) if self.operations else 0.0
        
        return SystemMetrics(
            total_operations=len(self.operations),
            successful_operations=len(successful),
            failed_operations=len(failed),
            total_processing_time=total_time,
            avg_processing_time=avg_time,
            avg_quality_score=avg_quality,
            cache_hit_rate=cache_hit_rate,
            throughput=throughput,
            methods_usage=dict(self.methods_usage),
            quality_distribution=dict(self.quality_buckets),
            error_rate=error_rate
        )
    
    def get_method_performance(self, method: str) -> Dict[str, Any]:
        """Get performance metrics for a specific method."""
        method_ops = [op for op in self.operations if op.method == method and op.success]
        
        if not method_ops:
            return {
                "method": method,
                "count": 0,
                "avg_time": 0.0,
                "avg_quality": 0.0,
                "success_rate": 0.0
            }
        
        return {
            "method": method,
            "count": len(method_ops),
            "avg_time": statistics.mean([op.processing_time for op in method_ops]),
            "avg_quality": statistics.mean([op.quality_score for op in method_ops]),
            "success_rate": len(method_ops) / max(1, sum(1 for op in self.operations if op.method == method)),
            "min_quality": min(op.quality_score for op in method_ops),
            "max_quality": max(op.quality_score for op in method_ops),
        }
    
    def get_quality_trends(self, window_size: int = 50) -> Dict[str, Any]:
        """Get quality trends over time."""
        if len(self.operations) < window_size:
            window_size = len(self.operations)
        
        if window_size == 0:
            return {
                "trend": "stable",
                "recent_avg": 0.0,
                "overall_avg": 0.0,
                "improvement": 0.0
            }
        
        recent_ops = list(self.operations)[-window_size:]
        recent_successful = [op for op in recent_ops if op.success]
        
        recent_avg = (
            statistics.mean([op.quality_score for op in recent_successful])
            if recent_successful else 0.0
        )
        
        overall_successful = [op for op in self.operations if op.success]
        overall_avg = (
            statistics.mean([op.quality_score for op in overall_successful])
            if overall_successful else 0.0
        )
        
        improvement = recent_avg - overall_avg
        
        if improvement > 0.05:
            trend = "improving"
        elif improvement < -0.05:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_avg": recent_avg,
            "overall_avg": overall_avg,
            "improvement": improvement,
            "window_size": window_size
        }
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis."""
        failed_ops = [op for op in self.operations if not op.success]
        
        if not failed_ops:
            return {
                "total_errors": 0,
                "error_types": {},
                "most_common_error": None,
                "error_rate": 0.0
            }
        
        return {
            "total_errors": len(failed_ops),
            "error_types": dict(self.error_types),
            "most_common_error": max(self.error_types.items(), key=lambda x: x[1])[0] if self.error_types else None,
            "error_rate": len(failed_ops) / len(self.operations) if self.operations else 0.0
        }
    
    def export_metrics(self, file_path: str) -> None:
        """Export metrics to file."""
        try:
            data = {
                "system_metrics": asdict(self.get_system_metrics()),
                "method_performance": {
                    method: self.get_method_performance(method)
                    for method in set(op.method for op in self.operations)
                },
                "quality_trends": self.get_quality_trends(),
                "error_analysis": self.get_error_analysis(),
                "export_timestamp": time.time()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.operations.clear()
        self.operation_counter = 0
        self.methods_usage.clear()
        self.quality_buckets.clear()
        self.error_types.clear()
        logger.info("Metrics reset")


