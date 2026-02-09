"""
Performance Analyzer
====================

Advanced performance analysis and optimization recommendations.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metric type."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"


@dataclass
class PerformanceSnapshot:
    """Performance snapshot."""
    timestamp: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceRecommendation:
    """Performance recommendation."""
    metric: PerformanceMetric
    current_value: float
    recommended_value: float
    improvement_percent: float
    description: str
    priority: str = "medium"


class PerformanceAnalyzer:
    """Advanced performance analyzer."""
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.snapshots: List[PerformanceSnapshot] = []
        self.benchmarks: Dict[str, float] = {}
        self.thresholds: Dict[PerformanceMetric, Tuple[float, float]] = {}  # (warning, critical)
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self) -> None:
        """Setup default thresholds."""
        self.thresholds = {
            PerformanceMetric.LATENCY: (100.0, 500.0),  # ms
            PerformanceMetric.THROUGHPUT: (10.0, 5.0),  # requests/sec
            PerformanceMetric.MEMORY: (80.0, 90.0),  # percent
            PerformanceMetric.CPU: (80.0, 90.0),  # percent
            PerformanceMetric.ERROR_RATE: (1.0, 5.0),  # percent
            PerformanceMetric.SUCCESS_RATE: (95.0, 90.0),  # percent
        }
    
    def record_snapshot(
        self,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceSnapshot:
        """
        Record performance snapshot.
        
        Args:
            metrics: Performance metrics
            metadata: Optional metadata
            
        Returns:
            Performance snapshot
        """
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            metrics=metrics,
            metadata=metadata or {},
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only last 10000 snapshots
        if len(self.snapshots) > 10000:
            self.snapshots = self.snapshots[-10000:]
        
        return snapshot
    
    def set_benchmark(
        self,
        name: str,
        value: float,
    ) -> None:
        """
        Set performance benchmark.
        
        Args:
            name: Benchmark name
            value: Benchmark value
        """
        self.benchmarks[name] = value
        logger.info(f"Benchmark set: {name} = {value}")
    
    def analyze_performance(
        self,
        time_range: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze performance over time range.
        
        Args:
            time_range: Optional time range in seconds
            
        Returns:
            Performance analysis
        """
        if not self.snapshots:
            return {"error": "No snapshots available"}
        
        snapshots = self.snapshots
        
        if time_range:
            cutoff_time = time.time() - time_range
            snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        if not snapshots:
            return {"error": "No snapshots in time range"}
        
        # Aggregate metrics
        metric_values: Dict[str, List[float]] = {}
        for snapshot in snapshots:
            for metric_name, value in snapshot.metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)
        
        # Calculate statistics
        analysis = {}
        for metric_name, values in metric_values.items():
            analysis[metric_name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2],
            }
        
        return {
            "time_range": time_range,
            "snapshots_analyzed": len(snapshots),
            "metrics": analysis,
        }
    
    def get_recommendations(
        self,
        time_range: Optional[float] = None,
    ) -> List[PerformanceRecommendation]:
        """
        Get performance recommendations.
        
        Args:
            time_range: Optional time range in seconds
            
        Returns:
            List of recommendations
        """
        analysis = self.analyze_performance(time_range)
        
        if "error" in analysis:
            return []
        
        recommendations = []
        
        # Check each metric against thresholds
        for metric_name, stats in analysis.get("metrics", {}).items():
            try:
                metric = PerformanceMetric(metric_name)
            except ValueError:
                continue
            
            if metric not in self.thresholds:
                continue
            
            warning_threshold, critical_threshold = self.thresholds[metric]
            avg_value = stats["avg"]
            
            # Determine if improvement needed
            if metric in [PerformanceMetric.LATENCY, PerformanceMetric.MEMORY, PerformanceMetric.CPU, PerformanceMetric.ERROR_RATE]:
                # Lower is better
                if avg_value > critical_threshold:
                    recommended = critical_threshold * 0.8
                    improvement = ((avg_value - recommended) / avg_value) * 100
                    recommendations.append(PerformanceRecommendation(
                        metric=metric,
                        current_value=avg_value,
                        recommended_value=recommended,
                        improvement_percent=improvement,
                        description=f"{metric.value} is above critical threshold",
                        priority="high",
                    ))
                elif avg_value > warning_threshold:
                    recommended = warning_threshold * 0.9
                    improvement = ((avg_value - recommended) / avg_value) * 100
                    recommendations.append(PerformanceRecommendation(
                        metric=metric,
                        current_value=avg_value,
                        recommended_value=recommended,
                        improvement_percent=improvement,
                        description=f"{metric.value} is above warning threshold",
                        priority="medium",
                    ))
            else:  # THROUGHPUT, SUCCESS_RATE - higher is better
                if avg_value < critical_threshold:
                    recommended = critical_threshold * 1.2
                    improvement = ((recommended - avg_value) / avg_value) * 100
                    recommendations.append(PerformanceRecommendation(
                        metric=metric,
                        current_value=avg_value,
                        recommended_value=recommended,
                        improvement_percent=improvement,
                        description=f"{metric.value} is below critical threshold",
                        priority="high",
                    ))
                elif avg_value < warning_threshold:
                    recommended = warning_threshold * 1.1
                    improvement = ((recommended - avg_value) / avg_value) * 100
                    recommendations.append(PerformanceRecommendation(
                        metric=metric,
                        current_value=avg_value,
                        recommended_value=recommended,
                        improvement_percent=improvement,
                        description=f"{metric.value} is below warning threshold",
                        priority="medium",
                    ))
        
        return recommendations
    
    def compare_with_benchmark(
        self,
        metric_name: str,
        current_value: float,
    ) -> Dict[str, Any]:
        """
        Compare current value with benchmark.
        
        Args:
            metric_name: Metric name
            current_value: Current value
            
        Returns:
            Comparison result
        """
        if metric_name not in self.benchmarks:
            return {
                "benchmark_available": False,
            }
        
        benchmark = self.benchmarks[metric_name]
        difference = current_value - benchmark
        difference_percent = (difference / benchmark * 100) if benchmark > 0 else 0
        
        return {
            "benchmark_available": True,
            "benchmark": benchmark,
            "current": current_value,
            "difference": difference,
            "difference_percent": difference_percent,
            "status": "better" if difference < 0 else "worse" if difference > 0 else "equal",
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance analyzer statistics."""
        return {
            "total_snapshots": len(self.snapshots),
            "benchmarks": len(self.benchmarks),
            "thresholds": len(self.thresholds),
        }

