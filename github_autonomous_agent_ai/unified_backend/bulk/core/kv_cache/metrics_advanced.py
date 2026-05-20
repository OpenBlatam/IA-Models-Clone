"""
Advanced metrics for KV Cache.

Provides detailed metrics and analysis capabilities.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict

from kv_cache.types import StatsDict

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """
    Advanced metrics collector and analyzer.
    
    Provides detailed metrics, trends, and analysis.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize advanced metrics.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        
        # Time series data
        self.hit_rate_history: deque = deque(maxlen=window_size)
        self.latency_history: deque = deque(maxlen=window_size)
        self.memory_history: deque = deque(maxlen=window_size)
        self.eviction_history: deque = deque(maxlen=window_size)
        
        # Per-operation metrics
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        
        # Anomaly detection
        self.anomalies: List[Dict[str, Any]] = []
        
        # Timestamps
        self.start_time = time.time()
    
    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True
    ) -> None:
        """
        Record operation metrics.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            success: Whether operation succeeded
        """
        self.operation_times[operation].append(duration)
        if len(self.operation_times[operation]) > self.window_size:
            self.operation_times[operation].pop(0)
        
        self.operation_counts[operation] += 1
        
        if not success:
            self.operation_counts[f"{operation}_failed"] = (
                self.operation_counts.get(f"{operation}_failed", 0) + 1
            )
    
    def record_hit_rate(self, hit_rate: float) -> None:
        """Record hit rate."""
        self.hit_rate_history.append(hit_rate)
    
    def record_latency(self, latency: float) -> None:
        """Record latency."""
        self.latency_history.append(latency)
    
    def record_memory(self, memory_mb: float) -> None:
        """Record memory usage."""
        self.memory_history.append(memory_mb)
    
    def record_eviction(self, count: int) -> None:
        """Record eviction count."""
        self.eviction_history.append(count)
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get statistics for specific operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Dictionary with operation statistics
        """
        if operation not in self.operation_times:
            return {}
        
        times = self.operation_times[operation]
        if not times:
            return {}
        
        sorted_times = sorted(times)
        n = len(sorted_times)
        
        return {
            "count": self.operation_counts[operation],
            "failed": self.operation_counts.get(f"{operation}_failed", 0),
            "mean": sum(times) / n,
            "median": sorted_times[n // 2],
            "min": min(times),
            "max": max(times),
            "p95": sorted_times[int(n * 0.95)] if n > 20 else sorted_times[-1],
            "p99": sorted_times[int(n * 0.99)] if n > 100 else sorted_times[-1],
            "std": (
                sum((t - sum(times) / n) ** 2 for t in times) / n
            ) ** 0.5 if n > 1 else 0.0
        }
    
    def get_all_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all operations.
        
        Returns:
            Dictionary mapping operation -> stats
        """
        return {
            op: self.get_operation_stats(op)
            for op in self.operation_times.keys()
        }
    
    def get_trends(self) -> Dict[str, Any]:
        """
        Get trend analysis.
        
        Returns:
            Dictionary with trend information
        """
        trends = {}
        
        # Hit rate trend
        if len(self.hit_rate_history) > 10:
            recent = list(self.hit_rate_history)[-10:]
            older = list(self.hit_rate_history)[-20:-10] if len(self.hit_rate_history) > 20 else []
            
            if older:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                trends["hit_rate_trend"] = "increasing" if recent_avg > older_avg else "decreasing"
                trends["hit_rate_change"] = recent_avg - older_avg
            else:
                trends["hit_rate_trend"] = "stable"
                trends["hit_rate_change"] = 0.0
        
        # Latency trend
        if len(self.latency_history) > 10:
            recent = list(self.latency_history)[-10:]
            older = list(self.latency_history)[-20:-10] if len(self.latency_history) > 20 else []
            
            if older:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                trends["latency_trend"] = "increasing" if recent_avg > older_avg else "decreasing"
                trends["latency_change"] = recent_avg - older_avg
            else:
                trends["latency_trend"] = "stable"
                trends["latency_change"] = 0.0
        
        # Memory trend
        if len(self.memory_history) > 10:
            recent = list(self.memory_history)[-10:]
            older = list(self.memory_history)[-20:-10] if len(self.memory_history) > 20 else []
            
            if older:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                trends["memory_trend"] = "increasing" if recent_avg > older_avg else "decreasing"
                trends["memory_change"] = recent_avg - older_avg
            else:
                trends["memory_trend"] = "stable"
                trends["memory_change"] = 0.0
        
        return trends
    
    def detect_anomalies(
        self,
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metrics.
        
        Args:
            threshold_std: Standard deviation threshold for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        current_time = time.time()
        
        # Check hit rate anomalies
        if len(self.hit_rate_history) > 20:
            recent = list(self.hit_rate_history)[-20:]
            mean = sum(recent) / len(recent)
            std = (sum((x - mean) ** 2 for x in recent) / len(recent)) ** 0.5
            
            if std > 0:
                latest = self.hit_rate_history[-1]
                if abs(latest - mean) > threshold_std * std:
                    anomalies.append({
                        "type": "hit_rate_anomaly",
                        "value": latest,
                        "expected_range": (mean - threshold_std * std, mean + threshold_std * std),
                        "timestamp": current_time
                    })
        
        # Check latency anomalies
        if len(self.latency_history) > 20:
            recent = list(self.latency_history)[-20:]
            mean = sum(recent) / len(recent)
            std = (sum((x - mean) ** 2 for x in recent) / len(recent)) ** 0.5
            
            if std > 0:
                latest = self.latency_history[-1]
                if abs(latest - mean) > threshold_std * std:
                    anomalies.append({
                        "type": "latency_anomaly",
                        "value": latest,
                        "expected_range": (mean - threshold_std * std, mean + threshold_std * std),
                        "timestamp": current_time
                    })
        
        self.anomalies.extend(anomalies)
        return anomalies
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary with comprehensive metrics
        """
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "operations": self.get_all_operation_stats(),
            "trends": self.get_trends(),
            "hit_rate": {
                "current": self.hit_rate_history[-1] if self.hit_rate_history else 0.0,
                "average": sum(self.hit_rate_history) / len(self.hit_rate_history) if self.hit_rate_history else 0.0,
                "min": min(self.hit_rate_history) if self.hit_rate_history else 0.0,
                "max": max(self.hit_rate_history) if self.hit_rate_history else 0.0
            },
            "latency": {
                "current": self.latency_history[-1] if self.latency_history else 0.0,
                "average": sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0.0,
                "min": min(self.latency_history) if self.latency_history else 0.0,
                "max": max(self.latency_history) if self.latency_history else 0.0
            },
            "memory": {
                "current": self.memory_history[-1] if self.memory_history else 0.0,
                "average": sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0.0,
                "min": min(self.memory_history) if self.memory_history else 0.0,
                "max": max(self.memory_history) if self.memory_history else 0.0
            },
            "anomalies_detected": len(self.anomalies),
            "recent_anomalies": self.anomalies[-10:]
        }

