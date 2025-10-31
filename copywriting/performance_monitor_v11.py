"""
Performance Monitor v11 for Ultra-Optimized Copywriting System

Comprehensive performance monitoring, analytics, and optimization system
for the v11 copywriting engine with real-time metrics and insights.
"""

import time
import asyncio
import psutil
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics."""
    REQUEST_COUNT = "request_count"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATIO = "cache_hit_ratio"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    BATCH_SIZE = "batch_size"
    QUEUE_SIZE = "queue_size"
    THROUGHPUT = "throughput"

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: datetime
    metrics: Dict[str, float]
    system_stats: Dict[str, Any]
    engine_stats: Dict[str, Any]

class PerformanceAnalyzer:
    """Analyzes performance data and provides insights."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.anomaly_thresholds = {
            "response_time_ms": 5000,  # 5 seconds
            "error_rate_percent": 10,   # 10%
            "memory_usage_percent": 90, # 90%
            "cpu_usage_percent": 95     # 95%
        }
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a new metric to the analyzer."""
        self.metrics_history.append(metric)
    
    def get_recent_metrics(self, minutes: int = 10) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def calculate_trends(self, metric_type: MetricType, minutes: int = 10) -> Dict[str, float]:
        """Calculate trends for a specific metric type."""
        recent_metrics = self.get_recent_metrics(minutes)
        metric_values = [m.value for m in recent_metrics if m.metric_type == metric_type]
        
        if not metric_values:
            return {"trend": "no_data", "change_percent": 0.0}
        
        if len(metric_values) < 2:
            return {"trend": "insufficient_data", "change_percent": 0.0}
        
        # Calculate trend
        first_half = metric_values[:len(metric_values)//2]
        second_half = metric_values[len(metric_values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        if avg_first == 0:
            change_percent = 0.0
        else:
            change_percent = ((avg_second - avg_first) / avg_first) * 100
        
        if change_percent > 5:
            trend = "increasing"
        elif change_percent < -5:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change_percent": round(change_percent, 2),
            "current_avg": round(avg_second, 2),
            "previous_avg": round(avg_first, 2)
        }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        anomalies = []
        recent_metrics = self.get_recent_metrics(5)  # Last 5 minutes
        
        for metric in recent_metrics:
            threshold = self.anomaly_thresholds.get(metric.metric_type.value, float('inf'))
            
            if metric.value > threshold:
                anomalies.append({
                    "metric_type": metric.metric_type.value,
                    "value": metric.value,
                    "threshold": threshold,
                    "timestamp": metric.timestamp.isoformat(),
                    "severity": "high" if metric.value > threshold * 1.5 else "medium"
                })
        
        return anomalies
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        recent_metrics = self.get_recent_metrics(10)
        
        if not recent_metrics:
            return {"error": "No metrics available"}
        
        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_type[metric.metric_type.value].append(metric.value)
        
        summary = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                summary[metric_type] = {
                    "current": round(values[-1], 2),
                    "average": round(sum(values) / len(values), 2),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                    "trend": self.calculate_trends(MetricType(metric_type), 10)
                }
        
        # Add anomaly detection
        anomalies = self.detect_anomalies()
        summary["anomalies"] = anomalies
        summary["anomaly_count"] = len(anomalies)
        
        return summary

class OptimizationRecommender:
    """Provides optimization recommendations based on performance data."""
    
    def __init__(self):
        self.recommendations = []
    
    def analyze_and_recommend(self, performance_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze performance and provide recommendations."""
        recommendations = []
        
        # Check response time
        if "response_time_ms" in performance_summary:
            response_time = performance_summary["response_time_ms"]["current"]
            if response_time > 3000:  # 3 seconds
                recommendations.append({
                    "type": "performance",
                    "priority": "high",
                    "issue": "High response time",
                    "current_value": f"{response_time}ms",
                    "recommendation": "Consider increasing batch size or reducing model complexity",
                    "action": "Increase batch_size parameter or enable GPU acceleration"
                })
        
        # Check error rate
        if "error_rate_percent" in performance_summary:
            error_rate = performance_summary["error_rate_percent"]["current"]
            if error_rate > 5:
                recommendations.append({
                    "type": "reliability",
                    "priority": "high",
                    "issue": "High error rate",
                    "current_value": f"{error_rate}%",
                    "recommendation": "Check system resources and enable circuit breaker",
                    "action": "Enable circuit breaker pattern and check memory usage"
                })
        
        # Check memory usage
        if "memory_usage_percent" in performance_summary:
            memory_usage = performance_summary["memory_usage_percent"]["current"]
            if memory_usage > 80:
                recommendations.append({
                    "type": "resource",
                    "priority": "medium",
                    "issue": "High memory usage",
                    "current_value": f"{memory_usage}%",
                    "recommendation": "Optimize memory usage and enable garbage collection",
                    "action": "Enable memory optimization and reduce cache size"
                })
        
        # Check cache hit ratio
        if "cache_hit_ratio_percent" in performance_summary:
            cache_hit_ratio = performance_summary["cache_hit_ratio_percent"]["current"]
            if cache_hit_ratio < 50:
                recommendations.append({
                    "type": "efficiency",
                    "priority": "medium",
                    "issue": "Low cache hit ratio",
                    "current_value": f"{cache_hit_ratio}%",
                    "recommendation": "Improve caching strategy",
                    "action": "Enable predictive caching and increase cache size"
                })
        
        return recommendations

class PerformanceMonitorV11:
    """Main performance monitoring system for v11."""
    
    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval
        self.analyzer = PerformanceAnalyzer()
        self.recommender = OptimizationRecommender()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # System monitoring
        self.system_stats_history = deque(maxlen=1000)
        self.engine_stats_history = deque(maxlen=1000)
        
        # Threading
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start the performance monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the performance monitoring system."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system stats
                system_stats = self._get_system_stats()
                self.system_stats_history.append(system_stats)
                
                # Collect engine stats (if available)
                engine_stats = self._get_engine_stats()
                if engine_stats:
                    self.engine_stats_history.append(engine_stats)
                
                # Add metrics to analyzer
                self._add_performance_metrics(system_stats, engine_stats)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        except Exception as e:
            logger.warning(f"Failed to get system stats: {e}")
            return {"error": "System stats unavailable"}
    
    def _get_engine_stats(self) -> Optional[Dict[str, Any]]:
        """Get engine statistics if available."""
        # This would be implemented to get stats from the v11 engine
        # For now, return None
        return None
    
    def _add_performance_metrics(self, system_stats: Dict[str, Any], engine_stats: Optional[Dict[str, Any]]):
        """Add performance metrics to the analyzer."""
        timestamp = datetime.now()
        
        # System metrics
        if "cpu_percent" in system_stats:
            self.analyzer.add_metric(PerformanceMetric(
                MetricType.CPU_USAGE,
                system_stats["cpu_percent"],
                timestamp,
                {"source": "system"}
            ))
        
        if "memory_percent" in system_stats:
            self.analyzer.add_metric(PerformanceMetric(
                MetricType.MEMORY_USAGE,
                system_stats["memory_percent"],
                timestamp,
                {"source": "system"}
            ))
        
        # Engine metrics (if available)
        if engine_stats:
            # Add engine-specific metrics here
            pass
    
    def track_request(self, response_time: float, cache_hit: bool = False, error: bool = False):
        """Track a single request."""
        with self._lock:
            self.request_count += 1
            self.total_response_time += response_time
            
            if error:
                self.error_count += 1
            
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            # Add metrics to analyzer
            timestamp = datetime.now()
            
            # Response time metric
            self.analyzer.add_metric(PerformanceMetric(
                MetricType.RESPONSE_TIME,
                response_time * 1000,  # Convert to milliseconds
                timestamp,
                {"cache_hit": cache_hit, "error": error}
            ))
            
            # Request count metric
            self.analyzer.add_metric(PerformanceMetric(
                MetricType.REQUEST_COUNT,
                self.request_count,
                timestamp
            ))
            
            # Error rate metric
            if self.request_count > 0:
                error_rate = (self.error_count / self.request_count) * 100
                self.analyzer.add_metric(PerformanceMetric(
                    MetricType.ERROR_RATE,
                    error_rate,
                    timestamp
                ))
            
            # Cache hit ratio metric
            total_cache_requests = self.cache_hits + self.cache_misses
            if total_cache_requests > 0:
                cache_hit_ratio = (self.cache_hits / total_cache_requests) * 100
                self.analyzer.add_metric(PerformanceMetric(
                    MetricType.CACHE_HIT_RATIO,
                    cache_hit_ratio,
                    timestamp
                ))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        performance_summary = self.analyzer.get_performance_summary()
        recommendations = self.recommender.analyze_and_recommend(performance_summary)
        
        # Calculate additional metrics
        with self._lock:
            avg_response_time = (self.total_response_time / max(self.request_count, 1)) * 1000
            error_rate = (self.error_count / max(self.request_count, 1)) * 100
            cache_hit_ratio = (self.cache_hits / max(self.cache_hits + self.cache_misses, 1)) * 100
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "average_response_time_ms": round(avg_response_time, 2),
                "error_rate_percent": round(error_rate, 2),
                "cache_hit_ratio_percent": round(cache_hit_ratio, 2)
            },
            "performance_analysis": performance_summary,
            "recommendations": recommendations,
            "system_stats": self._get_system_stats(),
            "monitoring_status": {
                "active": self.monitoring_active,
                "update_interval_seconds": self.update_interval
            }
        }
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        with self._lock:
            current_time = datetime.now()
            
            # Get recent metrics
            recent_metrics = self.analyzer.get_recent_metrics(1)  # Last minute
            
            # Calculate current rates
            requests_per_minute = len([m for m in recent_metrics if m.metric_type == MetricType.REQUEST_COUNT])
            errors_per_minute = len([m for m in recent_metrics if m.metric_type == MetricType.ERROR_RATE and m.value > 0])
            
            return {
                "timestamp": current_time.isoformat(),
                "current_rates": {
                    "requests_per_minute": requests_per_minute,
                    "errors_per_minute": errors_per_minute,
                    "success_rate_percent": round((requests_per_minute - errors_per_minute) / max(requests_per_minute, 1) * 100, 2)
                },
                "system_stats": self._get_system_stats(),
                "trends": {
                    "response_time": self.analyzer.calculate_trends(MetricType.RESPONSE_TIME, 5),
                    "error_rate": self.analyzer.calculate_trends(MetricType.ERROR_RATE, 5),
                    "cache_hit_ratio": self.analyzer.calculate_trends(MetricType.CACHE_HIT_RATIO, 5)
                }
            }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        report = self.get_performance_report()
        
        if format.lower() == "json":
            return json.dumps(report, indent=2, default=str)
        elif format.lower() == "csv":
            # Implement CSV export
            return "CSV export not implemented yet"
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitorV11:
    """Get the global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitorV11()
    
    return _performance_monitor

# Utility functions for easy monitoring
async def start_performance_monitoring():
    """Start performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    return monitor

async def stop_performance_monitoring():
    """Stop performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()

async def get_performance_report():
    """Get current performance report."""
    monitor = get_performance_monitor()
    return monitor.get_performance_report()

async def get_realtime_metrics():
    """Get real-time performance metrics."""
    monitor = get_performance_monitor()
    return monitor.get_realtime_metrics()

def track_request(response_time: float, cache_hit: bool = False, error: bool = False):
    """Track a single request."""
    monitor = get_performance_monitor()
    monitor.track_request(response_time, cache_hit, error) 