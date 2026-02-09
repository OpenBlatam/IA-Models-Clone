"""
Performance Monitor for Document Analyzer
==========================================

Advanced performance monitoring with real-time metrics and optimization suggestions.
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    timestamp: datetime
    value: float
    label: str
    tags: Dict[str, str] = field(default_factory=dict)

class DocumentPerformanceMonitor:
    """Advanced performance monitor for document analysis."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0
        
        # Default thresholds
        self.set_default_thresholds()
    
    def set_default_thresholds(self):
        """Set default performance thresholds."""
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 75.0, "critical": 90.0},
            "gpu_usage": {"warning": 80.0, "critical": 95.0},
            "analysis_time": {"warning": 5.0, "critical": 10.0},
            "error_rate": {"warning": 5.0, "critical": 10.0},
            "model_inference_time": {"warning": 2.0, "critical": 5.0}
        }
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            value=value,
            label=name,
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
        
        # Check thresholds
        self._check_thresholds(name, value)
    
    def _check_thresholds(self, metric_name: str, value: float):
        """Check if metric exceeds thresholds."""
        if metric_name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric_name]
        
        if value >= thresholds.get("critical", float('inf')):
            logger.error(f"CRITICAL: {metric_name} is critically high: {value}")
        elif value >= thresholds.get("warning", float('inf')):
            logger.warning(f"WARNING: {metric_name} is high: {value}")
    
    def get_metric_stats(self, name: str, window: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics for a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {}
        
        metrics = list(self.metrics[name])
        if window:
            metrics = metrics[-window:]
        
        values = [m.value for m in metrics]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
            "latest": values[-1] if values else None
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Record metrics
            self.record_metric("cpu_usage", cpu_percent)
            self.record_metric("memory_usage", memory.percent)
            
            metrics = {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # GPU metrics if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3  # GB
                    gpu_percent = (gpu_memory_used / gpu_memory) * 100 if gpu_memory > 0 else 0
                    
                    self.record_metric("gpu_usage", gpu_percent)
                    metrics["gpu"] = {
                        "available": torch.cuda.is_available(),
                        "memory_total_gb": gpu_memory,
                        "memory_used_gb": gpu_memory_used,
                        "percent": gpu_percent
                    }
            except ImportError:
                pass
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring."""
        if self.is_monitoring:
            return
        
        self.monitor_interval = interval
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    self.get_system_metrics()
                    time.sleep(self.monitor_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(self.monitor_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def record_analysis(self, task: str, duration: float, success: bool):
        """Record analysis task metrics."""
        self.record_metric(f"analysis_{task}_time", duration, tags={"task": task})
        self.record_metric(f"analysis_{task}_success", 1.0 if success else 0.0, tags={"task": task})
        
        if not success:
            # Calculate error rate
            error_metric = f"error_rate_{task}"
            if error_metric not in self.metrics:
                self.metrics[error_metric] = deque(maxlen=self.window_size)
            
            # Simple error rate calculation (can be improved)
            recent_metrics = list(self.metrics.get(f"analysis_{task}_success", deque()))[-100:]
            if recent_metrics:
                error_rate = (1.0 - sum(m.value for m in recent_metrics) / len(recent_metrics)) * 100
                self.record_metric(error_metric, error_rate, tags={"task": task})
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        return {
            "system": self.get_system_metrics(),
            "metrics": {
                name: self.get_metric_stats(name)
                for name in self.metrics.keys()
            },
            "timestamp": datetime.now().isoformat()
        }

# Global instance
performance_monitor = DocumentPerformanceMonitor()
















