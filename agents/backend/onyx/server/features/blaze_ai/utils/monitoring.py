"""
Monitoring and observability utilities for Blaze AI.

This module provides comprehensive monitoring capabilities including:
- Performance monitoring
- Resource monitoring
- Training monitoring
- System monitoring
- Metrics collection and reporting
"""

from __future__ import annotations

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class MetricSeries:
    """Series of metric data points."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, tags: Optional[Dict[str, str]] = None):
        """Add a new data point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        self.points.append(point)
    
    def get_latest(self) -> Optional[float]:
        """Get the latest metric value."""
        if self.points:
            return self.points[-1].value
        return None
    
    def get_average(self, window_minutes: int = 5) -> Optional[float]:
        """Get average value over time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_points = [p.value for p in self.points if p.timestamp > cutoff_time]
        
        if recent_points:
            return sum(recent_points) / len(recent_points)
        return None

class PerformanceMonitor:
    """Performance monitoring and profiling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics: Dict[str, MetricSeries] = defaultdict(lambda: MetricSeries(""))
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval_seconds: int = 5):
        """Start performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval_seconds,), daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_metrics(self):
        """Collect current performance metrics."""
        # CPU usage
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("memory_usage_percent", memory.percent)
            self.record_metric("memory_used_gb", memory.used / (1024**3))
        
        # GPU metrics if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                
                self.record_metric(f"gpu_{i}_memory_allocated_gb", allocated, {"gpu_id": str(i)})
                self.record_metric(f"gpu_{i}_memory_reserved_gb", reserved, {"gpu_id": str(i)})
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name, tags=tags or {})
        
        self.metrics[name].add_point(value, tags)
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get a specific metric series."""
        return self.metrics.get(name)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for name, series in self.metrics.items():
            summary[name] = {
                "current": series.get_latest(),
                "average_5min": series.get_average(5),
                "tags": series.tags
            }
        
        return summary

class ResourceMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        self.resource_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}
        
        resources = {
            "timestamp": time.time(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "percent": psutil.virtual_memory().percent
            },
            "disk": {},
            "network": {}
        }
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                resources["disk"][partition.device] = {
                    "mountpoint": partition.mountpoint,
                    "total_gb": usage.total / (1024**3),
                    "used_gb": usage.used / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "percent": (usage.used / usage.total) * 100
                }
            except PermissionError:
                continue
        
        # Network usage
        net_io = psutil.net_io_counters()
        resources["network"] = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        # Store in history
        self.resource_history.append(resources)
        if len(self.resource_history) > self.max_history_size:
            self.resource_history.pop(0)
        
        return resources
    
    def get_resource_trends(self, minutes: int = 60) -> Dict[str, List[float]]:
        """Get resource usage trends over time."""
        cutoff_time = time.time() - (minutes * 60)
        recent_data = [entry for entry in self.resource_history if entry["timestamp"] > cutoff_time]
        
        if not recent_data:
            return {}
        
        trends = {
            "timestamps": [entry["timestamp"] for entry in recent_data],
            "cpu_percent": [entry["cpu"]["percent"] for entry in recent_data],
            "memory_percent": [entry["memory"]["percent"] for entry in recent_data]
        }
        
        return trends

class TrainingMonitor:
    """Training process monitoring."""
    
    def __init__(self):
        self.training_metrics: Dict[str, MetricSeries] = defaultdict(lambda: MetricSeries(""))
        self.current_epoch = 0
        self.current_step = 0
    
    def record_training_metric(self, name: str, value: float, epoch: Optional[int] = None, step: Optional[int] = None):
        """Record a training metric."""
        tags = {}
        if epoch is not None:
            tags["epoch"] = str(epoch)
            self.current_epoch = epoch
        if step is not None:
            tags["step"] = str(step)
            self.current_step = step
        
        if name not in self.training_metrics:
            self.training_metrics[name] = MetricSeries(name, tags=tags)
        
        self.training_metrics[name].add_point(value, tags)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        summary = {
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "metrics": {}
        }
        
        for name, series in self.training_metrics.items():
            summary["metrics"][name] = {
                "current": series.get_latest(),
                "average_epoch": series.get_average(),
                "tags": series.tags
            }
        
        return summary

class SystemMonitor:
    """Comprehensive system monitoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.performance_monitor = PerformanceMonitor(config)
        self.resource_monitor = ResourceMonitor()
        self.training_monitor = TrainingMonitor()
        self.alerts: List[Dict[str, Any]] = []
    
    def start_monitoring(self, interval_seconds: int = 5):
        """Start all monitoring systems."""
        self.performance_monitor.start_monitoring(interval_seconds)
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.performance_monitor.stop_monitoring()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "performance": self.performance_monitor.get_performance_summary(),
            "resources": self.resource_monitor.get_system_resources(),
            "training": self.training_monitor.get_training_summary(),
            "alerts": self.alerts
        }
    
    def add_alert(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Add a system alert."""
        alert = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "details": details or {}
        }
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for system alerts based on current metrics."""
        alerts = []
        
        # Check CPU usage
        cpu_metric = self.performance_monitor.get_metric("cpu_usage")
        if cpu_metric and cpu_metric.get_latest() > 90:
            alerts.append({
                "level": "warning",
                "message": "High CPU usage detected",
                "value": cpu_metric.get_latest()
            })
        
        # Check memory usage
        memory_metric = self.performance_monitor.get_metric("memory_usage_percent")
        if memory_metric and memory_metric.get_latest() > 85:
            alerts.append({
                "level": "warning",
                "message": "High memory usage detected",
                "value": memory_metric.get_latest()
            })
        
        return alerts

# Utility functions
def create_system_monitor(config: Optional[Dict[str, Any]] = None) -> SystemMonitor:
    """Create a new system monitor."""
    return SystemMonitor(config)

def start_monitoring(config: Optional[Dict[str, Any]] = None, interval_seconds: int = 5) -> SystemMonitor:
    """Quick start monitoring."""
    monitor = SystemMonitor(config)
    monitor.start_monitoring(interval_seconds)
    return monitor

# Export main classes
__all__ = [
    "PerformanceMonitor",
    "ResourceMonitor",
    "TrainingMonitor",
    "SystemMonitor",
    "MetricPoint",
    "MetricSeries",
    "create_system_monitor",
    "start_monitoring"
]
