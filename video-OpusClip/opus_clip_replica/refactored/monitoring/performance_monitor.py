"""
Performance Monitor for Refactored Opus Clip

Advanced performance monitoring with:
- Real-time metrics collection
- Performance analysis
- Resource usage tracking
- Alerting system
- Performance optimization suggestions
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable
import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import structlog
import json
from pathlib import Path

logger = structlog.get_logger("performance_monitor")

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    timestamp: datetime
    severity: str  # info, warning, error, critical
    message: str
    metric_name: str
    threshold: float
    current_value: float
    resolved: bool = False

class PerformanceMonitor:
    """
    Advanced performance monitor for Opus Clip.
    
    Features:
    - Real-time metrics collection
    - Performance analysis
    - Resource usage tracking
    - Alerting system
    - Performance optimization suggestions
    """
    
    def __init__(self, 
                 max_metrics_history: int = 10000,
                 alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
                 enable_file_logging: bool = True,
                 log_file_path: str = "performance.log"):
        """Initialize performance monitor."""
        self.max_metrics_history = max_metrics_history
        self.enable_file_logging = enable_file_logging
        self.log_file_path = log_file_path
        self.logger = structlog.get_logger("performance_monitor")
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.current_metrics: Dict[str, float] = {}
        
        # Alert system
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "peak_memory_usage": 0.0,
            "peak_cpu_usage": 0.0,
            "total_processing_time": 0.0
        }
        
        # Monitoring control
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.monitor_interval = 1.0  # seconds
        
        # File logging
        if self.enable_file_logging:
            self.log_file = Path(self.log_file_path)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Performance monitor initialized")
    
    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default alert thresholds."""
        return {
            "cpu_usage": {"warning": 70.0, "error": 90.0, "critical": 95.0},
            "memory_usage": {"warning": 80.0, "error": 90.0, "critical": 95.0},
            "disk_usage": {"warning": 80.0, "error": 90.0, "critical": 95.0},
            "response_time": {"warning": 5.0, "error": 10.0, "critical": 30.0},
            "error_rate": {"warning": 5.0, "error": 10.0, "critical": 20.0},
            "queue_size": {"warning": 100, "error": 500, "critical": 1000}
        }
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Log metrics if enabled
                if self.enable_file_logging:
                    await self._log_metrics()
                
                # Wait for next interval
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            await self._record_metric("cpu_usage", cpu_percent, "percent")
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self._record_metric("memory_usage", memory.percent, "percent")
            await self._record_metric("memory_available", memory.available / (1024**3), "GB")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self._record_metric("disk_usage", disk_percent, "percent")
            await self._record_metric("disk_free", disk.free / (1024**3), "GB")
            
            # Network I/O
            net_io = psutil.net_io_counters()
            await self._record_metric("network_bytes_sent", net_io.bytes_sent, "bytes")
            await self._record_metric("network_bytes_recv", net_io.bytes_recv, "bytes")
            
            # Process-specific metrics
            process = psutil.Process()
            await self._record_metric("process_memory", process.memory_info().rss / (1024**2), "MB")
            await self._record_metric("process_cpu", process.cpu_percent(), "percent")
            
            # Update performance stats
            self.performance_stats["peak_memory_usage"] = max(
                self.performance_stats["peak_memory_usage"], 
                memory.percent
            )
            self.performance_stats["peak_cpu_usage"] = max(
                self.performance_stats["peak_cpu_usage"], 
                cpu_percent
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    async def _record_metric(self, name: str, value: float, unit: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        self.metrics_history.append(metric)
        self.current_metrics[name] = value
    
    async def _check_alerts(self):
        """Check for performance alerts."""
        try:
            for metric_name, thresholds in self.alert_thresholds.items():
                if metric_name not in self.current_metrics:
                    continue
                
                current_value = self.current_metrics[metric_name]
                
                # Check each threshold level
                for level, threshold in thresholds.items():
                    if current_value >= threshold:
                        await self._trigger_alert(
                            metric_name, level, threshold, current_value
                        )
                        break  # Only trigger highest level alert
                
        except Exception as e:
            self.logger.error(f"Failed to check alerts: {e}")
    
    async def _trigger_alert(self, metric_name: str, level: str, threshold: float, current_value: float):
        """Trigger a performance alert."""
        alert_id = f"{metric_name}_{level}_{int(time.time())}"
        
        # Check if alert already exists
        if alert_id in self.active_alerts:
            return
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=level,
            message=f"{metric_name} is {current_value:.2f} (threshold: {threshold:.2f})",
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value
        )
        
        self.active_alerts[alert_id] = alert
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        self.logger.warning(f"Performance alert: {alert.message}")
    
    async def _log_metrics(self):
        """Log metrics to file."""
        try:
            if not self.metrics_history:
                return
            
            # Get recent metrics
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "metrics": [
                    {
                        "name": m.metric_name,
                        "value": m.value,
                        "unit": m.unit,
                        "timestamp": m.timestamp.isoformat()
                    }
                    for m in recent_metrics
                ]
            }
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    async def record_request(self, endpoint: str, response_time: float, success: bool):
        """Record API request metrics."""
        try:
            self.performance_stats["total_requests"] += 1
            
            if success:
                self.performance_stats["successful_requests"] += 1
            else:
                self.performance_stats["failed_requests"] += 1
            
            # Update average response time
            total_requests = self.performance_stats["total_requests"]
            current_avg = self.performance_stats["average_response_time"]
            new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
            self.performance_stats["average_response_time"] = new_avg
            
            # Record response time metric
            await self._record_metric("response_time", response_time, "seconds", {
                "endpoint": endpoint,
                "success": success
            })
            
            # Calculate error rate
            error_rate = (self.performance_stats["failed_requests"] / total_requests) * 100
            await self._record_metric("error_rate", error_rate, "percent")
            
        except Exception as e:
            self.logger.error(f"Failed to record request: {e}")
    
    async def record_processing_time(self, operation: str, duration: float):
        """Record processing time for operations."""
        try:
            self.performance_stats["total_processing_time"] += duration
            
            await self._record_metric("processing_time", duration, "seconds", {
                "operation": operation
            })
            
        except Exception as e:
            self.logger.error(f"Failed to record processing time: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        try:
            # Calculate recent averages
            recent_metrics = list(self.metrics_history)[-100:] if self.metrics_history else []
            
            recent_cpu = [m.value for m in recent_metrics if m.metric_name == "cpu_usage"]
            recent_memory = [m.value for m in recent_metrics if m.metric_name == "memory_usage"]
            recent_response_time = [m.value for m in recent_metrics if m.metric_name == "response_time"]
            
            return {
                "current_metrics": self.current_metrics.copy(),
                "performance_stats": self.performance_stats.copy(),
                "recent_averages": {
                    "cpu_usage": sum(recent_cpu) / len(recent_cpu) if recent_cpu else 0,
                    "memory_usage": sum(recent_memory) / len(recent_memory) if recent_memory else 0,
                    "response_time": sum(recent_response_time) / len(recent_response_time) if recent_response_time else 0
                },
                "active_alerts": len(self.active_alerts),
                "total_metrics_collected": len(self.metrics_history),
                "monitoring_active": self.monitoring
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    async def get_metrics_history(self, metric_name: Optional[str] = None, 
                                 hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            filtered_metrics = [
                m for m in self.metrics_history
                if m.timestamp >= cutoff_time and (metric_name is None or m.metric_name == metric_name)
            ]
            
            return [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "metric_name": m.metric_name,
                    "value": m.value,
                    "unit": m.unit,
                    "metadata": m.metadata
                }
                for m in filtered_metrics
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics history: {e}")
            return []
    
    async def get_alerts(self, severity: Optional[str] = None, 
                        resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get performance alerts."""
        try:
            filtered_alerts = list(self.active_alerts.values())
            
            if severity:
                filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
            
            if resolved is not None:
                filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
            
            return [
                {
                    "id": a.id,
                    "timestamp": a.timestamp.isoformat(),
                    "severity": a.severity,
                    "message": a.message,
                    "metric_name": a.metric_name,
                    "threshold": a.threshold,
                    "current_value": a.current_value,
                    "resolved": a.resolved
                }
                for a in filtered_alerts
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to resolve alert: {e}")
            return False
    
    async def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get performance optimization suggestions."""
        try:
            suggestions = []
            
            # Check CPU usage
            if self.current_metrics.get("cpu_usage", 0) > 80:
                suggestions.append({
                    "type": "cpu_optimization",
                    "priority": "high",
                    "message": "High CPU usage detected. Consider reducing concurrent operations or optimizing algorithms.",
                    "current_value": self.current_metrics.get("cpu_usage", 0),
                    "recommendation": "Reduce max_workers or implement CPU throttling"
                })
            
            # Check memory usage
            if self.current_metrics.get("memory_usage", 0) > 85:
                suggestions.append({
                    "type": "memory_optimization",
                    "priority": "high",
                    "message": "High memory usage detected. Consider implementing memory cleanup or reducing cache size.",
                    "current_value": self.current_metrics.get("memory_usage", 0),
                    "recommendation": "Implement memory cleanup or reduce cache TTL"
                })
            
            # Check response time
            if self.performance_stats["average_response_time"] > 5.0:
                suggestions.append({
                    "type": "response_time_optimization",
                    "priority": "medium",
                    "message": "High average response time detected. Consider optimizing processing or adding caching.",
                    "current_value": self.performance_stats["average_response_time"],
                    "recommendation": "Implement response caching or optimize processing algorithms"
                })
            
            # Check error rate
            error_rate = (self.performance_stats["failed_requests"] / max(self.performance_stats["total_requests"], 1)) * 100
            if error_rate > 5.0:
                suggestions.append({
                    "type": "error_rate_optimization",
                    "priority": "high",
                    "message": "High error rate detected. Check error handling and retry mechanisms.",
                    "current_value": error_rate,
                    "recommendation": "Improve error handling and implement better retry logic"
                })
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization suggestions: {e}")
            return []
    
    async def cleanup_old_data(self, max_age_hours: int = 24):
        """Clean up old metrics and alerts."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Clean up old metrics
            old_metrics_count = len(self.metrics_history)
            self.metrics_history = deque(
                [m for m in self.metrics_history if m.timestamp >= cutoff_time],
                maxlen=self.max_metrics_history
            )
            cleaned_metrics = old_metrics_count - len(self.metrics_history)
            
            # Clean up old alerts
            old_alerts = list(self.active_alerts.values())
            self.active_alerts = {
                alert_id: alert for alert_id, alert in self.active_alerts.items()
                if alert.timestamp >= cutoff_time
            }
            cleaned_alerts = len(old_alerts) - len(self.active_alerts)
            
            if cleaned_metrics > 0 or cleaned_alerts > 0:
                self.logger.info(f"Cleaned up {cleaned_metrics} old metrics and {cleaned_alerts} old alerts")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")


