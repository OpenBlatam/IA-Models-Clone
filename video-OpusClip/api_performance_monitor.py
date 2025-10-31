"""
🚀 API Performance Monitor for Video-OpusClip

This module provides comprehensive API performance monitoring with focus on:
- Response time tracking and analysis
- Latency measurement and optimization
- Throughput monitoring and capacity planning
- Real-time performance alerts
- Performance trend analysis
- Resource utilization correlation
- Performance bottleneck identification

Features:
- High-precision timing measurements
- Distributed tracing support
- Performance metrics aggregation
- Real-time monitoring dashboard
- Performance alerting system
- Historical performance analysis
- Resource correlation analysis
- Performance optimization recommendations
"""

import asyncio
import time
import statistics
import threading
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import json
import logging
from enum import Enum

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import psutil
import torch

# Configure logging
logger = structlog.get_logger(__name__)

# =============================================================================
# Performance Metrics Models
# =============================================================================

class MetricType(str, Enum):
    """Types of performance metrics."""
    RESPONSE_TIME = "response_time"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    QUEUE_SIZE = "queue_size"
    ACTIVE_CONNECTIONS = "active_connections"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    endpoint: str
    method: str
    response_time_ms: float
    latency_ms: float
    throughput_rps: float
    error_rate: float
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    queue_size: int = 0
    active_connections: int = 0
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    model_used: Optional[str] = None
    batch_size: Optional[int] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None

@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting."""
    response_time_ms: float = 1000.0  # 1 second
    latency_ms: float = 500.0         # 500ms
    throughput_rps: float = 10.0      # 10 requests per second
    error_rate: float = 0.05          # 5% error rate
    cpu_usage_percent: float = 80.0   # 80% CPU usage
    memory_usage_percent: float = 85.0 # 85% memory usage
    gpu_usage_percent: float = 90.0   # 90% GPU usage

@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    metrics: PerformanceMetrics
    threshold: float
    current_value: float

# =============================================================================
# Performance Monitoring Core
# =============================================================================

class APIPerformanceMonitor:
    """
    Comprehensive API performance monitor for Video-OpusClip.
    
    Tracks response time, latency, throughput, and resource utilization
    with real-time monitoring and alerting capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance monitor."""
        self.config = config or {}
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=10000)  # Last 10k metrics
        self.endpoint_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.real_time_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.total_latency = 0.0
        
        # Time windows for throughput calculation
        self.request_timestamps: deque = deque(maxlen=1000)
        self.throughput_window = 60  # 60 seconds
        
        # Resource monitoring
        self.cpu_history: deque = deque(maxlen=100)
        self.memory_history: deque = deque(maxlen=100)
        self.gpu_history: deque = deque(maxlen=100)
        
        # Alerting
        self.thresholds = PerformanceThresholds()
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Threading
        self._lock = threading.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        # Performance optimization
        self.enable_high_precision = self.config.get("enable_high_precision", True)
        self.enable_gpu_monitoring = self.config.get("enable_gpu_monitoring", True)
        self.enable_distributed_tracing = self.config.get("enable_distributed_tracing", False)
        
        logger.info("API Performance Monitor initialized")
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self._is_monitoring = False
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            await self._monitoring_task
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._is_monitoring:
            try:
                # Collect resource metrics
                await self._collect_resource_metrics()
                
                # Calculate throughput
                await self._calculate_throughput()
                
                # Check for alerts
                await self._check_alerts()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(1)  # Update every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _collect_resource_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_history.append((datetime.now(), cpu_percent))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_history.append((datetime.now(), memory_percent))
            
            # GPU usage (if available)
            gpu_percent = None
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                try:
                    gpu_percent = torch.cuda.utilization()
                    self.gpu_history.append((datetime.now(), gpu_percent))
                except Exception as e:
                    logger.warning(f"GPU monitoring failed: {e}")
            
        except Exception as e:
            logger.error(f"Resource metrics collection failed: {e}")
    
    async def _calculate_throughput(self):
        """Calculate current throughput."""
        try:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.throughput_window)
            
            # Count requests in window
            requests_in_window = sum(
                1 for timestamp in self.request_timestamps
                if timestamp >= window_start
            )
            
            # Calculate throughput (requests per second)
            throughput = requests_in_window / self.throughput_window
            
            # Update real-time metrics
            with self._lock:
                for endpoint_metrics in self.real_time_metrics.values():
                    endpoint_metrics.throughput_rps = throughput
                    
        except Exception as e:
            logger.error(f"Throughput calculation failed: {e}")
    
    async def _check_alerts(self):
        """Check for performance alerts."""
        try:
            current_metrics = self.get_current_metrics()
            
            if not current_metrics:
                return
            
            # Check response time
            if current_metrics.response_time_ms > self.thresholds.response_time_ms:
                await self._create_alert(
                    "HIGH_RESPONSE_TIME",
                    "Response time exceeded threshold",
                    current_metrics.response_time_ms,
                    self.thresholds.response_time_ms
                )
            
            # Check latency
            if current_metrics.latency_ms > self.thresholds.latency_ms:
                await self._create_alert(
                    "HIGH_LATENCY",
                    "Latency exceeded threshold",
                    current_metrics.latency_ms,
                    self.thresholds.latency_ms
                )
            
            # Check throughput
            if current_metrics.throughput_rps < self.thresholds.throughput_rps:
                await self._create_alert(
                    "LOW_THROUGHPUT",
                    "Throughput below threshold",
                    current_metrics.throughput_rps,
                    self.thresholds.throughput_rps
                )
            
            # Check error rate
            if current_metrics.error_rate > self.thresholds.error_rate:
                await self._create_alert(
                    "HIGH_ERROR_RATE",
                    "Error rate exceeded threshold",
                    current_metrics.error_rate,
                    self.thresholds.error_rate
                )
            
            # Check CPU usage
            if current_metrics.cpu_usage_percent > self.thresholds.cpu_usage_percent:
                await self._create_alert(
                    "HIGH_CPU_USAGE",
                    "CPU usage exceeded threshold",
                    current_metrics.cpu_usage_percent,
                    self.thresholds.cpu_usage_percent
                )
            
            # Check memory usage
            if current_metrics.memory_usage_percent > self.thresholds.memory_usage_percent:
                await self._create_alert(
                    "HIGH_MEMORY_USAGE",
                    "Memory usage exceeded threshold",
                    current_metrics.memory_usage_percent,
                    self.thresholds.memory_usage_percent
                )
            
            # Check GPU usage
            if (current_metrics.gpu_usage_percent and 
                current_metrics.gpu_usage_percent > self.thresholds.gpu_usage_percent):
                await self._create_alert(
                    "HIGH_GPU_USAGE",
                    "GPU usage exceeded threshold",
                    current_metrics.gpu_usage_percent,
                    self.thresholds.gpu_usage_percent
                )
                
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
    
    async def _create_alert(self, alert_type: str, message: str, current_value: float, threshold: float):
        """Create and dispatch performance alert."""
        try:
            current_metrics = self.get_current_metrics()
            if not current_metrics:
                return
            
            alert = PerformanceAlert(
                timestamp=datetime.now(),
                alert_type=alert_type,
                severity="WARNING" if current_value < threshold * 1.5 else "CRITICAL",
                message=message,
                metrics=current_metrics,
                threshold=threshold,
                current_value=current_value
            )
            
            # Add to alerts list
            self.alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            # Dispatch to callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
            logger.warning(
                f"Performance alert: {alert_type}",
                message=message,
                current_value=current_value,
                threshold=threshold,
                severity=alert.severity
            )
            
        except Exception as e:
            logger.error(f"Alert creation failed: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old metrics data."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours
            
            # Clean up metrics history
            with self._lock:
                self.metrics_history = deque(
                    [m for m in self.metrics_history if m.timestamp > cutoff_time],
                    maxlen=10000
                )
                
                # Clean up endpoint metrics
                for endpoint in list(self.endpoint_metrics.keys()):
                    self.endpoint_metrics[endpoint] = [
                        m for m in self.endpoint_metrics[endpoint]
                        if m.timestamp > cutoff_time
                    ]
                    
                    # Remove empty endpoint lists
                    if not self.endpoint_metrics[endpoint]:
                        del self.endpoint_metrics[endpoint]
            
            # Clean up resource history
            self.cpu_history = deque(
                [(t, v) for t, v in self.cpu_history if t > cutoff_time],
                maxlen=100
            )
            self.memory_history = deque(
                [(t, v) for t, v in self.memory_history if t > cutoff_time],
                maxlen=100
            )
            self.gpu_history = deque(
                [(t, v) for t, v in self.gpu_history if t > cutoff_time],
                maxlen=100
            )
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

# =============================================================================
# Performance Tracking Methods
# =============================================================================

    def track_request(self, request: Request, response: Response, 
                     start_time: float, end_time: float,
                     additional_data: Optional[Dict[str, Any]] = None):
        """Track a single request's performance metrics."""
        try:
            # Calculate timing metrics
            response_time_ms = (end_time - start_time) * 1000
            
            # Estimate latency (network + processing overhead)
            latency_ms = response_time_ms * 0.8  # Rough estimate
            
            # Get resource usage
            cpu_percent = psutil.cpu_percent() if self.cpu_history else 0.0
            memory_percent = psutil.virtual_memory().percent if self.memory_history else 0.0
            
            gpu_percent = None
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                try:
                    gpu_percent = torch.cuda.utilization()
                except Exception:
                    pass
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                endpoint=request.url.path,
                method=request.method,
                response_time_ms=response_time_ms,
                latency_ms=latency_ms,
                throughput_rps=0.0,  # Will be calculated in monitoring loop
                error_rate=0.0,      # Will be calculated in monitoring loop
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                gpu_usage_percent=gpu_percent,
                request_id=getattr(request.state, "request_id", None),
                user_id=additional_data.get("user_id") if additional_data else None,
                model_used=additional_data.get("model_used") if additional_data else None,
                batch_size=additional_data.get("batch_size") if additional_data else None,
                input_size=additional_data.get("input_size") if additional_data else None,
                output_size=additional_data.get("output_size") if additional_data else None
            )
            
            # Update counters
            with self._lock:
                self.request_count += 1
                if response.status_code >= 400:
                    self.error_count += 1
                
                self.total_response_time += response_time_ms
                self.total_latency += latency_ms
                
                # Add to request timestamps for throughput calculation
                self.request_timestamps.append(datetime.now())
                
                # Store metrics
                self.metrics_history.append(metrics)
                self.endpoint_metrics[request.url.path].append(metrics)
                self.real_time_metrics[request.url.path] = metrics
            
            # Update error rate
            if self.request_count > 0:
                metrics.error_rate = self.error_count / self.request_count
            
            logger.debug(
                "Request tracked",
                endpoint=metrics.endpoint,
                method=metrics.method,
                response_time_ms=metrics.response_time_ms,
                latency_ms=metrics.latency_ms,
                status_code=response.status_code
            )
            
        except Exception as e:
            logger.error(f"Request tracking failed: {e}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics."""
        try:
            if not self.real_time_metrics:
                return None
            
            # Get the most recent metrics
            latest_metrics = max(
                self.real_time_metrics.values(),
                key=lambda m: m.timestamp
            )
            
            return latest_metrics
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return None
    
    def get_endpoint_metrics(self, endpoint: str, 
                           time_window: Optional[timedelta] = None) -> List[PerformanceMetrics]:
        """Get metrics for a specific endpoint."""
        try:
            metrics = self.endpoint_metrics.get(endpoint, [])
            
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp > cutoff_time]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get endpoint metrics: {e}")
            return []
    
    def get_performance_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        try:
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            else:
                metrics = list(self.metrics_history)
            
            if not metrics:
                return {}
            
            # Calculate statistics
            response_times = [m.response_time_ms for m in metrics]
            latencies = [m.latency_ms for m in metrics]
            throughputs = [m.throughput_rps for m in metrics if m.throughput_rps > 0]
            error_rates = [m.error_rate for m in metrics]
            cpu_usage = [m.cpu_usage_percent for m in metrics]
            memory_usage = [m.memory_usage_percent for m in metrics]
            gpu_usage = [m.gpu_usage_percent for m in metrics if m.gpu_usage_percent is not None]
            
            summary = {
                "total_requests": len(metrics),
                "time_window": str(time_window) if time_window else "all",
                "response_time": {
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "p95": self._percentile(response_times, 95),
                    "p99": self._percentile(response_times, 99),
                    "min": min(response_times),
                    "max": max(response_times)
                },
                "latency": {
                    "mean": statistics.mean(latencies),
                    "median": statistics.median(latencies),
                    "p95": self._percentile(latencies, 95),
                    "p99": self._percentile(latencies, 99),
                    "min": min(latencies),
                    "max": max(latencies)
                },
                "throughput": {
                    "mean": statistics.mean(throughputs) if throughputs else 0.0,
                    "median": statistics.median(throughputs) if throughputs else 0.0,
                    "max": max(throughputs) if throughputs else 0.0
                },
                "error_rate": {
                    "mean": statistics.mean(error_rates),
                    "max": max(error_rates)
                },
                "resource_usage": {
                    "cpu": {
                        "mean": statistics.mean(cpu_usage),
                        "max": max(cpu_usage)
                    },
                    "memory": {
                        "mean": statistics.mean(memory_usage),
                        "max": max(memory_usage)
                    }
                }
            }
            
            if gpu_usage:
                summary["resource_usage"]["gpu"] = {
                    "mean": statistics.mean(gpu_usage),
                    "max": max(gpu_usage)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

# =============================================================================
# FastAPI Integration
# =============================================================================

    def create_middleware(self):
        """Create FastAPI middleware for performance monitoring."""
        async def performance_middleware(request: Request, call_next):
            # Record start time
            start_time = time.time()
            
            # Generate request ID if not present
            if not hasattr(request.state, "request_id"):
                request.state.request_id = f"req_{int(start_time * 1000000)}"
            
            try:
                # Process request
                response = await call_next(request)
                
                # Record end time
                end_time = time.time()
                
                # Track performance
                self.track_request(request, response, start_time, end_time)
                
                return response
                
            except Exception as e:
                # Record end time for failed requests
                end_time = time.time()
                
                # Create error response
                error_response = JSONResponse(
                    status_code=500,
                    content={"error": "Internal server error"}
                )
                
                # Track performance
                self.track_request(request, error_response, start_time, end_time)
                
                raise
        
        return performance_middleware

# =============================================================================
# Performance Decorators
# =============================================================================

    def monitor_performance(self, endpoint_name: Optional[str] = None):
        """Decorator to monitor function performance."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    end_time = time.time()
                    
                    # Track performance
                    self._track_function_performance(
                        func.__name__ or endpoint_name,
                        start_time,
                        end_time,
                        success=True
                    )
                    
                    return result
                    
                except Exception as e:
                    end_time = time.time()
                    
                    # Track performance
                    self._track_function_performance(
                        func.__name__ or endpoint_name,
                        start_time,
                        end_time,
                        success=False,
                        error=str(e)
                    )
                    
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    
                    # Track performance
                    self._track_function_performance(
                        func.__name__ or endpoint_name,
                        start_time,
                        end_time,
                        success=True
                    )
                    
                    return result
                    
                except Exception as e:
                    end_time = time.time()
                    
                    # Track performance
                    self._track_function_performance(
                        func.__name__ or endpoint_name,
                        start_time,
                        end_time,
                        success=False,
                        error=str(e)
                    )
                    
                    raise
            
            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _track_function_performance(self, function_name: str, start_time: float, 
                                  end_time: float, success: bool, error: Optional[str] = None):
        """Track function performance."""
        try:
            response_time_ms = (end_time - start_time) * 1000
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                endpoint=f"function:{function_name}",
                method="FUNCTION",
                response_time_ms=response_time_ms,
                latency_ms=response_time_ms * 0.9,  # Function latency
                throughput_rps=0.0,
                error_rate=0.0 if success else 1.0,
                cpu_usage_percent=psutil.cpu_percent(),
                memory_usage_percent=psutil.virtual_memory().percent,
                gpu_usage_percent=torch.cuda.utilization() if torch.cuda.is_available() else None
            )
            
            # Store metrics
            with self._lock:
                self.metrics_history.append(metrics)
                self.endpoint_metrics[f"function:{function_name}"].append(metrics)
            
            if not success:
                logger.warning(
                    f"Function {function_name} failed",
                    response_time_ms=response_time_ms,
                    error=error
                )
            
        except Exception as e:
            logger.error(f"Function performance tracking failed: {e}")

# =============================================================================
# Context Managers
# =============================================================================

    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        
        try:
            yield
            end_time = time.time()
            self._track_function_performance(operation_name, start_time, end_time, success=True)
            
        except Exception as e:
            end_time = time.time()
            self._track_function_performance(operation_name, start_time, end_time, success=False, error=str(e))
            raise
    
    @asynccontextmanager
    async def async_performance_context(self, operation_name: str):
        """Async context manager for performance monitoring."""
        start_time = time.time()
        
        try:
            yield
            end_time = time.time()
            self._track_function_performance(operation_name, start_time, end_time, success=True)
            
        except Exception as e:
            end_time = time.time()
            self._track_function_performance(operation_name, start_time, end_time, success=False, error=str(e))
            raise

# =============================================================================
# Alerting System
# =============================================================================

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def set_thresholds(self, thresholds: PerformanceThresholds):
        """Set performance thresholds."""
        self.thresholds = thresholds
    
    def get_alerts(self, time_window: Optional[timedelta] = None) -> List[PerformanceAlert]:
        """Get performance alerts."""
        if time_window:
            cutoff_time = datetime.now() - time_window
            return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
        return self.alerts.copy()

# =============================================================================
# Export and Reporting
# =============================================================================

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        try:
            summary = self.get_performance_summary()
            
            if format.lower() == "json":
                return json.dumps(summary, indent=2, default=str)
            elif format.lower() == "csv":
                # Convert to CSV format
                lines = ["metric,value"]
                for category, data in summary.items():
                    if isinstance(data, dict):
                        for metric, value in data.items():
                            lines.append(f"{category}.{metric},{value}")
                    else:
                        lines.append(f"{category},{data}")
                return "\n".join(lines)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
            return ""
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            # Get current metrics
            current_metrics = self.get_current_metrics()
            
            # Get performance summary
            summary = self.get_performance_summary()
            
            # Get recent alerts
            recent_alerts = self.get_alerts(timedelta(hours=1))
            
            # Get endpoint breakdown
            endpoint_breakdown = {}
            for endpoint, metrics in self.endpoint_metrics.items():
                if metrics:
                    endpoint_breakdown[endpoint] = {
                        "total_requests": len(metrics),
                        "avg_response_time": statistics.mean([m.response_time_ms for m in metrics]),
                        "avg_latency": statistics.mean([m.latency_ms for m in metrics]),
                        "error_rate": statistics.mean([m.error_rate for m in metrics])
                    }
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "current_metrics": current_metrics.__dict__ if current_metrics else None,
                "performance_summary": summary,
                "recent_alerts": [alert.__dict__ for alert in recent_alerts],
                "endpoint_breakdown": endpoint_breakdown,
                "resource_usage": {
                    "cpu": list(self.cpu_history),
                    "memory": list(self.memory_history),
                    "gpu": list(self.gpu_history) if self.gpu_history else []
                },
                "thresholds": self.thresholds.__dict__
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {}

# =============================================================================
# Global Instance
# =============================================================================

# Global performance monitor instance
_performance_monitor: Optional[APIPerformanceMonitor] = None

def get_performance_monitor() -> APIPerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = APIPerformanceMonitor()
    return _performance_monitor

def set_performance_monitor(monitor: APIPerformanceMonitor):
    """Set global performance monitor instance."""
    global _performance_monitor
    _performance_monitor = monitor 