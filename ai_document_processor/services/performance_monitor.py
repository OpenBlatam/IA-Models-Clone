"""
Performance Monitoring Service for AI Document Processor
======================================================

Comprehensive performance monitoring with metrics collection,
health checks, and performance optimization recommendations.
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import weakref

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: str  # "healthy", "warning", "critical"
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Comprehensive performance monitoring service"""
    
    def __init__(self, 
                 metrics_retention_hours: int = 24,
                 health_check_interval: int = 60,
                 enable_system_metrics: bool = True):
        """
        Initialize performance monitor
        
        Args:
            metrics_retention_hours: How long to keep metrics in memory
            health_check_interval: Health check interval in seconds
            enable_system_metrics: Enable system-level metrics collection
        """
        self.metrics_retention_hours = metrics_retention_hours
        self.health_check_interval = health_check_interval
        self.enable_system_metrics = enable_system_metrics
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.active_operations: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Health checks
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_check: Dict[str, HealthCheck] = {}
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # System metrics
        self._last_system_metrics = {}
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage_percent': 80.0,
            'memory_usage_percent': 85.0,
            'disk_usage_percent': 90.0,
            'response_time_ms': 5000.0,
            'error_rate_percent': 5.0
        }
        
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_system_metrics())
        
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._periodic_health_checks())
        
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
    
    async def _monitor_system_metrics(self):
        """Monitor system metrics in background"""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                if self.enable_system_metrics:
                    await self._collect_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system metrics monitoring: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            # Collect metrics
            metrics = [
                ('system.cpu.usage_percent', cpu_percent, '%'),
                ('system.cpu.count', cpu_count, 'cores'),
                ('system.memory.usage_percent', memory.percent, '%'),
                ('system.memory.available_mb', memory.available / 1024 / 1024, 'MB'),
                ('system.memory.used_mb', memory.used / 1024 / 1024, 'MB'),
                ('system.memory.total_mb', memory.total / 1024 / 1024, 'MB'),
                ('system.swap.usage_percent', swap.percent, '%'),
                ('system.disk.usage_percent', disk.percent, '%'),
                ('system.disk.free_gb', disk.free / 1024 / 1024 / 1024, 'GB'),
                ('system.network.bytes_sent', network.bytes_sent, 'bytes'),
                ('system.network.bytes_recv', network.bytes_recv, 'bytes'),
                ('process.memory.usage_mb', process_memory.rss / 1024 / 1024, 'MB'),
                ('process.cpu.usage_percent', process_cpu, '%'),
            ]
            
            for name, value, unit in metrics:
                await self.record_metric(name, value, unit=unit)
            
            self._last_system_metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _periodic_health_checks(self):
        """Run periodic health checks"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._run_all_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health checks: {e}")
    
    async def _run_all_health_checks(self):
        """Run all registered health checks"""
        for name, check_func in self.health_checks.items():
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                self.last_health_check[name] = result
            except Exception as e:
                self.last_health_check[name] = HealthCheck(
                    name=name,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now()
                )
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory leaks"""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
                
                for metric_name, metric_deque in self.metrics.items():
                    # Remove old metrics
                    while metric_deque and metric_deque[0].timestamp < cutoff_time:
                        metric_deque.popleft()
                
                # Clean up operation times (keep last 1000)
                for op_name, times in self.operation_times.items():
                    if len(times) > 1000:
                        self.operation_times[op_name] = times[-1000:]
                
                # Force garbage collection
                gc.collect()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
    
    async def record_metric(self, 
                          name: str, 
                          value: float, 
                          tags: Optional[Dict[str, str]] = None,
                          unit: str = ""):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        self.metrics[name].append(metric)
        
        # Store metadata
        if name not in self.metric_metadata:
            self.metric_metadata[name] = {
                'unit': unit,
                'first_seen': datetime.now(),
                'count': 0
            }
        
        self.metric_metadata[name]['count'] += 1
        self.metric_metadata[name]['last_seen'] = datetime.now()
    
    def start_operation_timer(self, operation_name: str) -> str:
        """Start timing an operation and return timer ID"""
        timer_id = f"{operation_name}_{int(time.time() * 1000)}"
        self.active_operations[timer_id] = time.time()
        return timer_id
    
    def end_operation_timer(self, timer_id: str, success: bool = True):
        """End timing an operation"""
        if timer_id in self.active_operations:
            start_time = self.active_operations.pop(timer_id)
            duration = time.time() - start_time
            
            # Extract operation name from timer ID
            operation_name = timer_id.rsplit('_', 1)[0]
            
            # Record operation time
            self.operation_times[operation_name].append(duration)
            
            # Record metric
            asyncio.create_task(self.record_metric(
                f"operation.{operation_name}.duration_ms",
                duration * 1000,
                unit="ms"
            ))
            
            # Record success/failure
            status = "success" if success else "failure"
            asyncio.create_task(self.record_metric(
                f"operation.{operation_name}.{status}",
                1,
                unit="count"
            ))
            
            if not success:
                self.error_counts[operation_name] += 1
    
    def record_error(self, operation_name: str, error: Exception):
        """Record an error for an operation"""
        self.error_counts[operation_name] += 1
        
        asyncio.create_task(self.record_metric(
            f"error.{operation_name}.count",
            1,
            tags={'error_type': type(error).__name__},
            unit="count"
        ))
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
    
    async def get_metrics_summary(self, 
                                metric_names: Optional[List[str]] = None,
                                time_range_hours: int = 1) -> Dict[str, Any]:
        """Get summary of metrics"""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        summary = {}
        
        metrics_to_check = metric_names or list(self.metrics.keys())
        
        for metric_name in metrics_to_check:
            if metric_name not in self.metrics:
                continue
            
            recent_metrics = [
                m for m in self.metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                continue
            
            values = [m.value for m in recent_metrics]
            metadata = self.metric_metadata.get(metric_name, {})
            
            summary[metric_name] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1] if values else None,
                'unit': metadata.get('unit', ''),
                'time_range_hours': time_range_hours
            }
        
        return summary
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation performance statistics"""
        stats = {}
        
        for operation_name, times in self.operation_times.items():
            if not times:
                continue
            
            recent_times = times[-100:]  # Last 100 operations
            
            stats[operation_name] = {
                'total_operations': len(times),
                'recent_operations': len(recent_times),
                'avg_duration_ms': sum(recent_times) * 1000 / len(recent_times),
                'min_duration_ms': min(recent_times) * 1000,
                'max_duration_ms': max(recent_times) * 1000,
                'error_count': self.error_counts.get(operation_name, 0),
                'error_rate_percent': (
                    self.error_counts.get(operation_name, 0) / len(times) * 100
                    if times else 0
                )
            }
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'system_metrics': self._last_system_metrics,
            'thresholds': self.thresholds
        }
        
        # Check system metrics against thresholds
        if self._last_system_metrics:
            metrics = self._last_system_metrics
            
            if metrics.get('cpu_percent', 0) > self.thresholds['cpu_usage_percent']:
                health_status['overall_status'] = 'warning'
            
            if metrics.get('memory_percent', 0) > self.thresholds['memory_usage_percent']:
                health_status['overall_status'] = 'warning'
            
            if metrics.get('disk_percent', 0) > self.thresholds['disk_usage_percent']:
                health_status['overall_status'] = 'critical'
        
        # Check individual health checks
        for name, check in self.last_health_check.items():
            health_status['checks'][name] = {
                'status': check.status,
                'message': check.message,
                'timestamp': check.timestamp.isoformat(),
                'details': check.details
            }
            
            if check.status == 'critical':
                health_status['overall_status'] = 'critical'
            elif check.status == 'warning' and health_status['overall_status'] == 'healthy':
                health_status['overall_status'] = 'warning'
        
        return health_status
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        # Check system metrics
        if self._last_system_metrics:
            metrics = self._last_system_metrics
            
            if metrics.get('cpu_percent', 0) > 80:
                recommendations.append("High CPU usage detected. Consider scaling horizontally or optimizing CPU-intensive operations.")
            
            if metrics.get('memory_percent', 0) > 85:
                recommendations.append("High memory usage detected. Consider implementing memory optimization or increasing available memory.")
            
            if metrics.get('disk_percent', 0) > 90:
                recommendations.append("High disk usage detected. Consider cleaning up temporary files or increasing disk space.")
        
        # Check operation performance
        for operation_name, times in self.operation_times.items():
            if not times:
                continue
            
            recent_times = times[-50:]  # Last 50 operations
            avg_time = sum(recent_times) / len(recent_times)
            
            if avg_time > 5.0:  # 5 seconds
                recommendations.append(f"Operation '{operation_name}' is slow (avg: {avg_time:.2f}s). Consider optimization.")
            
            error_rate = self.error_counts.get(operation_name, 0) / len(times) * 100
            if error_rate > 5:
                recommendations.append(f"Operation '{operation_name}' has high error rate ({error_rate:.1f}%). Check error handling.")
        
        return recommendations
    
    async def close(self):
        """Close performance monitor and cleanup resources"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None

async def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

async def close_performance_monitor():
    """Close global performance monitor"""
    global _performance_monitor
    if _performance_monitor:
        await _performance_monitor.close()
        _performance_monitor = None

















