"""
Monitoring Module

Comprehensive monitoring with:
- Performance metrics collection
- Health checks and system monitoring
- Real-time metrics and alerting
- Resource usage tracking
- API performance monitoring
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import time
import asyncio
import psutil
import structlog
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager

logger = structlog.get_logger("monitoring")

# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for monitoring systems."""
    # Performance monitoring
    enable_performance_monitoring: bool = True
    metrics_retention_seconds: int = 3600  # 1 hour
    metrics_collection_interval: float = 1.0  # 1 second
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval: float = 30.0  # 30 seconds
    health_check_timeout: float = 10.0  # 10 seconds
    
    # Resource monitoring
    enable_resource_monitoring: bool = True
    resource_check_interval: float = 5.0  # 5 seconds
    
    # Alerting thresholds
    cpu_threshold: float = 80.0  # 80%
    memory_threshold: float = 85.0  # 85%
    disk_threshold: float = 90.0  # 90%
    response_time_threshold: float = 5.0  # 5 seconds
    
    # GPU monitoring
    enable_gpu_monitoring: bool = True
    gpu_memory_threshold: float = 90.0  # 90%

# =============================================================================
# METRICS MODELS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float = field(default_factory=time.time)
    request_count: int = 0
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    throughput: float = 0.0  # requests per second

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available: int = 0
    disk_percent: float = 0.0
    disk_free: int = 0
    network_io: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0

@dataclass
class GPUMetrics:
    """GPU metrics data structure."""
    timestamp: float = field(default_factory=time.time)
    gpu_count: int = 0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[int] = field(default_factory=list)
    gpu_memory_total: List[int] = field(default_factory=list)
    gpu_memory_percent: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)

@dataclass
class HealthStatus:
    """Health status data structure."""
    is_healthy: bool = True
    status: str = "healthy"
    issues: List[str] = field(default_factory=list)
    system_metrics: SystemMetrics = field(default_factory=SystemMetrics)
    gpu_metrics: GPUMetrics = field(default_factory=GPUMetrics)
    timestamp: float = field(default_factory=time.time)

# =============================================================================
# PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.metrics_history: deque = deque(maxlen=3600)  # 1 hour of data
        self.request_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.endpoint_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'request_count': 0,
            'total_time': 0.0,
            'error_count': 0,
            'last_request': 0.0
        })
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start performance monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop(self) -> None:
        """Stop performance monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float
    ) -> None:
        """Record a request for performance monitoring."""
        if not self._running:
            return
        
        # Record request time
        self.request_times.append(duration)
        
        # Update endpoint metrics
        endpoint_key = f"{method} {path}"
        endpoint_metrics = self.endpoint_metrics[endpoint_key]
        endpoint_metrics['request_count'] += 1
        endpoint_metrics['total_time'] += duration
        endpoint_metrics['last_request'] = time.time()
        
        # Record errors
        if status_code >= 400:
            self.error_counts[f"{status_code}"] += 1
            endpoint_metrics['error_count'] += 1
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval)
                await self._collect_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Performance monitoring error", error=str(e))
    
    async def _collect_metrics(self) -> None:
        """Collect and store performance metrics."""
        if not self.request_times:
            return
        
        # Calculate response time percentiles
        sorted_times = sorted(self.request_times)
        n = len(sorted_times)
        
        avg_time = sum(sorted_times) / n
        p95_time = sorted_times[int(n * 0.95)] if n > 0 else 0.0
        p99_time = sorted_times[int(n * 0.99)] if n > 0 else 0.0
        
        # Calculate error rate
        total_requests = sum(metrics['request_count'] for metrics in self.endpoint_metrics.values())
        total_errors = sum(metrics['error_count'] for metrics in self.endpoint_metrics.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0
        
        # Calculate throughput (requests per second)
        current_time = time.time()
        recent_requests = sum(
            1 for metrics in self.endpoint_metrics.values()
            if current_time - metrics['last_request'] < 60  # Last minute
        )
        throughput = recent_requests / 60.0
        
        # Create metrics record
        metrics = PerformanceMetrics(
            request_count=total_requests,
            response_time_avg=avg_time,
            response_time_p95=p95_time,
            response_time_p99=p99_time,
            error_count=total_errors,
            error_rate=error_rate,
            throughput=throughput
        )
        
        self.metrics_history.append(metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            'performance': {
                'request_count': latest_metrics.request_count,
                'response_time_avg': round(latest_metrics.response_time_avg, 3),
                'response_time_p95': round(latest_metrics.response_time_p95, 3),
                'response_time_p99': round(latest_metrics.response_time_p99, 3),
                'error_count': latest_metrics.error_count,
                'error_rate': round(latest_metrics.error_rate, 2),
                'throughput': round(latest_metrics.throughput, 2)
            },
            'endpoints': dict(self.endpoint_metrics),
            'error_breakdown': dict(self.error_counts),
            'timestamp': latest_metrics.timestamp
        }
    
    def get_metrics_history(self, duration_seconds: int = 300) -> List[Dict[str, Any]]:
        """Get performance metrics history."""
        cutoff_time = time.time() - duration_seconds
        
        return [
            {
                'timestamp': metrics.timestamp,
                'request_count': metrics.request_count,
                'response_time_avg': round(metrics.response_time_avg, 3),
                'response_time_p95': round(metrics.response_time_p95, 3),
                'response_time_p99': round(metrics.response_time_p99, 3),
                'error_count': metrics.error_count,
                'error_rate': round(metrics.error_rate, 2),
                'throughput': round(metrics.throughput, 2)
            }
            for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]

# =============================================================================
# HEALTH CHECKER
# =============================================================================

class HealthChecker:
    """System health monitoring and checking."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._last_health_status: Optional[HealthStatus] = None
    
    async def initialize(self) -> None:
        """Initialize health checker."""
        if self.config.enable_health_checks:
            self._running = True
            self._health_task = asyncio.create_task(self._health_check_loop())
            logger.info("Health checker initialized")
    
    async def close(self) -> None:
        """Close health checker."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker closed")
    
    async def check_system_health(self) -> HealthStatus:
        """Check current system health."""
        try:
            # Get system metrics
            system_metrics = await self._get_system_metrics()
            
            # Get GPU metrics
            gpu_metrics = await self._get_gpu_metrics()
            
            # Check for issues
            issues = []
            is_healthy = True
            
            # CPU check
            if system_metrics.cpu_percent > self.config.cpu_threshold:
                issues.append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
                is_healthy = False
            
            # Memory check
            if system_metrics.memory_percent > self.config.memory_threshold:
                issues.append(f"High memory usage: {system_metrics.memory_percent:.1f}%")
                is_healthy = False
            
            # Disk check
            if system_metrics.disk_percent > self.config.disk_threshold:
                issues.append(f"High disk usage: {system_metrics.disk_percent:.1f}%")
                is_healthy = False
            
            # GPU check
            for i, gpu_memory_percent in enumerate(gpu_metrics.gpu_memory_percent):
                if gpu_memory_percent > self.config.gpu_memory_threshold:
                    issues.append(f"High GPU {i} memory usage: {gpu_memory_percent:.1f}%")
                    is_healthy = False
            
            # Determine status
            if is_healthy:
                status = "healthy"
            elif len(issues) == 1:
                status = "degraded"
            else:
                status = "unhealthy"
            
            health_status = HealthStatus(
                is_healthy=is_healthy,
                status=status,
                issues=issues,
                system_metrics=system_metrics,
                gpu_metrics=gpu_metrics
            )
            
            self._last_health_status = health_status
            return health_status
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return HealthStatus(
                is_healthy=False,
                status="unhealthy",
                issues=[f"Health check error: {str(e)}"]
            )
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.check_system_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Health check loop error", error=str(e))
    
    async def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free = disk.free
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                disk_percent=disk_percent,
                disk_free=disk_free,
                network_io=network_io,
                process_count=process_count
            )
            
        except Exception as e:
            logger.warning("Failed to get system metrics", error=str(e))
            return SystemMetrics()
    
    async def _get_gpu_metrics(self) -> GPUMetrics:
        """Get current GPU metrics."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return GPUMetrics()
            
            gpu_count = torch.cuda.device_count()
            gpu_utilization = []
            gpu_memory_used = []
            gpu_memory_total = []
            gpu_memory_percent = []
            gpu_temperature = []
            
            for i in range(gpu_count):
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                
                gpu_memory_used.append(memory_allocated)
                gpu_memory_total.append(memory_total)
                gpu_memory_percent.append((memory_allocated / memory_total) * 100)
                
                # Utilization (approximate)
                gpu_utilization.append(0.0)  # Would need nvidia-ml-py for accurate utilization
                
                # Temperature (would need nvidia-ml-py)
                gpu_temperature.append(0.0)
            
            return GPUMetrics(
                gpu_count=gpu_count,
                gpu_utilization=gpu_utilization,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_memory_percent=gpu_memory_percent,
                gpu_temperature=gpu_temperature
            )
            
        except ImportError:
            logger.debug("PyTorch not available, GPU monitoring disabled")
            return GPUMetrics()
        except Exception as e:
            logger.warning("Failed to get GPU metrics", error=str(e))
            return GPUMetrics()
    
    def get_last_health_status(self) -> Optional[HealthStatus]:
        """Get the last health status."""
        return self._last_health_status

# =============================================================================
# MONITORING CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def monitoring_context(operation_name: str):
    """Context manager for monitoring operations."""
    start_time = time.time()
    
    try:
        yield
    except Exception as e:
        logger.error(f"Operation {operation_name} failed", error=str(e))
        raise
    finally:
        duration = time.time() - start_time
        logger.info(f"Operation {operation_name} completed", duration=duration)

# =============================================================================
# MONITORING DECORATORS
# =============================================================================

def monitor_performance(operation_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Operation {name} completed", duration=duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Operation {name} failed", duration=duration, error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Operation {name} completed", duration=duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Operation {name} failed", duration=duration, error=str(e))
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MonitoringConfig',
    'PerformanceMetrics',
    'SystemMetrics',
    'GPUMetrics',
    'HealthStatus',
    'PerformanceMonitor',
    'HealthChecker',
    'monitoring_context',
    'monitor_performance'
]






























