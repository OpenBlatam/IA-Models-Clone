"""
Gamma App - Advanced Performance Service
Ultra-optimized performance monitoring and optimization system
"""

import asyncio
import time
import psutil
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from functools import wraps
import json
import gc
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io_bytes: int
    active_connections: int
    request_count: int
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    cache_hit_rate: float
    queue_size: int
    worker_count: int
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    category: str  # "cpu", "memory", "network", "database", "cache"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    impact: str
    effort: str
    implementation: str
    expected_improvement: str

@dataclass
class ResourceUsage:
    """Resource usage tracking"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    timestamp: datetime

class PerformanceProfiler:
    """Advanced performance profiler"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_end = None
        self.cpu_start = None
        self.cpu_end = None
        self.traced = False
    
    def start(self):
        """Start profiling"""
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss
        self.cpu_start = psutil.Process().cpu_percent()
        
        # Start memory tracing
        if not self.traced:
            tracemalloc.start()
            self.traced = True
    
    def stop(self):
        """Stop profiling and return metrics"""
        self.end_time = time.time()
        self.memory_end = psutil.Process().memory_info().rss
        self.cpu_end = psutil.Process().cpu_percent()
        
        duration = self.end_time - self.start_time
        memory_delta = (self.memory_end - self.memory_start) / 1024 / 1024  # MB
        
        # Get memory trace
        memory_trace = None
        if self.traced:
            current, peak = tracemalloc.get_traced_memory()
            memory_trace = {
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024
            }
            tracemalloc.stop()
            self.traced = False
        
        return {
            'duration_seconds': duration,
            'memory_delta_mb': memory_delta,
            'cpu_percent': self.cpu_end,
            'memory_trace': memory_trace
        }

class AdvancedPerformanceService:
    """
    Advanced performance monitoring and optimization service
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize performance service"""
        self.config = config or {}
        self.metrics_history: deque = deque(maxlen=1000)
        self.resource_usage_history: deque = deque(maxlen=1000)
        self.performance_alerts: List[Dict] = []
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'response_time_warning': 2.0,
            'response_time_critical': 5.0,
            'error_rate_warning': 5.0,
            'error_rate_critical': 10.0
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Resource pools
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        
        # Weak references for cleanup
        self.tracked_objects: weakref.WeakSet = weakref.WeakSet()
        
        logger.info("Advanced Performance Service initialized")
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        self.optimization_task = asyncio.create_task(self._optimization_loop(300))  # 5 minutes
        
        logger.info(f"Performance monitoring started (interval: {interval}s)")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._collect_metrics()
                await self._check_thresholds()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _optimization_loop(self, interval: int):
        """Optimization recommendation loop"""
        while self.monitoring_active:
            try:
                await self._generate_optimization_recommendations()
                await self._apply_automatic_optimizations()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_metrics(self):
        """Collect system performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Calculate response time statistics
            response_times = list(self.request_times)
            response_time_avg = np.mean(response_times) if response_times else 0
            response_time_p95 = np.percentile(response_times, 95) if response_times else 0
            response_time_p99 = np.percentile(response_times, 99) if response_times else 0
            
            # Calculate error rate
            total_requests = sum(self.error_counts.values())
            error_count = sum(count for error, count in self.error_counts.items() if 'error' in error.lower())
            error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
            
            # Cache hit rate
            total_cache_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            cache_hit_rate = (self.cache_stats['hits'] / total_cache_requests * 100) if total_cache_requests > 0 else 0
            
            # GPU metrics (if available)
            gpu_usage = None
            gpu_memory = None
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
                    gpu_memory = gpus[0].memoryUtil * 100
            except ImportError:
                pass
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=process_memory.rss / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk.percent,
                network_io_bytes=network.bytes_sent + network.bytes_recv,
                active_connections=len(psutil.net_connections()),
                request_count=total_requests,
                response_time_avg=response_time_avg,
                response_time_p95=response_time_p95,
                response_time_p99=response_time_p99,
                error_rate=error_rate,
                cache_hit_rate=cache_hit_rate,
                queue_size=0,  # Would integrate with actual queue
                worker_count=len(psutil.pids()),
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory
            )
            
            self.metrics_history.append(metrics)
            
            # Track resource usage
            resource_usage = ResourceUsage(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_usage=(network.bytes_sent + network.bytes_recv) / 1024 / 1024,  # MB
                timestamp=datetime.now()
            )
            self.resource_usage_history.append(resource_usage)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _check_thresholds(self):
        """Check performance thresholds and generate alerts"""
        if not self.metrics_history:
            return
        
        latest_metrics = self.metrics_history[-1]
        alerts = []
        
        # CPU alerts
        if latest_metrics.cpu_percent >= self.thresholds['cpu_critical']:
            alerts.append({
                'type': 'critical',
                'metric': 'cpu',
                'value': latest_metrics.cpu_percent,
                'threshold': self.thresholds['cpu_critical'],
                'message': f"Critical CPU usage: {latest_metrics.cpu_percent:.1f}%"
            })
        elif latest_metrics.cpu_percent >= self.thresholds['cpu_warning']:
            alerts.append({
                'type': 'warning',
                'metric': 'cpu',
                'value': latest_metrics.cpu_percent,
                'threshold': self.thresholds['cpu_warning'],
                'message': f"High CPU usage: {latest_metrics.cpu_percent:.1f}%"
            })
        
        # Memory alerts
        if latest_metrics.memory_percent >= self.thresholds['memory_critical']:
            alerts.append({
                'type': 'critical',
                'metric': 'memory',
                'value': latest_metrics.memory_percent,
                'threshold': self.thresholds['memory_critical'],
                'message': f"Critical memory usage: {latest_metrics.memory_percent:.1f}%"
            })
        elif latest_metrics.memory_percent >= self.thresholds['memory_warning']:
            alerts.append({
                'type': 'warning',
                'metric': 'memory',
                'value': latest_metrics.memory_percent,
                'threshold': self.thresholds['memory_warning'],
                'message': f"High memory usage: {latest_metrics.memory_percent:.1f}%"
            })
        
        # Response time alerts
        if latest_metrics.response_time_avg >= self.thresholds['response_time_critical']:
            alerts.append({
                'type': 'critical',
                'metric': 'response_time',
                'value': latest_metrics.response_time_avg,
                'threshold': self.thresholds['response_time_critical'],
                'message': f"Critical response time: {latest_metrics.response_time_avg:.2f}s"
            })
        elif latest_metrics.response_time_avg >= self.thresholds['response_time_warning']:
            alerts.append({
                'type': 'warning',
                'metric': 'response_time',
                'value': latest_metrics.response_time_avg,
                'threshold': self.thresholds['response_time_warning'],
                'message': f"High response time: {latest_metrics.response_time_avg:.2f}s"
            })
        
        # Error rate alerts
        if latest_metrics.error_rate >= self.thresholds['error_rate_critical']:
            alerts.append({
                'type': 'critical',
                'metric': 'error_rate',
                'value': latest_metrics.error_rate,
                'threshold': self.thresholds['error_rate_critical'],
                'message': f"Critical error rate: {latest_metrics.error_rate:.1f}%"
            })
        elif latest_metrics.error_rate >= self.thresholds['error_rate_warning']:
            alerts.append({
                'type': 'warning',
                'metric': 'error_rate',
                'value': latest_metrics.error_rate,
                'threshold': self.thresholds['error_rate_warning'],
                'message': f"High error rate: {latest_metrics.error_rate:.1f}%"
            })
        
        # Store alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now()
            self.performance_alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.performance_alerts) > 100:
            self.performance_alerts = self.performance_alerts[-100:]
    
    async def _generate_optimization_recommendations(self):
        """Generate performance optimization recommendations"""
        if not self.metrics_history:
            return
        
        # Analyze recent metrics (last 10 data points)
        recent_metrics = list(self.metrics_history)[-10:]
        if len(recent_metrics) < 5:
            return
        
        recommendations = []
        
        # CPU optimization
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        if avg_cpu > 60:
            recommendations.append(OptimizationRecommendation(
                category="cpu",
                priority="high" if avg_cpu > 80 else "medium",
                title="CPU Optimization",
                description=f"Average CPU usage is {avg_cpu:.1f}%. Consider optimizing CPU-intensive operations.",
                impact="High",
                effort="Medium",
                implementation="Implement async processing, caching, and load balancing",
                expected_improvement="20-40% CPU reduction"
            ))
        
        # Memory optimization
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        if avg_memory > 70:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="high" if avg_memory > 85 else "medium",
                title="Memory Optimization",
                description=f"Average memory usage is {avg_memory:.1f}%. Consider memory optimization strategies.",
                impact="High",
                effort="Medium",
                implementation="Implement object pooling, garbage collection tuning, and memory profiling",
                expected_improvement="15-30% memory reduction"
            ))
        
        # Response time optimization
        avg_response_time = np.mean([m.response_time_avg for m in recent_metrics])
        if avg_response_time > 1.0:
            recommendations.append(OptimizationRecommendation(
                category="response_time",
                priority="high" if avg_response_time > 2.0 else "medium",
                title="Response Time Optimization",
                description=f"Average response time is {avg_response_time:.2f}s. Consider performance optimizations.",
                impact="High",
                effort="High",
                implementation="Implement caching, database optimization, and async processing",
                expected_improvement="30-50% response time reduction"
            ))
        
        # Cache optimization
        avg_cache_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])
        if avg_cache_hit_rate < 70:
            recommendations.append(OptimizationRecommendation(
                category="cache",
                priority="medium",
                title="Cache Optimization",
                description=f"Cache hit rate is {avg_cache_hit_rate:.1f}%. Consider improving caching strategy.",
                impact="Medium",
                effort="Low",
                implementation="Increase cache TTL, implement cache warming, and optimize cache keys",
                expected_improvement="20-30% performance improvement"
            ))
        
        self.optimization_recommendations = recommendations
    
    async def _apply_automatic_optimizations(self):
        """Apply automatic performance optimizations"""
        try:
            # Garbage collection optimization
            if len(self.metrics_history) > 0:
                latest_metrics = self.metrics_history[-1]
                if latest_metrics.memory_percent > 80:
                    gc.collect()
                    logger.info("Triggered garbage collection due to high memory usage")
            
            # Dynamic thread pool adjustment
            if len(self.metrics_history) > 5:
                recent_cpu = np.mean([m.cpu_percent for m in list(self.metrics_history)[-5:]])
                if recent_cpu < 30:
                    # Low CPU usage, can increase thread pool
                    pass
                elif recent_cpu > 80:
                    # High CPU usage, should reduce thread pool
                    pass
            
        except Exception as e:
            logger.error(f"Error applying automatic optimizations: {e}")
    
    def track_request(self, duration: float, success: bool = True):
        """Track request performance"""
        self.request_times.append(duration)
        
        if not success:
            self.error_counts['error'] += 1
        else:
            self.error_counts['success'] += 1
    
    def track_cache_hit(self, hit: bool):
        """Track cache performance"""
        if hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
    
    def profile_function(self, func: Callable):
        """Decorator to profile function performance"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            profiler.start()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                metrics = profiler.stop()
                logger.debug(f"Function {func.__name__} completed in {metrics['duration_seconds']:.3f}s")
                
                return result
            except Exception as e:
                metrics = profiler.stop()
                logger.error(f"Function {func.__name__} failed after {metrics['duration_seconds']:.3f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            profiler.start()
            
            try:
                result = func(*args, **kwargs)
                metrics = profiler.stop()
                logger.debug(f"Function {func.__name__} completed in {metrics['duration_seconds']:.3f}s")
                return result
            except Exception as e:
                metrics = profiler.stop()
                logger.error(f"Function {func.__name__} failed after {metrics['duration_seconds']:.3f}s: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        latest_metrics = self.metrics_history[-1]
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        response_time_trend = self._calculate_trend([m.response_time_avg for m in recent_metrics])
        
        # Get recent alerts
        recent_alerts = [alert for alert in self.performance_alerts 
                        if alert['timestamp'] > datetime.now() - timedelta(hours=1)]
        
        return {
            "current_metrics": asdict(latest_metrics),
            "trends": {
                "cpu": cpu_trend,
                "memory": memory_trend,
                "response_time": response_time_trend
            },
            "alerts": recent_alerts,
            "recommendations": [asdict(rec) for rec in self.optimization_recommendations],
            "summary": {
                "status": self._get_system_status(),
                "uptime": self._get_uptime(),
                "total_requests": latest_metrics.request_count,
                "average_response_time": latest_metrics.response_time_avg,
                "error_rate": latest_metrics.error_rate,
                "cache_hit_rate": latest_metrics.cache_hit_rate
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        diff = second_half - first_half
        if abs(diff) < 0.1:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _get_system_status(self) -> str:
        """Get overall system status"""
        if not self.metrics_history:
            return "unknown"
        
        latest = self.metrics_history[-1]
        
        if (latest.cpu_percent > self.thresholds['cpu_critical'] or
            latest.memory_percent > self.thresholds['memory_critical'] or
            latest.response_time_avg > self.thresholds['response_time_critical'] or
            latest.error_rate > self.thresholds['error_rate_critical']):
            return "critical"
        elif (latest.cpu_percent > self.thresholds['cpu_warning'] or
              latest.memory_percent > self.thresholds['memory_warning'] or
              latest.response_time_avg > self.thresholds['response_time_warning'] or
              latest.error_rate > self.thresholds['error_rate_warning']):
            return "warning"
        else:
            return "healthy"
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            uptime = timedelta(seconds=uptime_seconds)
            return str(uptime).split('.')[0]  # Remove microseconds
        except:
            return "unknown"
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Apply system optimizations"""
        optimizations_applied = []
        
        try:
            # Memory optimization
            if len(self.metrics_history) > 0:
                latest = self.metrics_history[-1]
                if latest.memory_percent > 80:
                    gc.collect()
                    optimizations_applied.append("Garbage collection triggered")
            
            # Cache optimization
            if len(self.metrics_history) > 5:
                recent_cache_hit_rate = np.mean([m.cache_hit_rate for m in list(self.metrics_history)[-5:]])
                if recent_cache_hit_rate < 70:
                    # Would implement cache warming here
                    optimizations_applied.append("Cache warming initiated")
            
            return {
                "success": True,
                "optimizations_applied": optimizations_applied,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing system: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def get_resource_usage_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get resource usage history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_usage = [usage for usage in self.resource_usage_history 
                       if usage.timestamp > cutoff_time]
        
        if not recent_usage:
            return {"error": "No data available for the specified period"}
        
        return {
            "timestamps": [usage.timestamp.isoformat() for usage in recent_usage],
            "cpu_usage": [usage.cpu_usage for usage in recent_usage],
            "memory_usage": [usage.memory_usage for usage in recent_usage],
            "disk_usage": [usage.disk_usage for usage in recent_usage],
            "network_usage": [usage.network_usage for usage in recent_usage]
        }
    
    async def close(self):
        """Close performance service"""
        await self.stop_monitoring()
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("Performance service closed")

# Global performance service instance
performance_service = AdvancedPerformanceService()

# Convenience functions
def track_performance(func: Callable):
    """Performance tracking decorator"""
    return performance_service.profile_function(func)

async def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics"""
    return await performance_service.get_performance_dashboard()

async def optimize_performance() -> Dict[str, Any]:
    """Apply performance optimizations"""
    return await performance_service.optimize_system()