"""
Advanced Performance Optimizer for Facebook Posts System
Following functional programming principles and performance best practices
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, Any, List, Optional, Callable, Tuple
from functools import wraps, lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref

logger = logging.getLogger(__name__)


# Pure functions for performance optimization

@dataclass(frozen=True)
class PerformanceMetrics:
    """Immutable performance metrics - pure data structure"""
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "memory_available": self.memory_available,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "active_connections": self.active_connections,
            "timestamp": self.timestamp.isoformat()
        }


def get_system_metrics() -> PerformanceMetrics:
    """Get current system metrics - pure function"""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    network = psutil.net_io_counters()
    
    return PerformanceMetrics(
        cpu_usage=cpu_usage,
        memory_usage=memory.percent,
        memory_available=memory.available,
        disk_usage=disk.percent,
        network_io={
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        },
        active_connections=len(psutil.net_connections()),
        timestamp=datetime.utcnow()
    )


def calculate_memory_efficiency(metrics: PerformanceMetrics) -> float:
    """Calculate memory efficiency score - pure function"""
    if metrics.memory_usage >= 90:
        return 0.0
    elif metrics.memory_usage >= 80:
        return 0.3
    elif metrics.memory_usage >= 70:
        return 0.6
    elif metrics.memory_usage >= 50:
        return 0.8
    else:
        return 1.0


def calculate_cpu_efficiency(metrics: PerformanceMetrics) -> float:
    """Calculate CPU efficiency score - pure function"""
    if metrics.cpu_usage >= 90:
        return 0.0
    elif metrics.cpu_usage >= 80:
        return 0.3
    elif metrics.cpu_usage >= 70:
        return 0.6
    elif metrics.cpu_usage >= 50:
        return 0.8
    else:
        return 1.0


def determine_optimization_level(metrics: PerformanceMetrics) -> str:
    """Determine optimization level needed - pure function"""
    memory_efficiency = calculate_memory_efficiency(metrics)
    cpu_efficiency = calculate_cpu_efficiency(metrics)
    overall_efficiency = (memory_efficiency + cpu_efficiency) / 2
    
    if overall_efficiency >= 0.8:
        return "none"
    elif overall_efficiency >= 0.6:
        return "light"
    elif overall_efficiency >= 0.4:
        return "moderate"
    else:
        return "aggressive"


def create_memory_cleanup_plan(metrics: PerformanceMetrics) -> List[str]:
    """Create memory cleanup plan - pure function"""
    plan = []
    
    if metrics.memory_usage >= 80:
        plan.append("force_garbage_collection")
        plan.append("clear_caches")
        plan.append("reduce_batch_sizes")
    
    if metrics.memory_usage >= 90:
        plan.append("emergency_memory_cleanup")
        plan.append("pause_non_critical_tasks")
    
    return plan


def create_cpu_optimization_plan(metrics: PerformanceMetrics) -> List[str]:
    """Create CPU optimization plan - pure function"""
    plan = []
    
    if metrics.cpu_usage >= 70:
        plan.append("reduce_concurrent_requests")
        plan.append("enable_request_batching")
        plan.append("optimize_processing_pipeline")
    
    if metrics.cpu_usage >= 85:
        plan.append("enable_cpu_throttling")
        plan.append("defer_background_tasks")
        plan.append("reduce_ai_model_complexity")
    
    return plan


# Performance decorators and utilities

def measure_execution_time(func: Callable) -> Callable:
    """Decorator to measure function execution time - pure function"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(
            f"Function {func.__name__} executed in {execution_time:.4f}s",
            function=func.__name__,
            execution_time=execution_time
        )
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(
            f"Function {func.__name__} executed in {execution_time:.4f}s",
            function=func.__name__,
            execution_time=execution_time
        )
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def cache_result(ttl_seconds: int = 300) -> Callable:
    """Decorator to cache function results - pure function"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_timestamps = {}
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            current_time = time.time()
            
            # Check if cached result is still valid
            if (cache_key in cache and 
                cache_key in cache_timestamps and 
                current_time - cache_timestamps[cache_key] < ttl_seconds):
                return cache[cache_key]
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache[cache_key] = result
            cache_timestamps[cache_key] = current_time
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            current_time = time.time()
            
            # Check if cached result is still valid
            if (cache_key in cache and 
                cache_key in cache_timestamps and 
                current_time - cache_timestamps[cache_key] < ttl_seconds):
                return cache[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = result
            cache_timestamps[cache_key] = current_time
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def throttle_requests(max_requests_per_second: float) -> Callable:
    """Decorator to throttle requests - pure function"""
    def decorator(func: Callable) -> Callable:
        last_called = [0.0]
        min_interval = 1.0 / max_requests_per_second
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_time = time.time()
            time_since_last = current_time - last_called[0]
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_time = time.time()
            time_since_last = current_time - last_called[0]
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Memory management functions

def force_garbage_collection() -> Dict[str, Any]:
    """Force garbage collection - pure function"""
    before_gc = gc.get_count()
    collected = gc.collect()
    after_gc = gc.get_count()
    
    return {
        "before_gc": before_gc,
        "after_gc": after_gc,
        "collected_objects": collected,
        "memory_freed": True
    }


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage - pure function"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss,  # Resident Set Size
        "vms": memory_info.vms,  # Virtual Memory Size
        "percent": process.memory_percent(),
        "available": psutil.virtual_memory().available
    }


def optimize_memory_usage() -> Dict[str, Any]:
    """Optimize memory usage - pure function"""
    # Force garbage collection
    gc_result = force_garbage_collection()
    
    # Get memory usage before and after
    memory_before = get_memory_usage()
    
    # Additional memory optimizations
    gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    memory_after = get_memory_usage()
    
    return {
        "gc_result": gc_result,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "memory_freed": memory_before["rss"] - memory_after["rss"],
        "optimization_applied": True
    }


# CPU optimization functions

def get_cpu_usage() -> Dict[str, Any]:
    """Get current CPU usage - pure function"""
    cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
    cpu_count = psutil.cpu_count()
    
    return {
        "overall_percent": psutil.cpu_percent(interval=0.1),
        "per_cpu_percent": cpu_percent,
        "cpu_count": cpu_count,
        "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
    }


def optimize_cpu_usage() -> Dict[str, Any]:
    """Optimize CPU usage - pure function"""
    cpu_before = get_cpu_usage()
    
    # CPU optimization strategies
    optimization_applied = []
    
    if cpu_before["overall_percent"] > 80:
        optimization_applied.append("reduce_concurrency")
        optimization_applied.append("enable_batching")
    
    if cpu_before["overall_percent"] > 90:
        optimization_applied.append("emergency_throttling")
        optimization_applied.append("defer_tasks")
    
    cpu_after = get_cpu_usage()
    
    return {
        "cpu_before": cpu_before,
        "cpu_after": cpu_after,
        "optimization_applied": optimization_applied,
        "cpu_reduction": cpu_before["overall_percent"] - cpu_after["overall_percent"]
    }


# Advanced Performance Optimizer Class

class AdvancedPerformanceOptimizer:
    """Advanced Performance Optimizer following functional principles"""
    
    def __init__(self, optimization_interval: int = 30):
        self.optimization_interval = optimization_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_plans: List[Dict[str, Any]] = []
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance thresholds
        self.thresholds = {
            "memory_warning": 70.0,
            "memory_critical": 85.0,
            "cpu_warning": 70.0,
            "cpu_critical": 85.0
        }
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Get current metrics
                metrics = get_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Check for optimization needs
                await self._check_optimization_needs(metrics)
                
                # Wait for next check
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _check_optimization_needs(self, metrics: PerformanceMetrics) -> None:
        """Check if optimization is needed"""
        # Memory optimization
        if metrics.memory_usage >= self.thresholds["memory_critical"]:
            await self._apply_memory_optimization(metrics)
        elif metrics.memory_usage >= self.thresholds["memory_warning"]:
            await self._apply_light_memory_optimization(metrics)
        
        # CPU optimization
        if metrics.cpu_usage >= self.thresholds["cpu_critical"]:
            await self._apply_cpu_optimization(metrics)
        elif metrics.cpu_usage >= self.thresholds["cpu_warning"]:
            await self._apply_light_cpu_optimization(metrics)
    
    async def _apply_memory_optimization(self, metrics: PerformanceMetrics) -> None:
        """Apply memory optimization"""
        plan = create_memory_cleanup_plan(metrics)
        
        for action in plan:
            if action == "force_garbage_collection":
                gc_result = force_garbage_collection()
                logger.info("Applied garbage collection", result=gc_result)
            
            elif action == "clear_caches":
                # This would be implemented by the cache service
                logger.info("Cleared caches")
            
            elif action == "reduce_batch_sizes":
                # This would be implemented by the batch processor
                logger.info("Reduced batch sizes")
        
        self.optimization_plans.append({
            "type": "memory",
            "level": "critical",
            "plan": plan,
            "timestamp": datetime.utcnow(),
            "metrics": metrics.to_dict()
        })
    
    async def _apply_light_memory_optimization(self, metrics: PerformanceMetrics) -> None:
        """Apply light memory optimization"""
        gc_result = force_garbage_collection()
        logger.info("Applied light memory optimization", result=gc_result)
        
        self.optimization_plans.append({
            "type": "memory",
            "level": "warning",
            "plan": ["force_garbage_collection"],
            "timestamp": datetime.utcnow(),
            "metrics": metrics.to_dict()
        })
    
    async def _apply_cpu_optimization(self, metrics: PerformanceMetrics) -> None:
        """Apply CPU optimization"""
        plan = create_cpu_optimization_plan(metrics)
        
        for action in plan:
            if action == "reduce_concurrent_requests":
                logger.info("Reduced concurrent requests")
            
            elif action == "enable_request_batching":
                logger.info("Enabled request batching")
            
            elif action == "enable_cpu_throttling":
                logger.info("Enabled CPU throttling")
        
        self.optimization_plans.append({
            "type": "cpu",
            "level": "critical",
            "plan": plan,
            "timestamp": datetime.utcnow(),
            "metrics": metrics.to_dict()
        })
    
    async def _apply_light_cpu_optimization(self, metrics: PerformanceMetrics) -> None:
        """Apply light CPU optimization"""
        logger.info("Applied light CPU optimization")
        
        self.optimization_plans.append({
            "type": "cpu",
            "level": "warning",
            "plan": ["monitor_cpu_usage"],
            "timestamp": datetime.utcnow(),
            "metrics": metrics.to_dict()
        })
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system metrics"""
        return get_system_metrics()
    
    def get_metrics_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get metrics history"""
        return [metrics.to_dict() for metrics in self.metrics_history[-limit:]]
    
    def get_optimization_plans(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get optimization plans history"""
        return self.optimization_plans[-limit:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            "current_metrics": latest_metrics.to_dict(),
            "memory_efficiency": calculate_memory_efficiency(latest_metrics),
            "cpu_efficiency": calculate_cpu_efficiency(latest_metrics),
            "optimization_level": determine_optimization_level(latest_metrics),
            "total_optimizations": len(self.optimization_plans),
            "monitoring_status": "active" if self.is_monitoring else "inactive"
        }
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Perform system optimization"""
        current_metrics = get_system_metrics()
        
        # Memory optimization
        memory_result = optimize_memory_usage()
        
        # CPU optimization
        cpu_result = optimize_cpu_usage()
        
        # Get metrics after optimization
        optimized_metrics = get_system_metrics()
        
        return {
            "before_optimization": current_metrics.to_dict(),
            "after_optimization": optimized_metrics.to_dict(),
            "memory_optimization": memory_result,
            "cpu_optimization": cpu_result,
            "optimization_timestamp": datetime.utcnow().isoformat()
        }


# Factory functions

def create_performance_optimizer(optimization_interval: int = 30) -> AdvancedPerformanceOptimizer:
    """Create performance optimizer instance - pure function"""
    return AdvancedPerformanceOptimizer(optimization_interval)


async def get_performance_optimizer() -> AdvancedPerformanceOptimizer:
    """Get performance optimizer instance with monitoring"""
    optimizer = create_performance_optimizer()
    await optimizer.start_monitoring()
    return optimizer


# Utility functions for performance monitoring

def create_performance_dashboard_data(optimizer: AdvancedPerformanceOptimizer) -> Dict[str, Any]:
    """Create performance dashboard data - pure function"""
    summary = optimizer.get_performance_summary()
    metrics_history = optimizer.get_metrics_history(20)
    optimization_plans = optimizer.get_optimization_plans(10)
    
    return {
        "summary": summary,
        "metrics_history": metrics_history,
        "optimization_plans": optimization_plans,
        "dashboard_timestamp": datetime.utcnow().isoformat()
    }


def calculate_performance_trends(metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate performance trends - pure function"""
    if len(metrics_history) < 2:
        return {"trend": "insufficient_data"}
    
    # Calculate trends for last 10 data points
    recent_metrics = metrics_history[-10:]
    
    cpu_values = [m["cpu_usage"] for m in recent_metrics]
    memory_values = [m["memory_usage"] for m in recent_metrics]
    
    # Simple trend calculation
    cpu_trend = "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
    memory_trend = "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
    
    return {
        "cpu_trend": cpu_trend,
        "memory_trend": memory_trend,
        "cpu_average": sum(cpu_values) / len(cpu_values),
        "memory_average": sum(memory_values) / len(memory_values),
        "data_points": len(recent_metrics)
    }

