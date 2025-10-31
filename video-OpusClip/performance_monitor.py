"""
Performance Monitor for Video-OpusClip

Comprehensive performance monitoring, benchmarking, and optimization tracking
for the video processing system.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger()

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    
    # Timing Metrics
    processing_time: float = 0.0
    response_time: float = 0.0
    throughput: float = 0.0  # requests per second
    
    # Resource Metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: Optional[float] = None
    disk_io: float = 0.0
    network_io: float = 0.0
    
    # Cache Metrics
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    cache_size: int = 0
    
    # Error Metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Custom Metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Benchmark test result."""
    
    test_name: str
    duration: float
    iterations: int
    avg_time: float
    min_time: float
    max_time: float
    throughput: float
    memory_peak: float
    cpu_peak: float
    success_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

# =============================================================================
# PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.benchmark_results = []
        self.active_monitors = {}
        self.monitoring_enabled = True
        
        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0,
            "memory_critical": 95.0,
            "response_time_warning": 5.0,
            "response_time_critical": 10.0,
            "error_rate_warning": 5.0,
            "error_rate_critical": 10.0
        }
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _start_background_monitoring(self):
        """Start background system monitoring."""
        def monitor_system():
            while self.monitoring_enabled:
                try:
                    metrics = self._collect_system_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Check thresholds
                    self._check_thresholds(metrics)
                    
                    time.sleep(1)  # Monitor every second
                except Exception as e:
                    logger.error("System monitoring error", error=str(e))
                    time.sleep(5)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        logger.info("Background performance monitoring started")
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_rate = (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024  # MB/s
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_rate = (network_io.bytes_sent + network_io.bytes_recv) / 1024 / 1024  # MB/s
            
            # GPU usage (if available)
            gpu_usage = self._get_gpu_usage()
            
            return PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                gpu_usage=gpu_usage,
                disk_io=disk_io_rate,
                network_io=network_io_rate,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            return PerformanceMetrics()
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        except Exception as e:
            logger.debug("GPU monitoring not available", error=str(e))
        return None
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and trigger alerts."""
        alerts = []
        
        # CPU threshold checks
        if metrics.cpu_usage > self.thresholds["cpu_critical"]:
            alerts.append(f"CRITICAL: CPU usage {metrics.cpu_usage:.1f}%")
        elif metrics.cpu_usage > self.thresholds["cpu_warning"]:
            alerts.append(f"WARNING: CPU usage {metrics.cpu_usage:.1f}%")
        
        # Memory threshold checks
        if metrics.memory_usage > self.thresholds["memory_critical"]:
            alerts.append(f"CRITICAL: Memory usage {metrics.memory_usage:.1f}%")
        elif metrics.memory_usage > self.thresholds["memory_warning"]:
            alerts.append(f"WARNING: Memory usage {metrics.memory_usage:.1f}%")
        
        # Log alerts
        for alert in alerts:
            logger.warning("Performance alert", alert=alert, metrics=metrics.__dict__)
    
    def start_monitoring(self, operation_name: str) -> str:
        """Start monitoring a specific operation."""
        monitor_id = f"{operation_name}_{int(time.time() * 1000)}"
        self.active_monitors[monitor_id] = {
            "operation": operation_name,
            "start_time": time.perf_counter(),
            "start_metrics": self._collect_system_metrics()
        }
        return monitor_id
    
    def stop_monitoring(self, monitor_id: str) -> Optional[PerformanceMetrics]:
        """Stop monitoring and return performance metrics."""
        if monitor_id not in self.active_monitors:
            return None
        
        monitor_data = self.active_monitors.pop(monitor_id)
        end_time = time.perf_counter()
        end_metrics = self._collect_system_metrics()
        
        # Calculate metrics
        duration = end_time - monitor_data["start_time"]
        start_metrics = monitor_data["start_metrics"]
        
        metrics = PerformanceMetrics(
            processing_time=duration,
            cpu_usage=end_metrics.cpu_usage - start_metrics.cpu_usage,
            memory_usage=end_metrics.memory_usage - start_metrics.memory_usage,
            gpu_usage=end_metrics.gpu_usage,
            disk_io=end_metrics.disk_io - start_metrics.disk_io,
            network_io=end_metrics.network_io - start_metrics.network_io
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system metrics."""
        return self._collect_system_metrics()
    
    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Get metrics history for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if hasattr(m, 'timestamp') and m.timestamp > cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
        
        return {
            "total_operations": len(self.metrics_history),
            "recent_operations": len(recent_metrics),
            "avg_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "avg_processing_time": sum(m.processing_time for m in recent_metrics) / len(recent_metrics),
            "peak_cpu_usage": max(m.cpu_usage for m in recent_metrics),
            "peak_memory_usage": max(m.memory_usage for m in recent_metrics),
            "current_metrics": self.get_current_metrics().__dict__
        }

# =============================================================================
# BENCHMARKING SYSTEM
# =============================================================================

class BenchmarkRunner:
    """Performance benchmarking system."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.results = []
    
    async def benchmark_async(
        self,
        test_name: str,
        test_func: Callable,
        iterations: int = 10,
        warmup_iterations: int = 3
    ) -> BenchmarkResult:
        """Run async benchmark test."""
        times = []
        memory_peaks = []
        cpu_peaks = []
        successes = 0
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                await test_func()
            except Exception:
                pass
        
        # Actual benchmark
        start_time = time.perf_counter()
        
        for i in range(iterations):
            iteration_start = time.perf_counter()
            
            try:
                # Start monitoring
                monitor_id = self.monitor.start_monitoring(f"{test_name}_iter_{i}")
                
                # Run test
                await test_func()
                
                # Stop monitoring
                metrics = self.monitor.stop_monitoring(monitor_id)
                
                iteration_time = time.perf_counter() - iteration_start
                times.append(iteration_time)
                memory_peaks.append(metrics.memory_usage if metrics else 0)
                cpu_peaks.append(metrics.cpu_usage if metrics else 0)
                successes += 1
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed", error=str(e))
        
        total_time = time.perf_counter() - start_time
        
        return BenchmarkResult(
            test_name=test_name,
            duration=total_time,
            iterations=iterations,
            avg_time=sum(times) / len(times) if times else 0,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            throughput=iterations / total_time if total_time > 0 else 0,
            memory_peak=max(memory_peaks) if memory_peaks else 0,
            cpu_peak=max(cpu_peaks) if cpu_peaks else 0,
            success_rate=successes / iterations if iterations > 0 else 0
        )
    
    def benchmark_sync(
        self,
        test_name: str,
        test_func: Callable,
        iterations: int = 10,
        warmup_iterations: int = 3
    ) -> BenchmarkResult:
        """Run sync benchmark test."""
        times = []
        memory_peaks = []
        cpu_peaks = []
        successes = 0
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                test_func()
            except Exception:
                pass
        
        # Actual benchmark
        start_time = time.perf_counter()
        
        for i in range(iterations):
            iteration_start = time.perf_counter()
            
            try:
                # Start monitoring
                monitor_id = self.monitor.start_monitoring(f"{test_name}_iter_{i}")
                
                # Run test
                test_func()
                
                # Stop monitoring
                metrics = self.monitor.stop_monitoring(monitor_id)
                
                iteration_time = time.perf_counter() - iteration_start
                times.append(iteration_time)
                memory_peaks.append(metrics.memory_usage if metrics else 0)
                cpu_peaks.append(metrics.cpu_usage if metrics else 0)
                successes += 1
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed", error=str(e))
        
        total_time = time.perf_counter() - start_time
        
        return BenchmarkResult(
            test_name=test_name,
            duration=total_time,
            iterations=iterations,
            avg_time=sum(times) / len(times) if times else 0,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            throughput=iterations / total_time if total_time > 0 else 0,
            memory_peak=max(memory_peaks) if memory_peaks else 0,
            cpu_peak=max(cpu_peaks) if cpu_peaks else 0,
            success_rate=successes / iterations if iterations > 0 else 0
        )
    
    def run_benchmark_suite(self, tests: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Run a suite of benchmark tests."""
        results = []
        
        for test in tests:
            try:
                if test.get("async", False):
                    result = await self.benchmark_async(
                        test["name"],
                        test["func"],
                        test.get("iterations", 10),
                        test.get("warmup_iterations", 3)
                    )
                else:
                    result = self.benchmark_sync(
                        test["name"],
                        test["func"],
                        test.get("iterations", 10),
                        test.get("warmup_iterations", 3)
                    )
                
                results.append(result)
                logger.info(f"Benchmark completed", test=test["name"], result=result.__dict__)
                
            except Exception as e:
                logger.error(f"Benchmark failed", test=test["name"], error=str(e))
        
        return results

# =============================================================================
# PERFORMANCE DECORATORS
# =============================================================================

def monitor_performance(operation_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        def sync_wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            monitor_id = monitor.start_monitoring(operation_name or func.__name__)
            
            try:
                result = func(*args, **kwargs)
                metrics = monitor.stop_monitoring(monitor_id)
                logger.info(f"Function {func.__name__} completed", metrics=metrics.__dict__)
                return result
            except Exception as e:
                monitor.stop_monitoring(monitor_id)
                raise
        
        async def async_wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            monitor_id = monitor.start_monitoring(operation_name or func.__name__)
            
            try:
                result = await func(*args, **kwargs)
                metrics = monitor.stop_monitoring(monitor_id)
                logger.info(f"Async function {func.__name__} completed", metrics=metrics.__dict__)
                return result
            except Exception as e:
                monitor.stop_monitoring(monitor_id)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# =============================================================================
# GLOBAL PERFORMANCE MONITOR
# =============================================================================

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor

def get_benchmark_runner() -> BenchmarkRunner:
    """Get a benchmark runner instance."""
    return BenchmarkRunner(performance_monitor) 