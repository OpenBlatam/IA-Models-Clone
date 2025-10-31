"""
ML NLP Benchmark Performance System
Real, working performance analysis and benchmarking for ML NLP Benchmark system
"""

import time
import psutil
import threading
import statistics
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class BenchmarkResult:
    """Benchmark result structure"""
    benchmark_name: str
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    input_size: int
    output_size: int
    success: bool
    error_message: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PerformanceProfile:
    """Performance profile structure"""
    profile_name: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: float
    median_execution_time: float
    min_execution_time: float
    max_execution_time: float
    std_execution_time: float
    p95_execution_time: float
    p99_execution_time: float
    average_memory_usage: float
    peak_memory_usage: float
    average_cpu_usage: float
    peak_cpu_usage: float
    throughput_per_second: float
    success_rate: float
    created_at: datetime
    metadata: Dict[str, Any]

class MLNLPBenchmarkPerformance:
    """Advanced performance analysis and benchmarking system"""
    
    def __init__(self):
        self.performance_metrics = []
        self.benchmark_results = []
        self.performance_profiles = {}
        self.lock = threading.RLock()
        
        # Performance monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 1.0  # seconds
        
        # System baseline
        self.system_baseline = self._capture_system_baseline()
        
        # Performance thresholds
        self.performance_thresholds = {
            "execution_time": {
                "warning": 1.0,  # seconds
                "critical": 5.0
            },
            "memory_usage": {
                "warning": 100 * 1024 * 1024,  # 100MB
                "critical": 500 * 1024 * 1024   # 500MB
            },
            "cpu_usage": {
                "warning": 80.0,  # percentage
                "critical": 95.0
            },
            "throughput": {
                "warning": 10,  # requests per second
                "critical": 1
            }
        }
    
    def _capture_system_baseline(self) -> Dict[str, Any]:
        """Capture system baseline metrics"""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_total": psutil.disk_usage('/').total,
                "disk_free": psutil.disk_usage('/').free,
                "timestamp": datetime.now()
            }
        except Exception as e:
            logger.error(f"Error capturing system baseline: {e}")
            return {}
    
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_interval = interval
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Performance monitoring loop"""
        while self.monitoring_active:
            try:
                self._capture_performance_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _capture_performance_metrics(self):
        """Capture current performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used
            memory_available = memory.available
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used = disk.used
            disk_free = disk.free
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            current_time = datetime.now()
            
            # Store metrics
            metrics = [
                PerformanceMetric("cpu_percent", cpu_percent, "percent", current_time, {}),
                PerformanceMetric("cpu_freq", cpu_freq.current if cpu_freq else 0, "MHz", current_time, {}),
                PerformanceMetric("memory_percent", memory_percent, "percent", current_time, {}),
                PerformanceMetric("memory_used", memory_used, "bytes", current_time, {}),
                PerformanceMetric("memory_available", memory_available, "bytes", current_time, {}),
                PerformanceMetric("disk_percent", disk_percent, "percent", current_time, {}),
                PerformanceMetric("disk_used", disk_used, "bytes", current_time, {}),
                PerformanceMetric("disk_free", disk_free, "bytes", current_time, {}),
                PerformanceMetric("network_bytes_sent", network.bytes_sent, "bytes", current_time, {}),
                PerformanceMetric("network_bytes_recv", network.bytes_recv, "bytes", current_time, {}),
                PerformanceMetric("process_memory_rss", process_memory.rss, "bytes", current_time, {}),
                PerformanceMetric("process_memory_vms", process_memory.vms, "bytes", current_time, {}),
                PerformanceMetric("process_cpu_percent", process_cpu, "percent", current_time, {})
            ]
            
            with self.lock:
                self.performance_metrics.extend(metrics)
                # Keep only last 1000 metrics
                if len(self.performance_metrics) > 1000:
                    self.performance_metrics = self.performance_metrics[-1000:]
        
        except Exception as e:
            logger.error(f"Error capturing performance metrics: {e}")
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Benchmark a function execution"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.cpu_percent()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            end_cpu = psutil.cpu_percent()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = end_cpu - start_cpu
            
            # Calculate input/output sizes
            input_size = len(str(args)) + len(str(kwargs))
            output_size = len(str(result)) if result is not None else 0
            
            benchmark_result = BenchmarkResult(
                benchmark_name=f"{func.__name__}_benchmark",
                function_name=func.__name__,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                input_size=input_size,
                output_size=output_size,
                success=True,
                error_message=None,
                timestamp=datetime.now(),
                metadata={
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                    "result_type": type(result).__name__ if result is not None else "None"
                }
            )
            
            with self.lock:
                self.benchmark_results.append(benchmark_result)
            
            return benchmark_result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            benchmark_result = BenchmarkResult(
                benchmark_name=f"{func.__name__}_benchmark",
                function_name=func.__name__,
                execution_time=execution_time,
                memory_usage=0,
                cpu_usage=0,
                input_size=len(str(args)) + len(str(kwargs)),
                output_size=0,
                success=False,
                error_message=str(e),
                timestamp=datetime.now(),
                metadata={"error": True}
            )
            
            with self.lock:
                self.benchmark_results.append(benchmark_result)
            
            return benchmark_result
    
    def benchmark_batch(self, func: Callable, inputs: List[Tuple], **kwargs) -> List[BenchmarkResult]:
        """Benchmark function with multiple inputs"""
        results = []
        
        for input_args in inputs:
            if isinstance(input_args, tuple):
                result = self.benchmark_function(func, *input_args, **kwargs)
            else:
                result = self.benchmark_function(func, input_args, **kwargs)
            results.append(result)
        
        return results
    
    def benchmark_concurrent(self, func: Callable, inputs: List[Tuple], 
                           max_workers: int = 4, **kwargs) -> List[BenchmarkResult]:
        """Benchmark function with concurrent execution"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for input_args in inputs:
                if isinstance(input_args, tuple):
                    future = executor.submit(self.benchmark_function, func, *input_args, **kwargs)
                else:
                    future = executor.submit(self.benchmark_function, func, input_args, **kwargs)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in concurrent benchmark: {e}")
        
        return results
    
    def create_performance_profile(self, profile_name: str, 
                                 function_name: Optional[str] = None,
                                 time_range: Optional[Tuple[datetime, datetime]] = None) -> PerformanceProfile:
        """Create performance profile from benchmark results"""
        with self.lock:
            # Filter results
            filtered_results = self.benchmark_results
            
            if function_name:
                filtered_results = [r for r in filtered_results if r.function_name == function_name]
            
            if time_range:
                start_time, end_time = time_range
                filtered_results = [
                    r for r in filtered_results 
                    if start_time <= r.timestamp <= end_time
                ]
            
            if not filtered_results:
                return PerformanceProfile(
                    profile_name=profile_name,
                    total_executions=0,
                    successful_executions=0,
                    failed_executions=0,
                    average_execution_time=0.0,
                    median_execution_time=0.0,
                    min_execution_time=0.0,
                    max_execution_time=0.0,
                    std_execution_time=0.0,
                    p95_execution_time=0.0,
                    p99_execution_time=0.0,
                    average_memory_usage=0.0,
                    peak_memory_usage=0.0,
                    average_cpu_usage=0.0,
                    peak_cpu_usage=0.0,
                    throughput_per_second=0.0,
                    success_rate=0.0,
                    created_at=datetime.now(),
                    metadata={}
                )
            
            # Calculate statistics
            execution_times = [r.execution_time for r in filtered_results if r.success]
            memory_usages = [r.memory_usage for r in filtered_results if r.success]
            cpu_usages = [r.cpu_usage for r in filtered_results if r.success]
            
            total_executions = len(filtered_results)
            successful_executions = len([r for r in filtered_results if r.success])
            failed_executions = total_executions - successful_executions
            
            if execution_times:
                avg_execution_time = statistics.mean(execution_times)
                median_execution_time = statistics.median(execution_times)
                min_execution_time = min(execution_times)
                max_execution_time = max(execution_times)
                std_execution_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
                p95_execution_time = np.percentile(execution_times, 95)
                p99_execution_time = np.percentile(execution_times, 99)
            else:
                avg_execution_time = median_execution_time = min_execution_time = max_execution_time = 0.0
                std_execution_time = p95_execution_time = p99_execution_time = 0.0
            
            if memory_usages:
                avg_memory_usage = statistics.mean(memory_usages)
                peak_memory_usage = max(memory_usages)
            else:
                avg_memory_usage = peak_memory_usage = 0.0
            
            if cpu_usages:
                avg_cpu_usage = statistics.mean(cpu_usages)
                peak_cpu_usage = max(cpu_usages)
            else:
                avg_cpu_usage = peak_cpu_usage = 0.0
            
            # Calculate throughput
            if filtered_results:
                time_span = (max(r.timestamp for r in filtered_results) - 
                           min(r.timestamp for r in filtered_results)).total_seconds()
                throughput_per_second = total_executions / max(time_span, 1)
            else:
                throughput_per_second = 0.0
            
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0.0
            
            profile = PerformanceProfile(
                profile_name=profile_name,
                total_executions=total_executions,
                successful_executions=successful_executions,
                failed_executions=failed_executions,
                average_execution_time=avg_execution_time,
                median_execution_time=median_execution_time,
                min_execution_time=min_execution_time,
                max_execution_time=max_execution_time,
                std_execution_time=std_execution_time,
                p95_execution_time=p95_execution_time,
                p99_execution_time=p99_execution_time,
                average_memory_usage=avg_memory_usage,
                peak_memory_usage=peak_memory_usage,
                average_cpu_usage=avg_cpu_usage,
                peak_cpu_usage=peak_cpu_usage,
                throughput_per_second=throughput_per_second,
                success_rate=success_rate,
                created_at=datetime.now(),
                metadata={
                    "function_name": function_name,
                    "time_range": time_range,
                    "baseline": self.system_baseline
                }
            )
            
            self.performance_profiles[profile_name] = profile
            return profile
    
    def get_performance_metrics(self, metric_name: Optional[str] = None, 
                              time_range: Optional[Tuple[datetime, datetime]] = None) -> List[PerformanceMetric]:
        """Get performance metrics"""
        with self.lock:
            metrics = self.performance_metrics
            
            if metric_name:
                metrics = [m for m in metrics if m.name == metric_name]
            
            if time_range:
                start_time, end_time = time_range
                metrics = [m for m in metrics if start_time <= m.timestamp <= end_time]
            
            return metrics
    
    def get_benchmark_results(self, function_name: Optional[str] = None,
                            time_range: Optional[Tuple[datetime, datetime]] = None) -> List[BenchmarkResult]:
        """Get benchmark results"""
        with self.lock:
            results = self.benchmark_results
            
            if function_name:
                results = [r for r in results if r.function_name == function_name]
            
            if time_range:
                start_time, end_time = time_range
                results = [r for r in results if start_time <= r.timestamp <= end_time]
            
            return results
    
    def get_performance_profile(self, profile_name: str) -> Optional[PerformanceProfile]:
        """Get performance profile"""
        return self.performance_profiles.get(profile_name)
    
    def get_all_performance_profiles(self) -> Dict[str, PerformanceProfile]:
        """Get all performance profiles"""
        return self.performance_profiles.copy()
    
    def check_performance_thresholds(self, profile: PerformanceProfile) -> Dict[str, Any]:
        """Check performance against thresholds"""
        alerts = []
        
        # Check execution time
        if profile.average_execution_time > self.performance_thresholds["execution_time"]["critical"]:
            alerts.append({
                "type": "execution_time",
                "level": "critical",
                "message": f"Average execution time {profile.average_execution_time:.2f}s exceeds critical threshold",
                "value": profile.average_execution_time,
                "threshold": self.performance_thresholds["execution_time"]["critical"]
            })
        elif profile.average_execution_time > self.performance_thresholds["execution_time"]["warning"]:
            alerts.append({
                "type": "execution_time",
                "level": "warning",
                "message": f"Average execution time {profile.average_execution_time:.2f}s exceeds warning threshold",
                "value": profile.average_execution_time,
                "threshold": self.performance_thresholds["execution_time"]["warning"]
            })
        
        # Check memory usage
        if profile.peak_memory_usage > self.performance_thresholds["memory_usage"]["critical"]:
            alerts.append({
                "type": "memory_usage",
                "level": "critical",
                "message": f"Peak memory usage {profile.peak_memory_usage / 1024 / 1024:.2f}MB exceeds critical threshold",
                "value": profile.peak_memory_usage,
                "threshold": self.performance_thresholds["memory_usage"]["critical"]
            })
        elif profile.peak_memory_usage > self.performance_thresholds["memory_usage"]["warning"]:
            alerts.append({
                "type": "memory_usage",
                "level": "warning",
                "message": f"Peak memory usage {profile.peak_memory_usage / 1024 / 1024:.2f}MB exceeds warning threshold",
                "value": profile.peak_memory_usage,
                "threshold": self.performance_thresholds["memory_usage"]["warning"]
            })
        
        # Check CPU usage
        if profile.peak_cpu_usage > self.performance_thresholds["cpu_usage"]["critical"]:
            alerts.append({
                "type": "cpu_usage",
                "level": "critical",
                "message": f"Peak CPU usage {profile.peak_cpu_usage:.2f}% exceeds critical threshold",
                "value": profile.peak_cpu_usage,
                "threshold": self.performance_thresholds["cpu_usage"]["critical"]
            })
        elif profile.peak_cpu_usage > self.performance_thresholds["cpu_usage"]["warning"]:
            alerts.append({
                "type": "cpu_usage",
                "level": "warning",
                "message": f"Peak CPU usage {profile.peak_cpu_usage:.2f}% exceeds warning threshold",
                "value": profile.peak_cpu_usage,
                "threshold": self.performance_thresholds["cpu_usage"]["warning"]
            })
        
        # Check throughput
        if profile.throughput_per_second < self.performance_thresholds["throughput"]["critical"]:
            alerts.append({
                "type": "throughput",
                "level": "critical",
                "message": f"Throughput {profile.throughput_per_second:.2f} req/s below critical threshold",
                "value": profile.throughput_per_second,
                "threshold": self.performance_thresholds["throughput"]["critical"]
            })
        elif profile.throughput_per_second < self.performance_thresholds["throughput"]["warning"]:
            alerts.append({
                "type": "throughput",
                "level": "warning",
                "message": f"Throughput {profile.throughput_per_second:.2f} req/s below warning threshold",
                "value": profile.throughput_per_second,
                "threshold": self.performance_thresholds["throughput"]["warning"]
            })
        
        return {
            "profile_name": profile.profile_name,
            "alerts": alerts,
            "alert_count": len(alerts),
            "critical_alerts": len([a for a in alerts if a["level"] == "critical"]),
            "warning_alerts": len([a for a in alerts if a["level"] == "warning"])
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        with self.lock:
            total_benchmarks = len(self.benchmark_results)
            successful_benchmarks = len([r for r in self.benchmark_results if r.success])
            failed_benchmarks = total_benchmarks - successful_benchmarks
            
            if self.benchmark_results:
                avg_execution_time = statistics.mean([r.execution_time for r in self.benchmark_results if r.success])
                total_execution_time = sum([r.execution_time for r in self.benchmark_results if r.success])
            else:
                avg_execution_time = total_execution_time = 0.0
            
            return {
                "total_benchmarks": total_benchmarks,
                "successful_benchmarks": successful_benchmarks,
                "failed_benchmarks": failed_benchmarks,
                "success_rate": (successful_benchmarks / total_benchmarks * 100) if total_benchmarks > 0 else 0.0,
                "average_execution_time": avg_execution_time,
                "total_execution_time": total_execution_time,
                "performance_profiles_count": len(self.performance_profiles),
                "performance_metrics_count": len(self.performance_metrics),
                "monitoring_active": self.monitoring_active,
                "system_baseline": self.system_baseline,
                "performance_thresholds": self.performance_thresholds
            }
    
    def export_performance_data(self, format: str = "json") -> str:
        """Export performance data"""
        with self.lock:
            data = {
                "performance_metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "timestamp": m.timestamp.isoformat(),
                        "metadata": m.metadata
                    }
                    for m in self.performance_metrics
                ],
                "benchmark_results": [
                    {
                        "benchmark_name": r.benchmark_name,
                        "function_name": r.function_name,
                        "execution_time": r.execution_time,
                        "memory_usage": r.memory_usage,
                        "cpu_usage": r.cpu_usage,
                        "input_size": r.input_size,
                        "output_size": r.output_size,
                        "success": r.success,
                        "error_message": r.error_message,
                        "timestamp": r.timestamp.isoformat(),
                        "metadata": r.metadata
                    }
                    for r in self.benchmark_results
                ],
                "performance_profiles": {
                    name: {
                        "profile_name": p.profile_name,
                        "total_executions": p.total_executions,
                        "successful_executions": p.successful_executions,
                        "failed_executions": p.failed_executions,
                        "average_execution_time": p.average_execution_time,
                        "median_execution_time": p.median_execution_time,
                        "min_execution_time": p.min_execution_time,
                        "max_execution_time": p.max_execution_time,
                        "std_execution_time": p.std_execution_time,
                        "p95_execution_time": p.p95_execution_time,
                        "p99_execution_time": p.p99_execution_time,
                        "average_memory_usage": p.average_memory_usage,
                        "peak_memory_usage": p.peak_memory_usage,
                        "average_cpu_usage": p.average_cpu_usage,
                        "peak_cpu_usage": p.peak_cpu_usage,
                        "throughput_per_second": p.throughput_per_second,
                        "success_rate": p.success_rate,
                        "created_at": p.created_at.isoformat(),
                        "metadata": p.metadata
                    }
                    for name, p in self.performance_profiles.items()
                },
                "export_timestamp": datetime.now().isoformat()
            }
            
            if format == "json":
                return json.dumps(data, indent=2)
            else:
                return str(data)
    
    def clear_performance_data(self):
        """Clear all performance data"""
        with self.lock:
            self.performance_metrics.clear()
            self.benchmark_results.clear()
            self.performance_profiles.clear()
        logger.info("Performance data cleared")

# Global performance instance
ml_nlp_benchmark_performance = MLNLPBenchmarkPerformance()

def get_performance() -> MLNLPBenchmarkPerformance:
    """Get the global performance instance"""
    return ml_nlp_benchmark_performance

def start_performance_monitoring(interval: float = 1.0):
    """Start performance monitoring"""
    ml_nlp_benchmark_performance.start_monitoring(interval)

def stop_performance_monitoring():
    """Stop performance monitoring"""
    ml_nlp_benchmark_performance.stop_monitoring()

def benchmark_function(func: Callable, *args, **kwargs) -> BenchmarkResult:
    """Benchmark a function execution"""
    return ml_nlp_benchmark_performance.benchmark_function(func, *args, **kwargs)

def benchmark_batch(func: Callable, inputs: List[Tuple], **kwargs) -> List[BenchmarkResult]:
    """Benchmark function with multiple inputs"""
    return ml_nlp_benchmark_performance.benchmark_batch(func, inputs, **kwargs)

def benchmark_concurrent(func: Callable, inputs: List[Tuple], 
                        max_workers: int = 4, **kwargs) -> List[BenchmarkResult]:
    """Benchmark function with concurrent execution"""
    return ml_nlp_benchmark_performance.benchmark_concurrent(func, inputs, max_workers, **kwargs)

def create_performance_profile(profile_name: str, 
                             function_name: Optional[str] = None,
                             time_range: Optional[Tuple[datetime, datetime]] = None) -> PerformanceProfile:
    """Create performance profile from benchmark results"""
    return ml_nlp_benchmark_performance.create_performance_profile(profile_name, function_name, time_range)

def get_performance_summary() -> Dict[str, Any]:
    """Get overall performance summary"""
    return ml_nlp_benchmark_performance.get_performance_summary()

def export_performance_data(format: str = "json") -> str:
    """Export performance data"""
    return ml_nlp_benchmark_performance.export_performance_data(format)

def clear_performance_data():
    """Clear all performance data"""
    ml_nlp_benchmark_performance.clear_performance_data()











