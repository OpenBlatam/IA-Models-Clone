#!/usr/bin/env python3
"""
Performance Optimizer - Advanced AI Document Processor
====================================================

Advanced performance optimization utilities for maximum speed and efficiency.
"""

import asyncio
import time
import psutil
import gc
import os
import sys
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import threading
from functools import wraps, lru_cache
import weakref
import tracemalloc
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    timestamp: float = 0.0

class PerformanceOptimizer:
    """Advanced performance optimizer for the AI document processor."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_rules: Dict[str, Callable] = {}
        self.performance_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_io': 1000.0,  # MB/s
            'network_io': 100.0  # MB/s
        }
        self.optimization_enabled = True
        self.auto_optimize = True
        self.monitoring_interval = 5.0  # seconds
        self._monitoring_task = None
        self._stop_monitoring = False
        
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self._monitoring_task is None:
            self._stop_monitoring = False
            self._monitoring_task = asyncio.create_task(self._monitor_performance())
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._stop_monitoring = True
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
        logger.info("Performance monitoring stopped")
    
    async def _monitor_performance(self):
        """Continuous performance monitoring loop."""
        while not self._stop_monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                
                # Auto-optimize if enabled
                if self.auto_optimize:
                    await self._auto_optimize(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_dict = {
            'read_bytes': disk_io.read_bytes if disk_io else 0,
            'write_bytes': disk_io.write_bytes if disk_io else 0,
            'read_count': disk_io.read_count if disk_io else 0,
            'write_count': disk_io.write_count if disk_io else 0
        }
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_io_dict = {
            'bytes_sent': network_io.bytes_sent if network_io else 0,
            'bytes_recv': network_io.bytes_recv if network_io else 0,
            'packets_sent': network_io.packets_sent if network_io else 0,
            'packets_recv': network_io.packets_recv if network_io else 0
        }
        
        # GPU usage (if available)
        gpu_usage = None
        gpu_memory = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage = gpu.load * 100
                gpu_memory = gpu.memoryUtil * 100
        except ImportError:
            pass
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            disk_io=disk_io_dict,
            network_io=network_io_dict,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            timestamp=time.time()
        )
    
    async def _auto_optimize(self, metrics: PerformanceMetrics):
        """Automatically optimize based on current metrics."""
        optimizations_applied = []
        
        # High CPU usage optimization
        if metrics.cpu_usage > self.performance_thresholds['cpu_usage']:
            await self._optimize_cpu_usage()
            optimizations_applied.append("CPU optimization")
        
        # High memory usage optimization
        if metrics.memory_usage > self.performance_thresholds['memory_usage']:
            await self._optimize_memory_usage()
            optimizations_applied.append("Memory optimization")
        
        # High disk I/O optimization
        if metrics.disk_io['read_bytes'] > self.performance_thresholds['disk_io'] * 1024**2:
            await self._optimize_disk_io()
            optimizations_applied.append("Disk I/O optimization")
        
        if optimizations_applied:
            logger.info(f"Auto-optimizations applied: {', '.join(optimizations_applied)}")
    
    async def _optimize_cpu_usage(self):
        """Optimize CPU usage."""
        # Reduce thread pool size
        # Force garbage collection
        gc.collect()
        
        # Adjust asyncio event loop
        loop = asyncio.get_event_loop()
        if hasattr(loop, 'set_default_executor'):
            # Use fewer threads for CPU-bound tasks
            executor = ThreadPoolExecutor(max_workers=max(1, mp.cpu_count() // 2))
            loop.set_default_executor(executor)
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage."""
        # Force garbage collection
        gc.collect()
        
        # Clear caches if available
        try:
            import sys
            # Clear module cache
            for module_name in list(sys.modules.keys()):
                if module_name.startswith('_') and module_name not in ['__main__']:
                    del sys.modules[module_name]
        except Exception:
            pass
        
        # Clear weak references
        gc.collect()
    
    async def _optimize_disk_io(self):
        """Optimize disk I/O."""
        # Sync filesystem
        try:
            os.sync()
        except Exception:
            pass
    
    def add_optimization_rule(self, name: str, rule: Callable):
        """Add a custom optimization rule."""
        self.optimization_rules[name] = rule
        logger.info(f"Added optimization rule: {name}")
    
    def remove_optimization_rule(self, name: str):
        """Remove an optimization rule."""
        if name in self.optimization_rules:
            del self.optimization_rules[name]
            logger.info(f"Removed optimization rule: {name}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        latest = self.metrics_history[-1]
        avg_cpu = sum(m.cpu_usage for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_usage for m in self.metrics_history) / len(self.metrics_history)
        
        return {
            "current": {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "memory_available_gb": latest.memory_available,
                "gpu_usage": latest.gpu_usage,
                "gpu_memory": latest.gpu_memory
            },
            "averages": {
                "cpu_usage": avg_cpu,
                "memory_usage": avg_memory
            },
            "optimization_enabled": self.optimization_enabled,
            "auto_optimize": self.auto_optimize,
            "monitoring_interval": self.monitoring_interval,
            "metrics_count": len(self.metrics_history)
        }
    
    def display_performance_dashboard(self):
        """Display performance dashboard."""
        summary = self.get_performance_summary()
        
        if "error" in summary:
            console.print(f"[red]Error: {summary['error']}[/red]")
            return
        
        # Create performance table
        table = Table(title="Performance Dashboard")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="green")
        table.add_column("Average", style="yellow")
        table.add_column("Status", style="magenta")
        
        current = summary["current"]
        averages = summary["averages"]
        
        # CPU
        cpu_status = "🟢 Good" if current["cpu_usage"] < 70 else "🟡 High" if current["cpu_usage"] < 90 else "🔴 Critical"
        table.add_row("CPU Usage", f"{current['cpu_usage']:.1f}%", f"{averages['cpu_usage']:.1f}%", cpu_status)
        
        # Memory
        memory_status = "🟢 Good" if current["memory_usage"] < 70 else "🟡 High" if current["memory_usage"] < 90 else "🔴 Critical"
        table.add_row("Memory Usage", f"{current['memory_usage']:.1f}%", f"{averages['memory_usage']:.1f}%", memory_status)
        
        # Available Memory
        table.add_row("Available Memory", f"{current['memory_available_gb']:.1f} GB", "-", "🟢 Good" if current['memory_available_gb'] > 1 else "🔴 Low")
        
        # GPU
        if current["gpu_usage"] is not None:
            gpu_status = "🟢 Good" if current["gpu_usage"] < 70 else "🟡 High" if current["gpu_usage"] < 90 else "🔴 Critical"
            table.add_row("GPU Usage", f"{current['gpu_usage']:.1f}%", "-", gpu_status)
        
        if current["gpu_memory"] is not None:
            gpu_mem_status = "🟢 Good" if current["gpu_memory"] < 70 else "🟡 High" if current["gpu_memory"] < 90 else "🔴 Critical"
            table.add_row("GPU Memory", f"{current['gpu_memory']:.1f}%", "-", gpu_mem_status)
        
        console.print(table)
        
        # Optimization status
        opt_table = Table(title="Optimization Status")
        opt_table.add_column("Setting", style="cyan")
        opt_table.add_column("Value", style="green")
        
        opt_table.add_row("Optimization Enabled", "✅ Yes" if summary["optimization_enabled"] else "❌ No")
        opt_table.add_row("Auto Optimize", "✅ Yes" if summary["auto_optimize"] else "❌ No")
        opt_table.add_row("Monitoring Interval", f"{summary['monitoring_interval']}s")
        opt_table.add_row("Metrics Collected", str(summary["metrics_count"]))
        
        console.print(opt_table)

# Performance decorators
def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024**2  # MB
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(f"Function {func.__name__} executed in {execution_time:.3f}s, memory delta: {memory_delta:.1f}MB")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024**2  # MB
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(f"Function {func.__name__} executed in {execution_time:.3f}s, memory delta: {memory_delta:.1f}MB")
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def memory_efficient(func):
    """Decorator to optimize memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before function
        gc.collect()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Force garbage collection after function
            gc.collect()
    
    return wrapper

def cpu_optimized(func):
    """Decorator to optimize CPU usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Set CPU affinity if possible
        try:
            import psutil
            process = psutil.Process()
            # Use only half of available CPUs for this function
            available_cpus = list(range(0, psutil.cpu_count(), 2))
            process.cpu_affinity(available_cpus)
        except Exception:
            pass
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Reset CPU affinity
            try:
                import psutil
                process = psutil.Process()
                process.cpu_affinity(list(range(psutil.cpu_count())))
            except Exception:
                pass
    
    return wrapper

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Utility functions
def start_performance_monitoring():
    """Start global performance monitoring."""
    performance_optimizer.start_monitoring()

def stop_performance_monitoring():
    """Stop global performance monitoring."""
    performance_optimizer.stop_monitoring()

def get_performance_summary():
    """Get global performance summary."""
    return performance_optimizer.get_performance_summary()

def display_performance_dashboard():
    """Display global performance dashboard."""
    performance_optimizer.display_performance_dashboard()

def optimize_system():
    """Run system optimization."""
    gc.collect()
    try:
        os.sync()
    except Exception:
        pass
    logger.info("System optimization completed")

if __name__ == "__main__":
    # Example usage
    async def main():
        # Start monitoring
        start_performance_monitoring()
        
        # Wait a bit to collect metrics
        await asyncio.sleep(10)
        
        # Display dashboard
        display_performance_dashboard()
        
        # Stop monitoring
        stop_performance_monitoring()
    
    asyncio.run(main())
















