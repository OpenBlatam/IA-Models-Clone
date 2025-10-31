from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import json
import logging
        import gc
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Real-time Performance Monitor for NotebookLM AI System
Continuous monitoring and optimization
"""


logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    io_read_mb: float
    io_write_mb: float
    thread_count: int
    process_count: int
    network_connections: int
    disk_usage_percent: float

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, history_size: int = 1000, monitor_interval: float = 1.0):
        
    """__init__ function."""
self.history_size = history_size
        self.monitor_interval = monitor_interval
        self.metrics_history = deque(maxlen=history_size)
        self.is_monitoring = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        self.alerts = []
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0
        }
        
    async def start_monitoring(self) -> Any:
        """Start continuous monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self) -> Any:
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
        
    def _monitor_loop(self) -> Any:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self._store_metrics(metrics)
                self._check_alerts(metrics)
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        io_counters = process.io_counters()
        
        # Get disk usage
        disk_usage = psutil.disk_usage('/')
        
        # Get network connections
        try:
            network_connections = len(process.connections())
        except:
            network_connections = 0
            
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=process.cpu_percent(),
            memory_mb=memory_info.rss / 1024 / 1024,
            memory_percent=process.memory_percent(),
            io_read_mb=io_counters.read_bytes / 1024 / 1024,
            io_write_mb=io_counters.write_bytes / 1024 / 1024,
            thread_count=process.num_threads(),
            process_count=len(psutil.pids()),
            network_connections=network_connections,
            disk_usage_percent=disk_usage.percent
        )
        
    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in history"""
        with self._lock:
            self.metrics_history.append(metrics)
            
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            
        if metrics.disk_usage_percent > self.thresholds["disk_usage_percent"]:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
            
        if alerts:
            self.alerts.extend(alerts)
            logger.warning(f"Performance alerts: {alerts}")
            
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.metrics_history:
            return {}
            
        with self._lock:
            latest = self.metrics_history[-1]
            return asdict(latest)
            
    def get_metrics_history(self, minutes: int = 10) -> List[Dict[str, Any]]:
        """Get metrics history for specified time period"""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            recent_metrics = [
                asdict(metric) for metric in self.metrics_history
                if metric.timestamp >= cutoff_time
            ]
            
        return recent_metrics
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics_history:
            return {}
            
        with self._lock:
            metrics_list = list(self.metrics_history)
            
        if not metrics_list:
            return {}
            
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in metrics_list]
        memory_values = [m.memory_percent for m in metrics_list]
        
        return {
            "cpu": {
                "current": cpu_values[-1],
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "current_mb": metrics_list[-1].memory_mb,
                "current_percent": memory_values[-1],
                "average_percent": sum(memory_values) / len(memory_values),
                "max_percent": max(memory_values)
            },
            "io": {
                "total_read_mb": sum(m.io_read_mb for m in metrics_list),
                "total_write_mb": sum(m.io_write_mb for m in metrics_list)
            },
            "system": {
                "thread_count": metrics_list[-1].thread_count,
                "process_count": metrics_list[-1].process_count,
                "network_connections": metrics_list[-1].network_connections
            },
            "alerts": self.alerts[-10:] if self.alerts else []
        }
        
    def export_metrics(self, filename: str):
        """Export metrics to JSON file"""
        with self._lock:
            metrics_data = [asdict(m) for m in self.metrics_history]
            
        with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metrics_data, f, indent=2)
            
        logger.info(f"Metrics exported to {filename}")
        
    def set_thresholds(self, **kwargs) -> Any:
        """Set performance thresholds"""
        self.thresholds.update(kwargs)
        logger.info(f"Updated thresholds: {kwargs}")

class OptimizedPerformanceMonitor(PerformanceMonitor):
    """Enhanced performance monitor with optimization features"""
    
    def __init__(self, *args, **kwargs) -> Any:
        super().__init__(*args, **kwargs)
        self.optimization_history = []
        self.auto_optimize = True
        
    async def auto_optimize_system(self) -> Any:
        """Automatically optimize system based on metrics"""
        if not self.auto_optimize:
            return
            
        current = self.get_current_metrics()
        if not current:
            return
            
        optimizations = []
        
        # Memory optimization
        if current.get('memory_percent', 0) > 80:
            optimizations.append(await self._optimize_memory())
            
        # CPU optimization
        if current.get('cpu_percent', 0) > 90:
            optimizations.append(await self._optimize_cpu())
            
        # I/O optimization
        if current.get('io_read_mb', 0) > 1000 or current.get('io_write_mb', 0) > 1000:
            optimizations.append(await self._optimize_io())
            
        if optimizations:
            self.optimization_history.extend(optimizations)
            logger.info(f"Auto-optimizations applied: {optimizations}")
            
    async def _optimize_memory(self) -> str:
        """Memory optimization"""
        gc.collect()
        return "memory_gc"
        
    async def _optimize_cpu(self) -> str:
        """CPU optimization"""
        # Reduce thread priority or adjust scheduling
        return "cpu_scheduling"
        
    async def _optimize_io(self) -> str:
        """I/O optimization"""
        # Flush buffers or adjust I/O scheduling
        return "io_buffering"

async def main():
    """Demo performance monitoring"""
    monitor = OptimizedPerformanceMonitor(monitor_interval=2.0)
    
    try:
        # Start monitoring
        await monitor.start_monitoring()
        
        # Monitor for 30 seconds
        for i in range(15):
            await asyncio.sleep(2)
            
            # Get current metrics
            current = monitor.get_current_metrics()
            if current:
                print(f"CPU: {current['cpu_percent']:.1f}%, "
                      f"Memory: {current['memory_mb']:.1f}MB "
                      f"({current['memory_percent']:.1f}%)")
                
            # Get summary every 10 seconds
            if i % 5 == 0:
                summary = monitor.get_performance_summary()
                if summary:
                    print(f"Summary - CPU avg: {summary['cpu']['average']:.1f}%, "
                          f"Memory avg: {summary['memory']['average_percent']:.1f}%")
                    
            # Auto-optimize
            await monitor.auto_optimize_system()
            
    finally:
        monitor.stop_monitoring()
        
        # Export final metrics
        monitor.export_metrics("performance_metrics.json")
        
        # Print final summary
        final_summary = monitor.get_performance_summary()
        print(f"Final summary: {final_summary}")

match __name__:
    case "__main__":
    asyncio.run(main()) 