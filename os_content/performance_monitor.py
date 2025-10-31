from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import time
import psutil
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Performance Monitor for OS Content UGC Video Generator
Tracks system performance, memory usage, and processing times
"""


logger = logging.getLogger("os_content.performance")

@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_connections: int
    processing_time: Optional[float] = None
    error_count: int = 0
    success_count: int = 0

class PerformanceMonitor:
    """Monitor system performance and track optimizations"""
    
    def __init__(self, log_file: str = "performance_metrics.json"):
        
    """__init__ function."""
self.log_file = Path(log_file)
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
    def get_system_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network connections (approximation for active connections)
            connections = len(psutil.net_connections())
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                active_connections=connections,
                error_count=self.error_count,
                success_count=self.request_count - self.error_count
            )
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_usage_percent=0.0,
                active_connections=0
            )
    
    def record_request(self, processing_time: float, success: bool = True):
        """Record a request with processing time"""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        metrics = self.get_system_metrics()
        metrics.processing_time = processing_time
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics to prevent memory issues
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_average_processing_time(self) -> float:
        """Get average processing time"""
        if not self.metrics_history:
            return 0.0
        
        times = [m.processing_time for m in self.metrics_history if m.processing_time is not None]
        return sum(times) / len(times) if times else 0.0
    
    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        if self.request_count == 0:
            return 100.0
        return ((self.request_count - self.error_count) / self.request_count) * 100
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    def save_metrics(self) -> Any:
        """Save metrics to JSON file"""
        try:
            data = {
                "summary": {
                    "uptime_seconds": self.get_uptime(),
                    "total_requests": self.request_count,
                    "success_rate": self.get_success_rate(),
                    "average_processing_time": self.get_average_processing_time(),
                    "error_count": self.error_count
                },
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "memory_used_mb": m.memory_used_mb,
                        "disk_usage_percent": m.disk_usage_percent,
                        "active_connections": m.active_connections,
                        "processing_time": m.processing_time,
                        "error_count": m.error_count,
                        "success_count": m.success_count
                    }
                    for m in self.metrics_history
                ]
            }
            
            with open(self.log_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(data, f, indent=2)
                
            logger.info(f"Performance metrics saved to {self.log_file}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def print_summary(self) -> Any:
        """Print performance summary"""
        print("\n" + "="*50)
        print("OS CONTENT PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Uptime: {self.get_uptime():.2f} seconds")
        print(f"Total Requests: {self.request_count}")
        print(f"Success Rate: {self.get_success_rate():.2f}%")
        print(f"Average Processing Time: {self.get_average_processing_time():.3f}s")
        print(f"Error Count: {self.error_count}")
        
        if self.metrics_history:
            latest = self.metrics_history[-1]
            print(f"\nCurrent System Status:")
            print(f"CPU Usage: {latest.cpu_percent:.1f}%")
            print(f"Memory Usage: {latest.memory_percent:.1f}% ({latest.memory_used_mb:.1f} MB)")
            print(f"Disk Usage: {latest.disk_usage_percent:.1f}%")
            print(f"Active Connections: {latest.active_connections}")
        
        print("="*50)

# Global monitor instance
monitor = PerformanceMonitor()

async def monitor_request(func) -> Any:
    """Decorator to monitor request performance"""
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        success = True
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            logger.error(f"Request failed: {e}")
            raise
        finally:
            processing_time = time.time() - start_time
            monitor.record_request(processing_time, success)
    
    return wrapper

async def periodic_metrics_save(interval: int = 300):
    """Periodically save metrics to file"""
    while True:
        await asyncio.sleep(interval)
        monitor.save_metrics()
        logger.info("Periodic metrics save completed")

def start_monitoring():
    """Start the performance monitoring"""
    logger.info("Performance monitoring started")
    
    # Start periodic save task
    loop = asyncio.get_event_loop()
    loop.create_task(periodic_metrics_save())

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Simulate some metrics
    for i in range(10):
        monitor.record_request(0.1 + i * 0.01, success=i < 9)
        time.sleep(0.1)
    
    monitor.print_summary()
    monitor.save_metrics() 