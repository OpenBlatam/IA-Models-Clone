import time
import asyncio
from typing import Dict, List, Any
from .settings import PerformanceMetrics

class PerformanceMonitor:
    """System-wide performance monitoring."""
    
    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetrics]] = {}
        self.system_start_time = time.time()
        self._lock = asyncio.Lock()
    
    async def record_operation(self, component_name: str, metrics: PerformanceMetrics):
        """Record operation metrics."""
        async with self._lock:
            if component_name not in self.metrics:
                self.metrics[component_name] = []
            self.metrics[component_name].append(metrics)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        total_operations = sum(len(metrics) for metrics in self.metrics.values())
        total_duration = sum(
            sum(m.duration for m in metrics if m.duration)
            for metrics in self.metrics.values()
        )
        
        return {
            "uptime": time.time() - self.system_start_time,
            "total_operations": total_operations,
            "total_duration": total_duration,
            "avg_operation_duration": total_duration / max(total_operations, 1),
            "component_count": len(self.metrics),
            "components": {
                name: {
                    "operation_count": len(metrics),
                    "success_rate": sum(1 for m in metrics if m.success) / len(metrics) if metrics else 0
                }
                for name, metrics in self.metrics.items()
            }
        }
