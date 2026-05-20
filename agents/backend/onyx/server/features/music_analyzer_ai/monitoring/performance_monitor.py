"""
Performance Monitor
Monitor training and inference performance
"""

from typing import Dict, Any, Optional, List
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = defaultdict(int)
    
    def start_timer(self, name: str):
        """Start timer"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timer and return elapsed time"""
        if name not in self.timers:
            logger.warning(f"Timer {name} not started")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        self.metrics[f"{name}_time"].append(elapsed)
        del self.timers[name]
        return elapsed
    
    def record_metric(self, name: str, value: float):
        """Record a metric"""
        self.metrics[name].append(value)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment counter"""
        self.counters[name] += value
    
    def get_average(self, name: str) -> float:
        """Get average of metric"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "averages": {},
            "counts": {},
            "totals": {}
        }
        
        for name, values in self.metrics.items():
            summary["averages"][name] = sum(values) / len(values) if values else 0.0
            summary["totals"][name] = sum(values)
        
        summary["counts"] = dict(self.counters)
        return summary
    
    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()
        self.timers.clear()
        self.counters.clear()



