"""
Metrics Logger
Specialized logger for metrics
"""

from typing import Dict, Any, Optional, List
import logging
from collections import defaultdict


class MetricsLogger:
    """Logger for metrics"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """Log metrics"""
        prefix_str = f"{prefix}_" if prefix else ""
        
        for key, value in metrics.items():
            full_key = f"{prefix_str}{key}"
            self.metrics_history[full_key].append(value)
            
            if step is not None:
                self.logger.info(f"Step {step}: {full_key} = {value:.6f}")
            else:
                self.logger.info(f"{full_key} = {value:.6f}")
    
    def log_metric_summary(self, metric_name: str):
        """Log metric summary"""
        if metric_name not in self.metrics_history:
            self.logger.warning(f"Metric {metric_name} not found")
            return
        
        values = self.metrics_history[metric_name]
        if not values:
            return
        
        summary = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1]
        }
        
        self.logger.info(f"{metric_name} summary: {summary}")
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get metric history"""
        return self.metrics_history.get(metric_name, []).copy()
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()



