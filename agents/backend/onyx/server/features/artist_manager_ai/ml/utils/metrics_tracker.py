"""
Metrics Tracker
===============

Utility for tracking training metrics.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks training and evaluation metrics.
    
    Features:
    - Metric logging
    - Metric aggregation
    - Best metric tracking
    - Metric history
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.best_metrics: Dict[str, float] = {}
        self._logger = logger
    
    def log(self, name: str, value: float, step: Optional[int] = None):
        """
        Log metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (optional)
        """
        self.metrics[name].append(value)
        
        # Update best metric
        if name not in self.best_metrics:
            self.best_metrics[name] = value
        else:
            # For loss metrics, lower is better
            if "loss" in name.lower():
                if value < self.best_metrics[name]:
                    self.best_metrics[name] = value
            # For other metrics, higher is better
            else:
                if value > self.best_metrics[name]:
                    self.best_metrics[name] = value
    
    def log_batch(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        for name, value in metrics.items():
            self.log(name, value, step)
    
    def get_metric(self, name: str) -> List[float]:
        """
        Get metric history.
        
        Args:
            name: Metric name
        
        Returns:
            List of metric values
        """
        return self.metrics.get(name, [])
    
    def get_best(self, name: str) -> Optional[float]:
        """
        Get best metric value.
        
        Args:
            name: Metric name
        
        Returns:
            Best value or None
        """
        return self.best_metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        """Get all metrics."""
        return dict(self.metrics)
    
    def get_all_best(self) -> Dict[str, float]:
        """Get all best metrics."""
        return dict(self.best_metrics)
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.best_metrics.clear()
        self._logger.info("Metrics tracker reset")
    
    def summary(self) -> str:
        """
        Get metrics summary.
        
        Returns:
            Summary string
        """
        summary_lines = ["Metrics Summary:"]
        
        for name, values in self.metrics.items():
            if values:
                avg = sum(values) / len(values)
                best = self.best_metrics.get(name, "N/A")
                summary_lines.append(f"  {name}: avg={avg:.4f}, best={best}")
        
        return "\n".join(summary_lines)




