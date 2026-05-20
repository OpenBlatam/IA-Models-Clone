"""
Training Monitor Module
=======================

Advanced training monitoring and visualization.

Author: BUL System
Date: 2024
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Advanced training monitoring and tracking.
    
    Provides:
    - Real-time metrics tracking
    - Training history
    - Performance analysis
    - Export capabilities
    
    Example:
        >>> monitor = TrainingMonitor()
        >>> monitor.track_metric("loss", 0.5, step=100)
        >>> history = monitor.get_history()
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize TrainingMonitor.
        
        Args:
            output_dir: Optional directory to save monitoring data
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.current_step = 0
    
    def track_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Track a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (auto-incremented if None)
        """
        if step is None:
            step = self.current_step
            self.current_step += 1
        
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        
        self.metrics_history[name].append({
            "step": step,
            "value": value
        })
    
    def track_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Track multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (auto-incremented if None)
        """
        if step is None:
            step = self.current_step
            self.current_step += 1
        
        for name, value in metrics.items():
            self.track_metric(name, value, step)
    
    def get_history(self, metric_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get metric history.
        
        Args:
            metric_name: Optional specific metric name
            
        Returns:
            Dictionary with metric history
        """
        if metric_name:
            return {metric_name: self.metrics_history.get(metric_name, [])}
        return self.metrics_history.copy()
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Get latest values for all metrics.
        
        Returns:
            Dictionary with latest metric values
        """
        latest = {}
        for name, history in self.metrics_history.items():
            if history:
                latest[name] = history[-1]["value"]
        return latest
    
    def get_best_metric(self, metric_name: str, mode: str = "min") -> Optional[Dict[str, Any]]:
        """
        Get best value for a metric.
        
        Args:
            metric_name: Name of metric
            mode: "min" or "max"
            
        Returns:
            Dictionary with best value and step, or None
        """
        if metric_name not in self.metrics_history:
            return None
        
        history = self.metrics_history[metric_name]
        if not history:
            return None
        
        if mode == "min":
            best = min(history, key=lambda x: x["value"])
        else:
            best = max(history, key=lambda x: x["value"])
        
        return best
    
    def export_to_json(self, file_path: Path) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Exported metrics to {file_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of training metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "total_steps": self.current_step,
            "metrics": {}
        }
        
        for name, history in self.metrics_history.items():
            if history:
                values = [h["value"] for h in history]
                summary["metrics"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1],
                }
        
        return summary

