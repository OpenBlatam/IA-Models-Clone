"""
Model Monitoring Utilities
===========================

Utilities for monitoring models in production.
"""

import logging
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import time

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitor model performance and health in production.
    """
    
    def __init__(
        self,
        model: nn.Module,
        window_size: int = 100,
        alert_threshold: float = 0.1
    ):
        """
        Initialize model monitor.
        
        Args:
            model: PyTorch model
            window_size: Size of sliding window for metrics
            alert_threshold: Threshold for alerts
        """
        self.model = model
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Metrics storage
        self.latency_history = deque(maxlen=window_size)
        self.error_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        
        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()
    
    def record_prediction(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        latency: float,
        error: Optional[Exception] = None
    ) -> None:
        """
        Record a prediction.
        
        Args:
            inputs: Input tensor
            outputs: Output tensor
            latency: Prediction latency in seconds
            error: Error if any
        """
        self.total_requests += 1
        self.latency_history.append(latency)
        
        if error:
            self.total_errors += 1
            self.error_history.append(1)
        else:
            self.error_history.append(0)
            self.prediction_history.append(outputs.detach().cpu().numpy())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Returns:
            Dictionary with statistics
        """
        uptime = time.time() - self.start_time
        
        avg_latency = np.mean(self.latency_history) if self.latency_history else 0.0
        p95_latency = np.percentile(self.latency_history, 95) if self.latency_history else 0.0
        p99_latency = np.percentile(self.latency_history, 99) if self.latency_history else 0.0
        
        error_rate = self.total_errors / max(self.total_requests, 1)
        
        return {
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'error_rate': error_rate,
            'uptime_seconds': uptime,
            'avg_latency_ms': avg_latency * 1000,
            'p95_latency_ms': p95_latency * 1000,
            'p99_latency_ms': p99_latency * 1000,
            'requests_per_second': self.total_requests / max(uptime, 1)
        }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for alerts.
        
        Returns:
            List of alerts
        """
        alerts = []
        stats = self.get_statistics()
        
        # High error rate
        if stats['error_rate'] > self.alert_threshold:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f"Error rate is {stats['error_rate']:.2%} (threshold: {self.alert_threshold:.2%})"
            })
        
        # High latency
        if stats['p95_latency_ms'] > 1000:  # 1 second
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"P95 latency is {stats['p95_latency_ms']:.2f}ms"
            })
        
        return alerts


class DriftDetector:
    """
    Detect data drift in production.
    """
    
    def __init__(
        self,
        reference_data: np.ndarray,
        threshold: float = 0.1
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference data distribution
            threshold: Drift detection threshold
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)
    
    def detect_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in current data.
        
        Args:
            current_data: Current data to check
            
        Returns:
            Drift detection results
        """
        current_mean = np.mean(current_data, axis=0)
        current_std = np.std(current_data, axis=0)
        
        # Calculate drift metrics
        mean_drift = np.abs(current_mean - self.reference_mean) / (self.reference_std + 1e-8)
        std_drift = np.abs(current_std - self.reference_std) / (self.reference_std + 1e-8)
        
        max_mean_drift = np.max(mean_drift)
        max_std_drift = np.max(std_drift)
        
        has_drift = max_mean_drift > self.threshold or max_std_drift > self.threshold
        
        return {
            'has_drift': has_drift,
            'mean_drift': float(max_mean_drift),
            'std_drift': float(max_std_drift),
            'severity': 'high' if has_drift else 'low'
        }


class PerformanceMonitor:
    """
    Monitor model performance metrics.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Size of sliding window
        """
        self.window_size = window_size
        self.metrics_history = {
            'accuracy': deque(maxlen=window_size),
            'loss': deque(maxlen=window_size),
            'f1_score': deque(maxlen=window_size)
        }
    
    def record_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Record performance metrics.
        
        Args:
            metrics: Dictionary with metric names and values
        """
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Get average metrics over window.
        
        Returns:
            Dictionary with average metrics
        """
        return {
            key: np.mean(list(values)) if values else 0.0
            for key, values in self.metrics_history.items()
        }
    
    def get_trend(self, metric_name: str) -> str:
        """
        Get trend for metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Trend ('increasing', 'decreasing', 'stable')
        """
        if metric_name not in self.metrics_history:
            return 'unknown'
        
        values = list(self.metrics_history[metric_name])
        if len(values) < 2:
            return 'insufficient_data'
        
        recent = np.mean(values[-10:]) if len(values) >= 10 else values[-1]
        older = np.mean(values[:10]) if len(values) >= 10 else values[0]
        
        if recent > older * 1.05:
            return 'increasing'
        elif recent < older * 0.95:
            return 'decreasing'
        else:
            return 'stable'



