"""
Anomaly Detector
================

ML-based anomaly detection.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Anomaly detection result."""
    metric: str
    value: float
    expected_range: Tuple[float, float]
    severity: str  # low, medium, high, critical
    timestamp: str


class AnomalyDetector:
    """ML-based anomaly detector."""
    
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold  # Z-score threshold
        self._metrics_history: Dict[str, deque] = {}
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._anomalies: List[Anomaly] = []
    
    def record_metric(self, metric_name: str, value: float):
        """Record metric for anomaly detection."""
        if metric_name not in self._metrics_history:
            self._metrics_history[metric_name] = deque(maxlen=self.window_size)
        
        self._metrics_history[metric_name].append(value)
        
        # Update baseline
        self._update_baseline(metric_name)
        
        # Check for anomaly
        anomaly = self.detect_anomaly(metric_name, value)
        if anomaly:
            self._anomalies.append(anomaly)
            logger.warning(f"Anomaly detected: {metric_name} = {value}")
    
    def _update_baseline(self, metric_name: str):
        """Update baseline statistics."""
        if metric_name not in self._metrics_history:
            return
        
        values = list(self._metrics_history[metric_name])
        if len(values) < 10:
            return
        
        mean = np.mean(values)
        std = np.std(values)
        
        self._baselines[metric_name] = {
            "mean": mean,
            "std": std,
            "min": mean - self.threshold * std,
            "max": mean + self.threshold * std
        }
    
    def detect_anomaly(self, metric_name: str, value: float) -> Optional[Anomaly]:
        """Detect anomaly in metric value."""
        if metric_name not in self._baselines:
            return None
        
        baseline = self._baselines[metric_name]
        
        # Calculate Z-score
        if baseline["std"] == 0:
            return None
        
        z_score = abs((value - baseline["mean"]) / baseline["std"])
        
        if z_score > self.threshold:
            # Determine severity
            if z_score > 5.0:
                severity = "critical"
            elif z_score > 4.0:
                severity = "high"
            elif z_score > 3.0:
                severity = "medium"
            else:
                severity = "low"
            
            from datetime import datetime
            return Anomaly(
                metric=metric_name,
                value=value,
                expected_range=(baseline["min"], baseline["max"]),
                severity=severity,
                timestamp=datetime.now().isoformat()
            )
        
        return None
    
    def get_anomalies(self, severity: Optional[str] = None, limit: int = 100) -> List[Anomaly]:
        """Get detected anomalies."""
        anomalies = self._anomalies
        
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        
        return anomalies[-limit:]
    
    def get_anomaly_stats(self) -> Dict[str, Any]:
        """Get anomaly statistics."""
        total = len(self._anomalies)
        by_severity = {}
        
        for severity in ["low", "medium", "high", "critical"]:
            count = sum(1 for a in self._anomalies if a.severity == severity)
            by_severity[severity] = count
        
        return {
            "total_anomalies": total,
            "by_severity": by_severity,
            "recent_anomalies": len([a for a in self._anomalies[-100:]])
        }

