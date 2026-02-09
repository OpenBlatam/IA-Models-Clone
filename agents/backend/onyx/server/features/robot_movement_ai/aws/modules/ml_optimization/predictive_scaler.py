"""
Predictive Scaler
=================

ML-based predictive auto-scaling.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PredictiveScaler:
    """Predictive auto-scaler using ML."""
    
    def __init__(self, prediction_window: int = 5):
        self.prediction_window = prediction_window
        self._metrics_history: Dict[str, deque] = {}
        self._predictions: List[Dict[str, Any]] = []
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Record metric for prediction."""
        if metric_name not in self._metrics_history:
            self._metrics_history[metric_name] = deque(maxlen=1000)
        
        self._metrics_history[metric_name].append({
            "value": value,
            "timestamp": timestamp or datetime.now()
        })
    
    def predict_load(self, metric_name: str, minutes_ahead: int = 5) -> Optional[float]:
        """Predict future load using simple linear regression."""
        if metric_name not in self._metrics_history:
            return None
        
        history = list(self._metrics_history[metric_name])
        if len(history) < 10:
            return None
        
        # Simple linear regression
        values = [m["value"] for m in history[-20:]]
        x = np.arange(len(values))
        
        # Calculate trend
        slope = np.polyfit(x, values, 1)[0]
        intercept = np.polyfit(x, values, 1)[1]
        
        # Predict future value
        future_x = len(values) + (minutes_ahead / 1.0)  # Assuming 1 minute intervals
        predicted = slope * future_x + intercept
        
        return max(0, predicted)  # Ensure non-negative
    
    def should_scale_up(self, current_load: float, threshold: float = 0.8) -> bool:
        """Determine if should scale up."""
        predicted = self.predict_load("cpu_usage", minutes_ahead=5)
        
        if predicted is None:
            return current_load > threshold
        
        # Scale up if predicted load exceeds threshold
        return predicted > threshold
    
    def should_scale_down(self, current_load: float, threshold: float = 0.3) -> bool:
        """Determine if should scale down."""
        predicted = self.predict_load("cpu_usage", minutes_ahead=5)
        
        if predicted is None:
            return current_load < threshold
        
        # Scale down if predicted load is low
        return predicted < threshold
    
    def get_scale_recommendation(self) -> Dict[str, Any]:
        """Get scaling recommendation."""
        current_load = self._get_current_load()
        
        if current_load is None:
            return {"action": "maintain", "reason": "insufficient_data"}
        
        if self.should_scale_up(current_load):
            return {
                "action": "scale_up",
                "current_load": current_load,
                "predicted_load": self.predict_load("cpu_usage", minutes_ahead=5),
                "reason": "predicted_high_load"
            }
        
        if self.should_scale_down(current_load):
            return {
                "action": "scale_down",
                "current_load": current_load,
                "predicted_load": self.predict_load("cpu_usage", minutes_ahead=5),
                "reason": "predicted_low_load"
            }
        
        return {"action": "maintain", "current_load": current_load}
    
    def _get_current_load(self) -> Optional[float]:
        """Get current load."""
        if "cpu_usage" not in self._metrics_history:
            return None
        
        history = list(self._metrics_history["cpu_usage"])
        if not history:
            return None
        
        return history[-1]["value"]















