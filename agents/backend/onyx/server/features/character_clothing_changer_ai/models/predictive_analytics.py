"""
Predictive Analytics for Flux2 Clothing Changer
===============================================

Predictive analytics and forecasting.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Prediction result."""
    metric: str
    predicted_value: float
    confidence: float
    timestamp: float
    time_horizon: str  # "1h", "24h", "7d", "30d"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PredictiveAnalytics:
    """Predictive analytics and forecasting system."""
    
    def __init__(
        self,
        history_size: int = 10000,
    ):
        """
        Initialize predictive analytics.
        
        Args:
            history_size: Maximum number of data points
        """
        self.history_size = history_size
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a metric value.
        
        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.metric_history[metric_name].append({
            "value": value,
            "timestamp": timestamp,
        })
    
    def predict(
        self,
        metric_name: str,
        time_horizon: str = "24h",
    ) -> Optional[Prediction]:
        """
        Predict future value of metric.
        
        Args:
            metric_name: Metric name
            time_horizon: Time horizon ("1h", "24h", "7d", "30d")
            
        Returns:
            Prediction or None
        """
        if metric_name not in self.metric_history:
            return None
        
        history = list(self.metric_history[metric_name])
        if len(history) < 10:
            return None
        
        # Convert time horizon to seconds
        horizon_seconds = self._parse_time_horizon(time_horizon)
        
        # Extract values and timestamps
        values = [d["value"] for d in history]
        timestamps = [d["timestamp"] for d in history]
        
        # Simple linear regression for prediction
        predicted_value, confidence = self._linear_regression_predict(
            values,
            timestamps,
            horizon_seconds,
        )
        
        return Prediction(
            metric=metric_name,
            predicted_value=predicted_value,
            confidence=confidence,
            timestamp=time.time() + horizon_seconds,
            time_horizon=time_horizon,
            metadata={
                "method": "linear_regression",
                "data_points": len(history),
            },
        )
    
    def predict_trend(
        self,
        metric_name: str,
        window_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Predict trend direction.
        
        Args:
            metric_name: Metric name
            window_size: Window size for analysis
            
        Returns:
            Trend prediction
        """
        if metric_name not in self.metric_history:
            return {"trend": "insufficient_data"}
        
        history = list(self.metric_history[metric_name])[-window_size:]
        if len(history) < 5:
            return {"trend": "insufficient_data"}
        
        values = [d["value"] for d in history]
        
        # Calculate trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Determine trend
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # Calculate confidence
        variance = np.var(values)
        mean_value = np.mean(values)
        confidence = max(0.0, min(1.0, 1.0 - (variance / (mean_value ** 2 + 1))))
        
        return {
            "trend": trend,
            "slope": float(slope),
            "confidence": confidence,
            "current_value": float(values[-1]),
            "predicted_change": float(slope * window_size),
        }
    
    def forecast_demand(
        self,
        metric_name: str = "requests",
        days: int = 7,
    ) -> List[Prediction]:
        """
        Forecast demand for multiple days.
        
        Args:
            metric_name: Metric name
            days: Number of days to forecast
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for day in range(1, days + 1):
            prediction = self.predict(metric_name, time_horizon=f"{day}d")
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    def detect_anomaly(
        self,
        metric_name: str,
        value: float,
        threshold: float = 2.0,
    ) -> bool:
        """
        Detect if value is anomalous.
        
        Args:
            metric_name: Metric name
            value: Current value
            threshold: Standard deviations threshold
            
        Returns:
            True if anomalous
        """
        if metric_name not in self.metric_history:
            return False
        
        history = list(self.metric_history[metric_name])
        if len(history) < 10:
            return False
        
        values = [d["value"] for d in history]
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return False
        
        z_score = abs((value - mean) / std)
        return z_score > threshold
    
    def _linear_regression_predict(
        self,
        values: List[float],
        timestamps: List[float],
        horizon_seconds: float,
    ) -> Tuple[float, float]:
        """Predict using linear regression."""
        if len(values) < 2:
            return values[-1] if values else 0.0, 0.0
        
        # Normalize timestamps
        base_time = timestamps[0]
        normalized_times = [(t - base_time) / 3600.0 for t in timestamps]  # Hours
        
        # Linear regression
        x = np.array(normalized_times)
        y = np.array(values)
        
        # Calculate slope and intercept
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
        intercept = np.mean(y) - slope * np.mean(x)
        
        # Predict
        future_time = (timestamps[-1] + horizon_seconds - base_time) / 3600.0
        predicted = slope * future_time + intercept
        
        # Calculate confidence (based on R-squared)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        confidence = max(0.0, min(1.0, r_squared))
        
        return float(predicted), float(confidence)
    
    def _parse_time_horizon(self, time_horizon: str) -> float:
        """Parse time horizon string to seconds."""
        if time_horizon.endswith("h"):
            hours = int(time_horizon[:-1])
            return hours * 3600.0
        elif time_horizon.endswith("d"):
            days = int(time_horizon[:-1])
            return days * 86400.0
        elif time_horizon.endswith("m"):
            minutes = int(time_horizon[:-1])
            return minutes * 60.0
        else:
            return 86400.0  # Default 24 hours
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictive analytics statistics."""
        return {
            "tracked_metrics": list(self.metric_history.keys()),
            "data_points": {
                metric: len(history)
                for metric, history in self.metric_history.items()
            },
        }


