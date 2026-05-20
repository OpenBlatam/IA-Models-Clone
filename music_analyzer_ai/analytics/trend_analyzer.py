"""
Trend Analysis System
Analyze trends in music data over time
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Analyze trends in music data
    """
    
    def __init__(self, window_days: int = 30):
        self.window_days = window_days
        self.data_points: List[Dict[str, Any]] = []
    
    def add_data_point(
        self,
        timestamp: datetime,
        metrics: Dict[str, float]
    ):
        """Add data point for trend analysis"""
        self.data_points.append({
            "timestamp": timestamp,
            "metrics": metrics
        })
    
    def analyze_trends(
        self,
        metric_name: str,
        window_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze trends for a specific metric"""
        window = window_days or self.window_days
        cutoff_date = datetime.now() - timedelta(days=window)
        
        # Filter data points
        recent_data = [
            dp for dp in self.data_points
            if dp["timestamp"] >= cutoff_date and metric_name in dp["metrics"]
        ]
        
        if len(recent_data) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Extract values
        values = [dp["metrics"][metric_name] for dp in recent_data]
        timestamps = [dp["timestamp"] for dp in recent_data]
        
        # Calculate trend
        trend_direction = self._calculate_trend(values)
        trend_strength = self._calculate_trend_strength(values)
        
        # Calculate statistics
        mean_value = np.mean(values)
        std_value = np.std(values)
        min_value = np.min(values)
        max_value = np.max(values)
        
        # Predict next value (simple linear regression)
        predicted_value = self._predict_next_value(values)
        
        return {
            "metric": metric_name,
            "trend_direction": trend_direction,  # "increasing", "decreasing", "stable"
            "trend_strength": trend_strength,  # 0-1
            "mean": float(mean_value),
            "std": float(std_value),
            "min": float(min_value),
            "max": float(max_value),
            "predicted_next": float(predicted_value),
            "data_points": len(recent_data),
            "window_days": window
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate trend strength (0-1)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize to 0-1
        max_slope = np.std(values) * 2
        strength = min(abs(slope) / max_slope if max_slope > 0 else 0, 1.0)
        
        return float(strength)
    
    def _predict_next_value(self, values: List[float]) -> float:
        """Predict next value using linear regression"""
        if len(values) < 2:
            return values[-1] if values else 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        next_x = len(values)
        predicted = np.polyval(coeffs, next_x)
        
        return float(predicted)
    
    def compare_periods(
        self,
        metric_name: str,
        period1_days: int = 7,
        period2_days: int = 7
    ) -> Dict[str, Any]:
        """Compare metrics between two periods"""
        now = datetime.now()
        period1_end = now - timedelta(days=period1_days)
        period1_start = period1_end - timedelta(days=period1_days)
        
        period2_end = period1_start
        period2_start = period2_end - timedelta(days=period2_days)
        
        # Get data for each period
        period1_data = [
            dp["metrics"][metric_name]
            for dp in self.data_points
            if period1_start <= dp["timestamp"] <= period1_end
            and metric_name in dp["metrics"]
        ]
        
        period2_data = [
            dp["metrics"][metric_name]
            for dp in self.data_points
            if period2_start <= dp["timestamp"] <= period2_end
            and metric_name in dp["metrics"]
        ]
        
        if not period1_data or not period2_data:
            return {"error": "Insufficient data for comparison"}
        
        period1_mean = np.mean(period1_data)
        period2_mean = np.mean(period2_data)
        
        change = ((period1_mean - period2_mean) / period2_mean * 100) if period2_mean > 0 else 0
        
        return {
            "metric": metric_name,
            "period1": {
                "mean": float(period1_mean),
                "samples": len(period1_data)
            },
            "period2": {
                "mean": float(period2_mean),
                "samples": len(period2_data)
            },
            "change_percent": float(change),
            "direction": "increase" if change > 0 else "decrease" if change < 0 else "stable"
        }

