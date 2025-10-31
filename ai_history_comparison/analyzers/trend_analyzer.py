"""
Trend Analyzer

This module provides trend analysis capabilities including trend detection,
future prediction, and anomaly detection in time series data.
"""

import statistics
import math
from typing import Dict, List, Any, Optional
import logging

from ..core.base import BaseAnalyzer
from ..core.config import SystemConfig
from ..core.interfaces import ITrendAnalyzer
from ..core.exceptions import AnalysisError, ValidationError

logger = logging.getLogger(__name__)


class TrendAnalyzer(BaseAnalyzer[List[Dict[str, Any]]], ITrendAnalyzer):
    """Advanced trend analyzer for time series data"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._analysis_metrics = [
            "trend_direction",
            "trend_strength",
            "confidence",
            "slope",
            "r_squared",
            "anomalies",
            "predictions"
        ]
    
    async def _analyze(self, data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Perform trend analysis on the provided data"""
        try:
            metric = kwargs.get("metric", "value")
            analysis_type = kwargs.get("analysis_type", "trend")
            
            if analysis_type == "trend":
                return await self.analyze_trends(data, metric)
            elif analysis_type == "prediction":
                days = kwargs.get("days", 7)
                return await self.predict_future(data, metric, days)
            elif analysis_type == "anomaly":
                return await self.detect_anomalies(data, metric)
            else:
                raise ValidationError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise AnalysisError(f"Trend analysis failed: {str(e)}", analyzer_name="TrendAnalyzer")
    
    async def analyze_trends(self, data: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
        """Analyze trends in the data"""
        try:
            # Extract values and timestamps
            values, timestamps = self._extract_data(data, metric)
            
            if len(values) < 2:
                raise ValidationError("At least 2 data points required for trend analysis")
            
            # Calculate trend statistics
            trend_stats = self._calculate_trend_statistics(values, timestamps)
            
            # Determine trend direction and strength
            trend_direction = self._determine_trend_direction(trend_stats["slope"])
            trend_strength = self._calculate_trend_strength(trend_stats["r_squared"])
            confidence = self._calculate_confidence(trend_stats["r_squared"], len(values))
            
            return {
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "confidence": confidence,
                "slope": trend_stats["slope"],
                "r_squared": trend_stats["r_squared"],
                "intercept": trend_stats["intercept"],
                "data_points": len(values),
                "time_span": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
                "statistics": trend_stats
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise AnalysisError(f"Trend analysis failed: {str(e)}", analyzer_name="TrendAnalyzer")
    
    async def predict_future(self, data: List[Dict[str, Any]], metric: str, days: int = 7) -> Dict[str, Any]:
        """Predict future values based on historical data"""
        try:
            # Extract values and timestamps
            values, timestamps = self._extract_data(data, metric)
            
            if len(values) < 3:
                raise ValidationError("At least 3 data points required for prediction")
            
            # Calculate trend statistics
            trend_stats = self._calculate_trend_statistics(values, timestamps)
            
            # Generate predictions
            predictions = []
            confidence_intervals = []
            
            last_timestamp = timestamps[-1]
            time_step = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) > 1 else 1
            
            for i in range(1, days + 1):
                future_timestamp = last_timestamp + (i * time_step)
                predicted_value = trend_stats["intercept"] + trend_stats["slope"] * future_timestamp
                
                # Calculate confidence interval
                confidence_interval = self._calculate_prediction_confidence(
                    trend_stats, future_timestamp, len(values)
                )
                
                predictions.append({
                    "timestamp": future_timestamp,
                    "value": predicted_value,
                    "day": i
                })
                
                confidence_intervals.append({
                    "timestamp": future_timestamp,
                    "lower_bound": predicted_value - confidence_interval,
                    "upper_bound": predicted_value + confidence_interval,
                    "day": i
                })
            
            return {
                "predictions": predictions,
                "confidence_intervals": confidence_intervals,
                "model_info": {
                    "slope": trend_stats["slope"],
                    "intercept": trend_stats["intercept"],
                    "r_squared": trend_stats["r_squared"],
                    "data_points": len(values)
                },
                "prediction_days": days
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise AnalysisError(f"Prediction failed: {str(e)}", analyzer_name="TrendAnalyzer")
    
    async def detect_anomalies(self, data: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
        """Detect anomalies in the data"""
        try:
            # Extract values and timestamps
            values, timestamps = self._extract_data(data, metric)
            
            if len(values) < 5:
                raise ValidationError("At least 5 data points required for anomaly detection")
            
            # Calculate statistical measures
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            
            # Detect anomalies using z-score method
            anomalies = []
            threshold = 2.0  # 2 standard deviations
            
            for i, (value, timestamp) in enumerate(zip(values, timestamps)):
                if std_dev > 0:
                    z_score = abs((value - mean) / std_dev)
                    if z_score > threshold:
                        anomalies.append({
                            "index": i,
                            "timestamp": timestamp,
                            "value": value,
                            "z_score": z_score,
                            "severity": self._get_anomaly_severity(z_score),
                            "deviation": value - mean
                        })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise AnalysisError(f"Anomaly detection failed: {str(e)}", analyzer_name="TrendAnalyzer")
    
    def _extract_data(self, data: List[Dict[str, Any]], metric: str) -> tuple[List[float], List[float]]:
        """Extract values and timestamps from data"""
        values = []
        timestamps = []
        
        for item in data:
            if metric in item:
                try:
                    value = float(item[metric])
                    timestamp = float(item.get("timestamp", len(values)))
                    values.append(value)
                    timestamps.append(timestamp)
                except (ValueError, TypeError):
                    continue
        
        return values, timestamps
    
    def _calculate_trend_statistics(self, values: List[float], timestamps: List[float]) -> Dict[str, float]:
        """Calculate linear regression statistics"""
        n = len(values)
        if n < 2:
            return {"slope": 0, "intercept": 0, "r_squared": 0}
        
        # Calculate means
        x_mean = statistics.mean(timestamps)
        y_mean = statistics.mean(values)
        
        # Calculate slope and intercept
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(timestamps, values))
        denominator = sum((x - x_mean) ** 2 for x in timestamps)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = [intercept + slope * x for x in timestamps]
        ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(values))
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared
        }
    
    def _determine_trend_direction(self, slope: float) -> str:
        """Determine trend direction from slope"""
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_trend_strength(self, r_squared: float) -> float:
        """Calculate trend strength from R-squared"""
        return min(1.0, max(0.0, r_squared))
    
    def _calculate_confidence(self, r_squared: float, n: int) -> float:
        """Calculate confidence level"""
        # Simple confidence calculation based on R-squared and sample size
        base_confidence = r_squared
        size_factor = min(1.0, n / 20.0)  # More data points = higher confidence
        return min(1.0, base_confidence * size_factor)
    
    def _calculate_prediction_confidence(self, trend_stats: Dict[str, float], 
                                       future_timestamp: float, n: int) -> float:
        """Calculate confidence interval for predictions"""
        # Simple confidence interval calculation
        base_uncertainty = 1.0 - trend_stats["r_squared"]
        time_factor = 1.0 + (future_timestamp / 1000.0)  # Uncertainty increases with time
        size_factor = max(0.5, 1.0 - (n / 50.0))  # More data = less uncertainty
        
        return base_uncertainty * time_factor * size_factor
    
    def _get_anomaly_severity(self, z_score: float) -> str:
        """Get anomaly severity level"""
        if z_score >= 4.0:
            return "critical"
        elif z_score >= 3.0:
            return "high"
        elif z_score >= 2.5:
            return "medium"
        else:
            return "low"
    
    def get_analysis_metrics(self) -> List[str]:
        """Get list of metrics this analyzer produces"""
        return self._analysis_metrics
    
    def validate_input(self, data: List[Dict[str, Any]]) -> bool:
        """Validate input data before analysis"""
        if not isinstance(data, list):
            return False
        if len(data) == 0:
            return False
        if len(data) > 10000:  # Reasonable limit
            return False
        
        # Check that all items are dictionaries
        for item in data:
            if not isinstance(item, dict):
                return False
        
        return True





















