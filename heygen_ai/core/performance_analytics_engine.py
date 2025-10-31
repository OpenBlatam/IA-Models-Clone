"""
Performance Analytics Engine for HeyGen AI Enterprise

This module provides advanced performance analytics capabilities:
- Performance trend analysis and forecasting
- Anomaly detection and root cause analysis
- Performance optimization recommendations
- Real-time performance scoring
- Historical performance tracking
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsConfig:
    """Configuration for performance analytics engine."""
    
    # Analysis settings
    trend_window_size: int = 100
    anomaly_threshold: float = 2.0
    forecast_horizon: int = 10
    confidence_level: float = 0.95
    
    # Scoring settings
    performance_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_weights is None:
            self.performance_weights = {
                "inference_time": 0.3,
                "memory_usage": 0.25,
                "gpu_utilization": 0.25,
                "throughput": 0.2
            }


class PerformanceAnalyzer:
    """Advanced performance analysis and insights engine."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.trend_window_size)
        self.anomaly_history = deque(maxlen=1000)
        self.trend_models = {}
        
    def analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics and generate insights."""
        try:
            # Add timestamp and store in history
            metrics["timestamp"] = time.time()
            self.performance_history.append(metrics)
            
            # Perform analysis
            analysis = {
                "timestamp": metrics["timestamp"],
                "current_metrics": metrics,
                "trends": self._analyze_trends(),
                "anomalies": self._detect_anomalies(metrics),
                "performance_score": self._calculate_performance_score(metrics),
                "recommendations": self._generate_recommendations(metrics),
                "forecast": self._generate_forecast()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            if len(self.performance_history) < 3:
                return {"status": "insufficient_data"}
            
            trends = {}
            
            # Analyze each metric
            for metric_name in self.config.performance_weights.keys():
                if metric_name in self.performance_history[0]:
                    values = [h.get(metric_name, 0) for h in self.performance_history]
                    trends[metric_name] = self._calculate_trend(values)
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend for a series of values."""
        try:
            if len(values) < 2:
                return {"trend": "stable", "slope": 0, "confidence": 0}
            
            # Calculate slope using linear regression
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Calculate trend direction
            if abs(slope) < 0.01:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            
            # Calculate confidence based on R-squared
            y_pred = slope * x + intercept
            r_squared = 1 - (np.sum((np.array(values) - y_pred) ** 2) / 
                            np.sum((np.array(values) - np.mean(values)) ** 2))
            
            return {
                "trend": trend,
                "slope": float(slope),
                "confidence": max(0, min(1, r_squared)),
                "change_rate": float(slope / max(np.mean(values), 1))
            }
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {e}")
            return {"trend": "unknown", "slope": 0, "confidence": 0}
    
    def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        try:
            anomalies = []
            
            for metric_name, value in metrics.items():
                if metric_name in self.config.performance_weights:
                    anomaly = self._check_metric_anomaly(metric_name, value)
                    if anomaly:
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def _check_metric_anomaly(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """Check if a metric value is anomalous."""
        try:
            if len(self.performance_history) < 5:
                return None
            
            # Get historical values for this metric
            values = [h.get(metric_name, 0) for h in self.performance_history if metric_name in h]
            
            if len(values) < 3:
                return None
            
            # Calculate statistics
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if std_val == 0:
                return None
            
            # Check if current value is anomalous
            z_score = abs(value - mean_val) / std_val
            
            if z_score > self.config.anomaly_threshold:
                return {
                    "metric": metric_name,
                    "value": value,
                    "expected_range": [mean_val - 2*std_val, mean_val + 2*std_val],
                    "z_score": z_score,
                    "severity": "high" if z_score > 3 else "medium",
                    "timestamp": time.time()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Metric anomaly check failed: {e}")
            return None
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance score."""
        try:
            scores = {}
            total_score = 0
            total_weight = 0
            
            for metric_name, weight in self.config.performance_weights.items():
                if metric_name in metrics:
                    # Normalize metric value (0-100 scale)
                    normalized_score = self._normalize_metric(metric_name, metrics[metric_name])
                    scores[metric_name] = normalized_score
                    total_score += normalized_score * weight
                    total_weight += weight
            
            overall_score = total_score / max(total_weight, 1)
            
            return {
                "overall_score": overall_score,
                "metric_scores": scores,
                "grade": self._get_performance_grade(overall_score),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Performance score calculation failed: {e}")
            return {"overall_score": 0, "grade": "F"}
    
    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric value to 0-100 scale."""
        try:
            # Define normalization ranges for different metrics
            ranges = {
                "inference_time": (0, 1000),      # 0-1000ms
                "memory_usage": (0, 10000),       # 0-10GB
                "gpu_utilization": (0, 100),      # 0-100%
                "throughput": (0, 1000)           # 0-1000 samples/sec
            }
            
            if metric_name in ranges:
                min_val, max_val = ranges[metric_name]
                normalized = max(0, min(100, (value - min_val) / (max_val - min_val) * 100))
                
                # Invert scores for metrics where lower is better
                if metric_name in ["inference_time", "memory_usage"]:
                    normalized = 100 - normalized
                
                return normalized
            
            return 50  # Default neutral score
            
        except Exception as e:
            logger.error(f"Metric normalization failed: {e}")
            return 50
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert performance score to letter grade."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        try:
            recommendations = []
            analysis = self._analyze_trends()
            
            # Check for performance degradation
            for metric_name, trend_info in analysis.items():
                if isinstance(trend_info, dict) and "trend" in trend_info:
                    if trend_info["trend"] == "increasing" and metric_name in ["inference_time", "memory_usage"]:
                        recommendations.append(f"Performance degradation detected in {metric_name}. Consider optimization.")
                    
                    if trend_info["trend"] == "decreasing" and metric_name in ["gpu_utilization", "throughput"]:
                        recommendations.append(f"Performance degradation detected in {metric_name}. Investigate bottlenecks.")
            
            # Check current metric values
            if metrics.get("gpu_utilization", 0) < 50:
                recommendations.append("Low GPU utilization detected. Consider batch size optimization or model parallelization.")
            
            if metrics.get("memory_usage", 0) > 8000:  # 8GB
                recommendations.append("High memory usage detected. Consider model compression or memory optimization.")
            
            if metrics.get("inference_time", 0) > 500:  # 500ms
                recommendations.append("High inference time detected. Consider quantization or model optimization.")
            
            if not recommendations:
                recommendations.append("Performance is within optimal ranges. Continue monitoring.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations due to error"]
    
    def _generate_forecast(self) -> Dict[str, Any]:
        """Generate performance forecasts."""
        try:
            if len(self.performance_history) < 10:
                return {"status": "insufficient_data"}
            
            forecasts = {}
            
            for metric_name in self.config.performance_weights.keys():
                values = [h.get(metric_name, 0) for h in self.performance_history if metric_name in h]
                
                if len(values) >= 5:
                    forecast = self._forecast_metric(values)
                    if forecast:
                        forecasts[metric_name] = forecast
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return {"error": str(e)}
    
    def _forecast_metric(self, values: List[float]) -> Optional[Dict[str, Any]]:
        """Forecast future values for a metric."""
        try:
            if len(values) < 5:
                return None
            
            # Simple linear forecasting
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Generate forecast
            forecast_values = []
            for i in range(1, self.config.forecast_horizon + 1):
                forecast_val = slope * (len(values) + i) + intercept
                forecast_values.append(max(0, forecast_val))
            
            return {
                "forecast_values": forecast_values,
                "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "confidence": 0.7,  # Simplified confidence
                "horizon": self.config.forecast_horizon
            }
            
        except Exception as e:
            logger.error(f"Metric forecasting failed: {e}")
            return None
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        try:
            if not self.performance_history:
                return {"status": "no_data"}
            
            latest_metrics = self.performance_history[-1]
            
            summary = {
                "timestamp": time.time(),
                "total_measurements": len(self.performance_history),
                "latest_analysis": self.analyze_performance(latest_metrics),
                "anomaly_count": len(self.anomaly_history),
                "trend_summary": self._get_trend_summary()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Analytics summary generation failed: {e}")
            return {"error": str(e)}
    
    def _get_trend_summary(self) -> Dict[str, Any]:
        """Get summary of all trends."""
        try:
            trends = self._analyze_trends()
            
            if "error" in trends:
                return trends
            
            trend_summary = {
                "improving_metrics": [],
                "degrading_metrics": [],
                "stable_metrics": []
            }
            
            for metric_name, trend_info in trends.items():
                if isinstance(trend_info, dict) and "trend" in trend_info:
                    trend = trend_info["trend"]
                    if trend == "decreasing" and metric_name in ["inference_time", "memory_usage"]:
                        trend_summary["improving_metrics"].append(metric_name)
                    elif trend == "increasing" and metric_name in ["inference_time", "memory_usage"]:
                        trend_summary["degrading_metrics"].append(metric_name)
                    elif trend == "stable":
                        trend_summary["stable_metrics"].append(metric_name)
            
            return trend_summary
            
        except Exception as e:
            logger.error(f"Trend summary generation failed: {e}")
            return {"error": str(e)}


# Factory functions
def create_performance_analyzer(config: Optional[AnalyticsConfig] = None) -> PerformanceAnalyzer:
    """Create a performance analyzer."""
    if config is None:
        config = AnalyticsConfig()
    
    return PerformanceAnalyzer(config)


def create_analytics_config_for_performance() -> AnalyticsConfig:
    """Create analytics configuration optimized for performance monitoring."""
    return AnalyticsConfig(
        trend_window_size=200,
        anomaly_threshold=2.5,
        forecast_horizon=20,
        confidence_level=0.99
    )


def create_analytics_config_for_accuracy() -> AnalyticsConfig:
    """Create analytics configuration optimized for accuracy."""
    return AnalyticsConfig(
        trend_window_size=500,
        anomaly_threshold=1.5,
        forecast_horizon=10,
        confidence_level=0.999
    )


if __name__ == "__main__":
    # Test the performance analytics engine
    config = create_analytics_config_for_performance()
    analyzer = create_performance_analyzer(config)
    
    # Simulate some performance data
    test_metrics = [
        {"inference_time": 100, "memory_usage": 2000, "gpu_utilization": 80, "throughput": 100},
        {"inference_time": 95, "memory_usage": 2100, "gpu_utilization": 85, "throughput": 105},
        {"inference_time": 110, "memory_usage": 2200, "gpu_utilization": 75, "throughput": 95},
        {"inference_time": 105, "memory_usage": 2300, "gpu_utilization": 70, "throughput": 90},
        {"inference_time": 120, "memory_usage": 2400, "gpu_utilization": 65, "throughput": 85}
    ]
    
    # Analyze each set of metrics
    for i, metrics in enumerate(test_metrics):
        print(f"\n--- Analysis {i+1} ---")
        analysis = analyzer.analyze_performance(metrics)
        
        if "error" not in analysis:
            print(f"Performance Score: {analysis['performance_score']['overall_score']:.1f} ({analysis['performance_score']['grade']})")
            print(f"Recommendations: {analysis['recommendations']}")
            
            if analysis['anomalies']:
                print(f"Anomalies detected: {len(analysis['anomalies'])}")
        else:
            print(f"Analysis failed: {analysis['error']}")
    
    # Get final summary
    print(f"\n--- Final Analytics Summary ---")
    summary = analyzer.get_analytics_summary()
    print(f"Total measurements: {summary.get('total_measurements', 0)}")
    print(f"Anomaly count: {summary.get('anomaly_count', 0)}")
    
    print("Performance analytics engine test completed")
