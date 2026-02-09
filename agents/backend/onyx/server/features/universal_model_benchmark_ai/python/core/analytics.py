"""
Analytics Module - Enhanced analytics and insights.

Provides:
- Trend analysis with multiple methods
- Performance predictions with confidence intervals
- Anomaly detection with multiple algorithms
- Statistical analysis and correlations
- Model comparison and ranking
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrendMethod(str, Enum):
    """Trend analysis methods."""
    LINEAR = "linear"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"


class AnomalyMethod(str, Enum):
    """Anomaly detection methods."""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"  # Requires scikit-learn


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    metric: str
    trend: str  # "increasing", "decreasing", "stable"
    change_percentage: float
    confidence: float
    data_points: int
    method: str = "linear"
    slope: float = 0.0
    r_squared: float = 0.0


@dataclass
class Anomaly:
    """Anomaly detection result."""
    result_id: str
    metric: str
    value: float
    expected_range: Tuple[float, float]
    severity: str  # "low", "medium", "high"
    description: str
    z_score: Optional[float] = None
    percentile: Optional[float] = None


@dataclass
class Prediction:
    """Performance prediction result."""
    metric: str
    predicted_value: float
    confidence: float
    confidence_interval: Tuple[float, float]
    method: str
    historical_points: int


class AnalyticsEngine:
    """
    Advanced analytics engine for benchmark results.
    
    Features:
    - Multiple trend analysis methods
    - Anomaly detection with various algorithms
    - Performance predictions with confidence intervals
    - Statistical analysis and correlations
    """
    
    def __init__(self):
        """Initialize analytics engine."""
        pass
    
    def analyze_trends(
        self,
        results: List[Any],  # BenchmarkResult objects
        metric: str = "accuracy",
        time_window: Optional[int] = None,
        method: TrendMethod = TrendMethod.LINEAR,
    ) -> TrendAnalysis:
        """
        Analyze trends in results over time with multiple methods.
        
        Args:
            results: List of benchmark results
            metric: Metric to analyze (accuracy, throughput, latency_p50)
            time_window: Optional time window in days
            method: Trend analysis method
        
        Returns:
            TrendAnalysis object
        """
        if not results:
            return TrendAnalysis(
                metric=metric,
                trend="stable",
                change_percentage=0.0,
                confidence=0.0,
                data_points=0,
                method=method.value,
            )
        
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - timedelta(days=time_window)
            results = [
                r for r in results
                if hasattr(r, 'timestamp') and datetime.fromisoformat(r.timestamp) >= cutoff
            ]
        
        if len(results) < 2:
            return TrendAnalysis(
                metric=metric,
                trend="stable",
                change_percentage=0.0,
                confidence=0.0,
                data_points=len(results),
                method=method.value,
            )
        
        # Sort by timestamp if available
        if hasattr(results[0], 'timestamp'):
            results = sorted(results, key=lambda r: r.timestamp)
        
        # Extract metric values
        values = [getattr(r, metric, 0.0) for r in results]
        
        if method == TrendMethod.LINEAR:
            return self._linear_trend(values, metric, len(results))
        elif method == TrendMethod.MOVING_AVERAGE:
            return self._moving_average_trend(values, metric, len(results))
        elif method == TrendMethod.EXPONENTIAL:
            return self._exponential_trend(values, metric, len(results))
        else:
            return self._linear_trend(values, metric, len(results))
    
    def _linear_trend(
        self,
        values: List[float],
        metric: str,
        data_points: int,
    ) -> TrendAnalysis:
        """Calculate linear trend."""
        if len(values) < 2:
            return TrendAnalysis(
                metric=metric,
                trend="stable",
                change_percentage=0.0,
                confidence=0.0,
                data_points=data_points,
                method="linear",
            )
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        # Calculate change percentage
        if values[0] == 0:
            change_pct = 0.0
        else:
            change_pct = ((values[-1] - values[0]) / values[0]) * 100.0
        
        # Determine trend
        if abs(change_pct) < 5.0:
            trend = "stable"
        elif change_pct > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # Calculate R-squared
        y_pred = [y_mean + slope * (x[i] - x_mean) for i in range(n)]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Confidence based on R-squared and data points
        confidence = min(1.0, (len(values) / 20.0) * r_squared)
        
        return TrendAnalysis(
            metric=metric,
            trend=trend,
            change_percentage=change_pct,
            confidence=confidence,
            data_points=data_points,
            method="linear",
            slope=slope,
            r_squared=r_squared,
        )
    
    def _moving_average_trend(
        self,
        values: List[float],
        metric: str,
        data_points: int,
    ) -> TrendAnalysis:
        """Calculate trend using moving average."""
        if len(values) < 4:
            return self._linear_trend(values, metric, data_points)
        
        # Use first and second half averages
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if first_avg == 0:
            change_pct = 0.0
        else:
            change_pct = ((second_avg - first_avg) / first_avg) * 100.0
        
        # Determine trend
        if abs(change_pct) < 5.0:
            trend = "stable"
        elif change_pct > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # Confidence based on variance
        variance = statistics.variance(values) if len(values) > 1 else 0.0
        confidence = min(1.0, len(values) / 20.0) * (1.0 - min(1.0, variance))
        
        return TrendAnalysis(
            metric=metric,
            trend=trend,
            change_percentage=change_pct,
            confidence=confidence,
            data_points=data_points,
            method="moving_average",
        )
    
    def _exponential_trend(
        self,
        values: List[float],
        metric: str,
        data_points: int,
    ) -> TrendAnalysis:
        """Calculate exponential trend."""
        # For simplicity, use linear trend on log values
        if any(v <= 0 for v in values):
            return self._linear_trend(values, metric, data_points)
        
        log_values = [statistics.log(v) for v in values]
        linear_result = self._linear_trend(log_values, metric, data_points)
        
        # Convert back to percentage change
        change_pct = (statistics.exp(linear_result.change_percentage / 100.0) - 1.0) * 100.0
        
        return TrendAnalysis(
            metric=metric,
            trend=linear_result.trend,
            change_percentage=change_pct,
            confidence=linear_result.confidence,
            data_points=data_points,
            method="exponential",
            slope=linear_result.slope,
            r_squared=linear_result.r_squared,
        )
    
    def detect_anomalies(
        self,
        results: List[Any],
        metric: str = "accuracy",
        threshold_std: float = 2.0,
        method: AnomalyMethod = AnomalyMethod.Z_SCORE,
    ) -> List[Anomaly]:
        """
        Detect anomalies in results using multiple methods.
        
        Args:
            results: List of benchmark results
            metric: Metric to analyze
            threshold_std: Standard deviation threshold (for Z-score)
            method: Anomaly detection method
        
        Returns:
            List of detected anomalies
        """
        if len(results) < 3:
            return []
        
        values = [getattr(r, metric, 0.0) for r in results]
        
        if method == AnomalyMethod.Z_SCORE:
            return self._z_score_anomalies(results, values, metric, threshold_std)
        elif method == AnomalyMethod.IQR:
            return self._iqr_anomalies(results, values, metric)
        else:
            return self._z_score_anomalies(results, values, metric, threshold_std)
    
    def _z_score_anomalies(
        self,
        results: List[Any],
        values: List[float],
        metric: str,
        threshold: float,
    ) -> List[Anomaly]:
        """Detect anomalies using Z-score method."""
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        if std_dev == 0:
            return []
        
        anomalies = []
        lower_bound = mean - (threshold * std_dev)
        upper_bound = mean + (threshold * std_dev)
        
        for result, value in zip(results, values):
            z_score = (value - mean) / std_dev if std_dev > 0 else 0.0
            
            if abs(z_score) > threshold:
                severity = "high" if abs(z_score) > 3.0 else "medium"
                
                # Calculate percentile
                sorted_values = sorted(values)
                percentile = (sorted_values.index(value) / len(sorted_values)) * 100.0
                
                result_id = f"{getattr(result, 'model_name', 'unknown')}_{getattr(result, 'benchmark_name', 'unknown')}"
                
                anomaly = Anomaly(
                    result_id=result_id,
                    metric=metric,
                    value=value,
                    expected_range=(lower_bound, upper_bound),
                    severity=severity,
                    description=(
                        f"{metric} value {value:.3f} is {abs(z_score):.2f} standard deviations "
                        f"from mean ({mean:.3f})"
                    ),
                    z_score=z_score,
                    percentile=percentile,
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _iqr_anomalies(
        self,
        results: List[Any],
        values: List[float],
        metric: str,
    ) -> List[Anomaly]:
        """Detect anomalies using IQR (Interquartile Range) method."""
        if len(values) < 4:
            return []
        
        sorted_values = sorted(values)
        q1_idx = len(sorted_values) // 4
        q3_idx = 3 * len(sorted_values) // 4
        
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = []
        for result, value in zip(results, values):
            if value < lower_bound or value > upper_bound:
                severity = "high" if value < q1 - 3 * iqr or value > q3 + 3 * iqr else "medium"
                
                percentile = (sorted_values.index(value) / len(sorted_values)) * 100.0
                result_id = f"{getattr(result, 'model_name', 'unknown')}_{getattr(result, 'benchmark_name', 'unknown')}"
                
                anomaly = Anomaly(
                    result_id=result_id,
                    metric=metric,
                    value=value,
                    expected_range=(lower_bound, upper_bound),
                    severity=severity,
                    description=(
                        f"{metric} value {value:.3f} is outside IQR range "
                        f"[{lower_bound:.3f}, {upper_bound:.3f}]"
                    ),
                    percentile=percentile,
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def predict_performance(
        self,
        historical_results: List[Any],
        model_name: str,
        benchmark_name: str,
        metric: str = "accuracy",
    ) -> Prediction:
        """
        Predict future performance based on historical data.
        
        Args:
            historical_results: Historical benchmark results
            model_name: Model name
            benchmark_name: Benchmark name
            metric: Metric to predict
        
        Returns:
            Prediction object
        """
        # Filter relevant results
        relevant = [
            r for r in historical_results
            if (hasattr(r, 'model_name') and r.model_name == model_name) and
               (hasattr(r, 'benchmark_name') and r.benchmark_name == benchmark_name)
        ]
        
        if len(relevant) < 2:
            return Prediction(
                metric=metric,
                predicted_value=0.0,
                confidence=0.0,
                confidence_interval=(0.0, 0.0),
                method="linear",
                historical_points=len(relevant),
            )
        
        # Sort by timestamp
        if hasattr(relevant[0], 'timestamp'):
            relevant.sort(key=lambda r: r.timestamp)
        
        values = [getattr(r, metric, 0.0) for r in relevant]
        
        # Simple linear regression for prediction
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator > 0 else 0.0
        
        # Predict next value
        predicted_value = y_mean + slope * (n - x_mean)
        
        # Calculate confidence interval
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        margin = 1.96 * std_dev / (n ** 0.5)  # 95% confidence interval
        confidence_interval = (predicted_value - margin, predicted_value + margin)
        
        # Confidence based on data points and variance
        confidence = min(1.0, len(relevant) / 10.0) * (1.0 - min(1.0, std_dev))
        
        return Prediction(
            metric=metric,
            predicted_value=predicted_value,
            confidence=confidence,
            confidence_interval=confidence_interval,
            method="linear",
            historical_points=len(relevant),
        )
    
    def compare_models(
        self,
        results: List[Any],
        benchmark_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive model comparison with statistical analysis.
        
        Args:
            results: List of benchmark results
            benchmark_name: Optional benchmark name to filter
        
        Returns:
            Comparison analysis dictionary
        """
        # Filter by benchmark if specified
        if benchmark_name:
            benchmark_results = [
                r for r in results
                if hasattr(r, 'benchmark_name') and r.benchmark_name == benchmark_name
            ]
        else:
            benchmark_results = results
        
        if not benchmark_results:
            return {}
        
        # Group by model
        by_model = defaultdict(list)
        for result in benchmark_results:
            model_name = getattr(result, 'model_name', 'unknown')
            by_model[model_name].append(result)
        
        # Calculate comprehensive statistics per model
        model_stats = {}
        for model_name, model_results in by_model.items():
            if model_results:
                accuracies = [r.accuracy for r in model_results]
                throughputs = [r.throughput for r in model_results]
                latencies = [r.latency_p50 for r in model_results]
                
                model_stats[model_name] = {
                    "avg_accuracy": statistics.mean(accuracies),
                    "median_accuracy": statistics.median(accuracies),
                    "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                    "min_accuracy": min(accuracies),
                    "max_accuracy": max(accuracies),
                    "avg_throughput": statistics.mean(throughputs),
                    "median_throughput": statistics.median(throughputs),
                    "std_throughput": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
                    "avg_latency": statistics.mean(latencies),
                    "median_latency": statistics.median(latencies),
                    "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                    "runs": len(model_results),
                    "consistency_score": 1.0 - min(1.0, statistics.stdev(accuracies)) if len(accuracies) > 1 else 1.0,
                }
        
        # Find best model using composite score
        if model_stats:
            best_model = max(
                model_stats.items(),
                key=lambda x: (
                    x[1]["avg_accuracy"] * 0.5 +
                    min(x[1]["avg_throughput"] / 1000.0, 1.0) * 0.3 +
                    min(1.0 / (x[1]["avg_latency"] + 0.001), 1.0) * 0.2
                )
            )[0]
        else:
            best_model = None
        
        return {
            "benchmark": benchmark_name or "all",
            "models": model_stats,
            "best_model": best_model,
            "total_comparisons": len(benchmark_results),
            "total_models": len(model_stats),
        }
    
    def calculate_correlations(
        self,
        results: List[Any],
    ) -> Dict[str, float]:
        """
        Calculate correlations between metrics.
        
        Args:
            results: List of benchmark results
        
        Returns:
            Dictionary with correlation coefficients
        """
        if len(results) < 3:
            return {}
        
        accuracies = [r.accuracy for r in results]
        throughputs = [r.throughput for r in results]
        latencies = [r.latency_p50 for r in results]
        
        def correlation(x: List[float], y: List[float]) -> float:
            """Calculate Pearson correlation coefficient."""
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(y)
            
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
            x_std = (sum((x[i] - x_mean) ** 2 for i in range(len(x))) / len(x)) ** 0.5
            y_std = (sum((y[i] - y_mean) ** 2 for i in range(len(y))) / len(y)) ** 0.5
            
            if x_std == 0 or y_std == 0:
                return 0.0
            
            return numerator / (len(x) * x_std * y_std)
        
        return {
            "accuracy_vs_throughput": correlation(accuracies, throughputs),
            "accuracy_vs_latency": correlation(accuracies, latencies),
            "throughput_vs_latency": correlation(throughputs, latencies),
        }


__all__ = [
    "TrendMethod",
    "AnomalyMethod",
    "TrendAnalysis",
    "Anomaly",
    "Prediction",
    "AnalyticsEngine",
]
