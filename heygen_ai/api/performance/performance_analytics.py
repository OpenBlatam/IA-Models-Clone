from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Tuple
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict, field
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
import json
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from fastapi import Request, Response
from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Performance Analytics for HeyGen AI FastAPI
Advanced analytics and reporting for API performance metrics.
"""



logger = structlog.get_logger()

# =============================================================================
# Analytics Types
# =============================================================================

class AnalysisType(Enum):
    """Analysis type enumeration."""
    TREND = "trend"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"
    FORECAST = "forecast"
    COMPARISON = "comparison"

class TimeWindow(Enum):
    """Time window enumeration."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    metric: str
    time_window: TimeWindow
    start_time: datetime
    end_time: datetime
    data_points: List[Tuple[datetime, float]]
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    slope: float
    r_squared: float
    confidence_interval: Tuple[float, float]

@dataclass
class PerformanceAnomaly:
    """Performance anomaly detection."""
    metric: str
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float
    severity: str  # "low", "medium", "high", "critical"
    confidence: float
    description: str

@dataclass
class PerformanceCorrelation:
    """Performance correlation analysis."""
    metric1: str
    metric2: str
    correlation_coefficient: float
    p_value: float
    significance: str  # "strong", "moderate", "weak", "none"
    relationship: str  # "positive", "negative", "none"

@dataclass
class PerformanceForecast:
    """Performance forecast."""
    metric: str
    forecast_horizon: int  # minutes
    forecast_values: List[Tuple[datetime, float]]
    confidence_intervals: List[Tuple[float, float]]
    model_accuracy: float
    model_type: str

@dataclass
class PerformanceReport:
    """Performance report."""
    report_id: str
    title: str
    generated_at: datetime
    time_period: Tuple[datetime, datetime]
    summary: Dict[str, Any]
    trends: List[PerformanceTrend]
    anomalies: List[PerformanceAnomaly]
    correlations: List[PerformanceCorrelation]
    forecasts: List[PerformanceForecast]
    recommendations: List[str]
    charts: List[Dict[str, Any]]

# =============================================================================
# Performance Analytics
# =============================================================================

class PerformanceAnalytics:
    """Advanced performance analytics system."""
    
    def __init__(self, monitor) -> Any:
        self.monitor = monitor
        self.analysis_cache: Dict[str, Any] = {}
        self.report_templates: Dict[str, Dict[str, Any]] = {}
        self._setup_report_templates()
    
    def _setup_report_templates(self) -> Any:
        """Setup report templates."""
        self.report_templates = {
            "daily": {
                "title": "Daily Performance Report",
                "time_window": TimeWindow.DAY,
                "metrics": ["response_time", "throughput", "error_rate", "cpu_usage", "memory_usage"],
                "analysis_types": [AnalysisType.TREND, AnalysisType.ANOMALY, AnalysisType.CORRELATION]
            },
            "hourly": {
                "title": "Hourly Performance Report",
                "time_window": TimeWindow.HOUR,
                "metrics": ["response_time", "throughput", "error_rate"],
                "analysis_types": [AnalysisType.TREND, AnalysisType.ANOMALY]
            },
            "weekly": {
                "title": "Weekly Performance Report",
                "time_window": TimeWindow.WEEK,
                "metrics": ["response_time", "throughput", "error_rate", "cpu_usage", "memory_usage"],
                "analysis_types": [AnalysisType.TREND, AnalysisType.ANOMALY, AnalysisType.CORRELATION, AnalysisType.FORECAST]
            }
        }
    
    async def analyze_trends(
        self,
        metric: str,
        time_window: TimeWindow,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> PerformanceTrend:
        """Analyze trends for a specific metric."""
        # Get data for the time period
        data = await self._get_metric_data(metric, time_window, start_time, end_time)
        
        if not data:
            return None
        
        # Convert to pandas DataFrame for analysis
        df = pd.DataFrame(data, columns=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate trend using linear regression
        x = np.arange(len(df))
        y = df['value'].values
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend direction
        if slope > 0.01:
            trend_direction = "increasing"
        elif slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(y, 0.95)
        
        return PerformanceTrend(
            metric=metric,
            time_window=time_window,
            start_time=df['timestamp'].min(),
            end_time=df['timestamp'].max(),
            data_points=data,
            trend_direction=trend_direction,
            trend_strength=abs(r_squared),
            slope=slope,
            r_squared=r_squared,
            confidence_interval=confidence_interval
        )
    
    async def detect_anomalies(
        self,
        metric: str,
        time_window: TimeWindow,
        sensitivity: float = 2.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PerformanceAnomaly]:
        """Detect anomalies in performance metrics."""
        # Get data for the time period
        data = await self._get_metric_data(metric, time_window, start_time, end_time)
        
        if not data:
            return []
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate rolling statistics
        window_size = min(20, len(df) // 4)  # Adaptive window size
        if window_size < 3:
            return []
        
        df['rolling_mean'] = df['value'].rolling(window=window_size, center=True).mean()
        df['rolling_std'] = df['value'].rolling(window=window_size, center=True).std()
        
        # Detect anomalies using z-score
        anomalies = []
        for idx, row in df.iterrows():
            if pd.isna(row['rolling_mean']) or pd.isna(row['rolling_std']):
                continue
            
            if row['rolling_std'] == 0:
                continue
            
            z_score = abs((row['value'] - row['rolling_mean']) / row['rolling_std'])
            
            if z_score > sensitivity:
                # Calculate severity
                if z_score > 4.0:
                    severity = "critical"
                elif z_score > 3.0:
                    severity = "high"
                elif z_score > 2.5:
                    severity = "medium"
                else:
                    severity = "low"
                
                # Calculate confidence
                confidence = min(0.99, 1 - (1 / (1 + z_score)))
                
                anomaly = PerformanceAnomaly(
                    metric=metric,
                    timestamp=row['timestamp'],
                    value=row['value'],
                    expected_value=row['rolling_mean'],
                    deviation=z_score,
                    severity=severity,
                    confidence=confidence,
                    description=f"Anomaly detected: {row['value']:.2f} (expected: {row['rolling_mean']:.2f})"
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    async def analyze_correlations(
        self,
        metrics: List[str],
        time_window: TimeWindow,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PerformanceCorrelation]:
        """Analyze correlations between metrics."""
        correlations = []
        
        # Get data for all metrics
        metric_data = {}
        for metric in metrics:
            data = await self._get_metric_data(metric, time_window, start_time, end_time)
            if data:
                metric_data[metric] = data
        
        if len(metric_data) < 2:
            return correlations
        
        # Create DataFrame with all metrics
        all_timestamps = set()
        for data in metric_data.values():
            all_timestamps.update([point[0] for point in data])
        
        df = pd.DataFrame(index=sorted(all_timestamps))
        df.index.name = 'timestamp'
        
        for metric, data in metric_data.items():
            data_dict = {point[0]: point[1] for point in data}
            df[metric] = df.index.map(data_dict)
        
        # Fill missing values with forward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate correlations
        correlation_matrix = df.corr()
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics[i+1:], i+1):
                if metric1 in df.columns and metric2 in df.columns:
                    corr_value = correlation_matrix.loc[metric1, metric2]
                    
                    if not pd.isna(corr_value):
                        # Calculate significance
                        if abs(corr_value) > 0.7:
                            significance = "strong"
                        elif abs(corr_value) > 0.5:
                            significance = "moderate"
                        elif abs(corr_value) > 0.3:
                            significance = "weak"
                        else:
                            significance = "none"
                        
                        # Determine relationship
                        if corr_value > 0.1:
                            relationship = "positive"
                        elif corr_value < -0.1:
                            relationship = "negative"
                        else:
                            relationship = "none"
                        
                        # Calculate p-value (simplified)
                        n = len(df)
                        t_stat = corr_value * np.sqrt((n - 2) / (1 - corr_value**2))
                        p_value = 2 * (1 - self._t_cdf(abs(t_stat), n - 2))
                        
                        correlation = PerformanceCorrelation(
                            metric1=metric1,
                            metric2=metric2,
                            correlation_coefficient=corr_value,
                            p_value=p_value,
                            significance=significance,
                            relationship=relationship
                        )
                        
                        correlations.append(correlation)
        
        return correlations
    
    async def generate_forecast(
        self,
        metric: str,
        forecast_horizon: int,
        time_window: TimeWindow,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> PerformanceForecast:
        """Generate performance forecast."""
        # Get historical data
        data = await self._get_metric_data(metric, time_window, start_time, end_time)
        
        if not data or len(data) < 10:
            return None
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Simple moving average forecast
        window_size = min(10, len(df) // 3)
        last_values = df['value'].tail(window_size).values
        
        # Generate forecast
        forecast_values = []
        confidence_intervals = []
        
        for i in range(forecast_horizon):
            forecast_time = df['timestamp'].max() + timedelta(minutes=i+1)
            forecast_value = np.mean(last_values)
            
            # Simple confidence interval
            std_dev = np.std(last_values)
            confidence_interval = (
                forecast_value - 1.96 * std_dev,
                forecast_value + 1.96 * std_dev
            )
            
            forecast_values.append((forecast_time, forecast_value))
            confidence_intervals.append(confidence_interval)
            
            # Update last values (simple approach)
            last_values = np.append(last_values[1:], forecast_value)
        
        # Calculate model accuracy (using historical data)
        actual_values = df['value'].tail(forecast_horizon).values if len(df) >= forecast_horizon else []
        if len(actual_values) > 0:
            predicted_values = [v[1] for v in forecast_values[:len(actual_values)]]
            mse = np.mean((actual_values - predicted_values) ** 2)
            accuracy = 1 / (1 + mse)  # Simple accuracy metric
        else:
            accuracy = 0.5  # Default accuracy
        
        return PerformanceForecast(
            metric=metric,
            forecast_horizon=forecast_horizon,
            forecast_values=forecast_values,
            confidence_intervals=confidence_intervals,
            model_accuracy=accuracy,
            model_type="moving_average"
        )
    
    async def generate_report(
        self,
        report_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> PerformanceReport:
        """Generate a comprehensive performance report."""
        if report_type not in self.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")
        
        template = self.report_templates[report_type]
        
        # Set time period
        if not start_time or not end_time:
            end_time = datetime.now(timezone.utc)
            if template["time_window"] == TimeWindow.HOUR:
                start_time = end_time - timedelta(hours=1)
            elif template["time_window"] == TimeWindow.DAY:
                start_time = end_time - timedelta(days=1)
            elif template["time_window"] == TimeWindow.WEEK:
                start_time = end_time - timedelta(weeks=1)
        
        # Generate analyses
        trends = []
        anomalies = []
        correlations = []
        forecasts = []
        
        for metric in template["metrics"]:
            if AnalysisType.TREND in template["analysis_types"]:
                trend = await self.analyze_trends(metric, template["time_window"], start_time, end_time)
                if trend:
                    trends.append(trend)
            
            if AnalysisType.ANOMALY in template["analysis_types"]:
                metric_anomalies = await self.detect_anomalies(metric, template["time_window"], start_time=start_time, end_time=end_time)
                anomalies.extend(metric_anomalies)
            
            if AnalysisType.FORECAST in template["analysis_types"]:
                forecast = await self.generate_forecast(metric, 60, template["time_window"], start_time, end_time)
                if forecast:
                    forecasts.append(forecast)
        
        if AnalysisType.CORRELATION in template["analysis_types"]:
            correlations = await self.analyze_correlations(template["metrics"], template["time_window"], start_time, end_time)
        
        # Generate summary
        summary = await self._generate_summary(template["metrics"], start_time, end_time)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trends, anomalies, correlations)
        
        # Generate charts
        charts = await self._generate_charts(template["metrics"], start_time, end_time)
        
        return PerformanceReport(
            report_id=f"report_{int(time.time())}",
            title=template["title"],
            generated_at=datetime.now(timezone.utc),
            time_period=(start_time, end_time),
            summary=summary,
            trends=trends,
            anomalies=anomalies,
            correlations=correlations,
            forecasts=forecasts,
            recommendations=recommendations,
            charts=charts
        )
    
    async def _get_metric_data(
        self,
        metric: str,
        time_window: TimeWindow,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Tuple[datetime, float]]:
        """Get metric data for analysis."""
        if not start_time:
            if time_window == TimeWindow.MINUTE:
                start_time = datetime.now(timezone.utc) - timedelta(minutes=60)
            elif time_window == TimeWindow.HOUR:
                start_time = datetime.now(timezone.utc) - timedelta(hours=1)
            elif time_window == TimeWindow.DAY:
                start_time = datetime.now(timezone.utc) - timedelta(days=1)
            elif time_window == TimeWindow.WEEK:
                start_time = datetime.now(timezone.utc) - timedelta(weeks=1)
            elif time_window == TimeWindow.MONTH:
                start_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        if not end_time:
            end_time = datetime.now(timezone.utc)
        
        # Get system metrics from monitor
        system_metrics = self.monitor.get_system_metrics_history(
            minutes=int((end_time - start_time).total_seconds() / 60)
        )
        
        # Filter by time range
        filtered_metrics = [
            metrics for metrics in system_metrics
            if start_time <= metrics.timestamp <= end_time
        ]
        
        # Extract metric values
        data = []
        for metrics in filtered_metrics:
            if metric == "response_time":
                value = metrics.avg_response_time_ms
            elif metric == "throughput":
                value = metrics.throughput_rps
            elif metric == "error_rate":
                value = metrics.error_rate_percent
            elif metric == "cpu_usage":
                value = metrics.cpu_usage_percent
            elif metric == "memory_usage":
                value = metrics.memory_usage_percent
            else:
                continue
            
            data.append((metrics.timestamp, value))
        
        return data
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for data."""
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        std_err = np.std(data) / np.sqrt(len(data))
        
        # Z-score for confidence level
        z_score = 1.96 if confidence == 0.95 else 1.645 if confidence == 0.90 else 2.576
        
        margin_of_error = z_score * std_err
        
        return (mean - margin_of_error, mean + margin_of_error)
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Calculate t-distribution CDF (simplified)."""
        # Simplified implementation - in practice, use scipy.stats
        return 0.5 + 0.5 * np.tanh(t / np.sqrt(df))
    
    async def _generate_summary(
        self,
        metrics: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {
            "time_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_hours": (end_time - start_time).total_seconds() / 3600
            },
            "metrics": {}
        }
        
        for metric in metrics:
            data = await self._get_metric_data(metric, TimeWindow.HOUR, start_time, end_time)
            if data:
                values = [point[1] for point in data]
                summary["metrics"][metric] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values)
                }
        
        return summary
    
    def _generate_recommendations(
        self,
        trends: List[PerformanceTrend],
        anomalies: List[PerformanceAnomaly],
        correlations: List[PerformanceCorrelation]
    ) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Analyze trends
        for trend in trends:
            if trend.trend_direction == "increasing" and trend.metric == "response_time":
                recommendations.append(
                    f"Response time is increasing (slope: {trend.slope:.3f}). "
                    "Consider optimizing database queries or adding caching."
                )
            elif trend.trend_direction == "decreasing" and trend.metric == "throughput":
                recommendations.append(
                    f"Throughput is decreasing (slope: {trend.slope:.3f}). "
                    "Check for resource bottlenecks or increased load."
                )
        
        # Analyze anomalies
        critical_anomalies = [a for a in anomalies if a.severity in ["high", "critical"]]
        if critical_anomalies:
            recommendations.append(
                f"Found {len(critical_anomalies)} critical anomalies. "
                "Investigate system stability and resource usage."
            )
        
        # Analyze correlations
        strong_correlations = [c for c in correlations if c.significance == "strong"]
        for corr in strong_correlations:
            if corr.relationship == "positive" and "cpu_usage" in [corr.metric1, corr.metric2]:
                recommendations.append(
                    f"Strong positive correlation between {corr.metric1} and {corr.metric2}. "
                    "Consider scaling resources or optimizing code."
                )
        
        return recommendations
    
    async def _generate_charts(
        self,
        metrics: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Generate performance charts."""
        charts = []
        
        for metric in metrics:
            data = await self._get_metric_data(metric, TimeWindow.HOUR, start_time, end_time)
            if not data:
                continue
            
            # Create chart
            timestamps = [point[0] for point in data]
            values = [point[1] for point in data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, values, marker='o', linewidth=2, markersize=4)
            plt.title(f"{metric.replace('_', ' ').title()} Over Time")
            plt.xlabel("Time")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            charts.append({
                "metric": metric,
                "title": f"{metric.replace('_', ' ').title()} Over Time",
                "data": chart_data,
                "type": "line"
            })
        
        return charts

# =============================================================================
# Performance Dashboard
# =============================================================================

class PerformanceDashboard:
    """Real-time performance dashboard."""
    
    def __init__(self, monitor: PerformanceMonitor, analytics: PerformanceAnalytics):
        
    """__init__ function."""
self.monitor = monitor
        self.analytics = analytics
        self.dashboard_data: Dict[str, Any] = {}
        self._update_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start_dashboard(self) -> Any:
        """Start dashboard updates."""
        if self._is_running:
            return
        
        self._is_running = True
        self._update_task = asyncio.create_task(self._dashboard_update_loop())
        logger.info("Performance dashboard started")
    
    async def stop_dashboard(self) -> Any:
        """Stop dashboard updates."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance dashboard stopped")
    
    async def _dashboard_update_loop(self) -> Any:
        """Dashboard update loop."""
        while self._is_running:
            try:
                await self._update_dashboard_data()
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                await asyncio.sleep(30)
    
    async def _update_dashboard_data(self) -> Any:
        """Update dashboard data."""
        # Get current performance summary
        summary = self.monitor.get_performance_summary()
        
        # Get recent trends
        trends = []
        for metric in ["response_time", "throughput", "error_rate"]:
            trend = await self.analytics.analyze_trends(metric, TimeWindow.HOUR)
            if trend:
                trends.append(trend)
        
        # Get recent anomalies
        anomalies = []
        for metric in ["response_time", "throughput", "error_rate", "cpu_usage"]:
            metric_anomalies = await self.analytics.detect_anomalies(metric, TimeWindow.HOUR)
            anomalies.extend(metric_anomalies)
        
        # Update dashboard data
        self.dashboard_data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "trends": [asdict(trend) for trend in trends],
            "anomalies": [asdict(anomaly) for anomaly in anomalies],
            "alerts": summary.get("alerts", [])
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data.copy()

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "AnalysisType",
    "TimeWindow",
    "PerformanceTrend",
    "PerformanceAnomaly",
    "PerformanceCorrelation",
    "PerformanceForecast",
    "PerformanceReport",
    "PerformanceAnalytics",
    "PerformanceDashboard",
] 