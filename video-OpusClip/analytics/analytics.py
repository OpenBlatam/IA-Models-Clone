#!/usr/bin/env python3
"""
Advanced Analytics and Business Intelligence System

Comprehensive analytics system with:
- Real-time metrics collection
- Business intelligence dashboards
- Performance analytics
- User behavior tracking
- Revenue analytics
- Predictive analytics
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import numpy as np
from collections import defaultdict, deque
import statistics

logger = structlog.get_logger("analytics")

# =============================================================================
# ANALYTICS MODELS
# =============================================================================

class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"

class AnalyticsEvent(Enum):
    """Types of analytics events."""
    VIDEO_PROCESSED = "video_processed"
    USER_LOGIN = "user_login"
    API_REQUEST = "api_request"
    ERROR_OCCURRED = "error_occurred"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    BATCH_COMPLETED = "batch_completed"
    VIRAL_GENERATED = "viral_generated"
    PAYMENT_PROCESSED = "payment_processed"
    SYSTEM_HEALTH = "system_health"

@dataclass
class MetricData:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: MetricType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "metric_type": self.metric_type.value
        }

@dataclass
class AnalyticsEventData:
    """Analytics event data."""
    event_type: AnalyticsEvent
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    properties: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "properties": self.properties,
            "metadata": self.metadata
        }

@dataclass
class PerformanceMetrics:
    """Performance metrics data."""
    response_time: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response_time": self.response_time,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class BusinessMetrics:
    """Business metrics data."""
    total_users: int
    active_users: int
    total_videos_processed: int
    total_revenue: float
    conversion_rate: float
    churn_rate: float
    customer_satisfaction: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_users": self.total_users,
            "active_users": self.active_users,
            "total_videos_processed": self.total_videos_processed,
            "total_revenue": self.total_revenue,
            "conversion_rate": self.conversion_rate,
            "churn_rate": self.churn_rate,
            "customer_satisfaction": self.customer_satisfaction,
            "timestamp": self.timestamp.isoformat()
        }

# =============================================================================
# ANALYTICS COLLECTOR
# =============================================================================

class AnalyticsCollector:
    """Advanced analytics data collector."""
    
    def __init__(self, max_buffer_size: int = 10000):
        self.max_buffer_size = max_buffer_size
        self.metrics_buffer: deque = deque(maxlen=max_buffer_size)
        self.events_buffer: deque = deque(maxlen=max_buffer_size)
        self.performance_buffer: deque = deque(maxlen=max_buffer_size)
        self.business_buffer: deque = deque(maxlen=max_buffer_size)
        
        # Real-time aggregations
        self.real_time_metrics: Dict[str, Any] = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'last_updated': time.time()
        })
        
        # Time-based windows
        self.time_windows = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '24h': 86400
        }
        
        # Windowed data
        self.windowed_data: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
    
    async def collect_metric(self, metric: MetricData) -> None:
        """Collect a metric data point."""
        try:
            # Add to buffer
            self.metrics_buffer.append(metric)
            
            # Update real-time aggregations
            self._update_real_time_metric(metric)
            
            # Update windowed data
            self._update_windowed_data(metric)
            
            # Log metric collection
            logger.debug(
                "Metric collected",
                metric_name=metric.name,
                value=metric.value,
                labels=metric.labels
            )
            
        except Exception as e:
            logger.error("Failed to collect metric", error=str(e), metric=metric.to_dict())
    
    async def collect_event(self, event: AnalyticsEventData) -> None:
        """Collect an analytics event."""
        try:
            # Add to buffer
            self.events_buffer.append(event)
            
            # Update real-time aggregations
            self._update_real_time_event(event)
            
            # Log event collection
            logger.info(
                "Analytics event collected",
                event_type=event.event_type.value,
                user_id=event.user_id,
                properties=event.properties
            )
            
        except Exception as e:
            logger.error("Failed to collect event", error=str(e), event=event.to_dict())
    
    async def collect_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect performance metrics."""
        try:
            # Add to buffer
            self.performance_buffer.append(metrics)
            
            # Update real-time aggregations
            self._update_real_time_performance(metrics)
            
            # Log performance collection
            logger.debug(
                "Performance metrics collected",
                response_time=metrics.response_time,
                throughput=metrics.throughput,
                error_rate=metrics.error_rate
            )
            
        except Exception as e:
            logger.error("Failed to collect performance metrics", error=str(e))
    
    async def collect_business_metrics(self, metrics: BusinessMetrics) -> None:
        """Collect business metrics."""
        try:
            # Add to buffer
            self.business_buffer.append(metrics)
            
            # Update real-time aggregations
            self._update_real_time_business(metrics)
            
            # Log business metrics collection
            logger.info(
                "Business metrics collected",
                total_users=metrics.total_users,
                active_users=metrics.active_users,
                total_revenue=metrics.total_revenue
            )
            
        except Exception as e:
            logger.error("Failed to collect business metrics", error=str(e))
    
    def _update_real_time_metric(self, metric: MetricData) -> None:
        """Update real-time metric aggregations."""
        key = f"{metric.name}_{json.dumps(metric.labels, sort_keys=True)}"
        
        if key not in self.real_time_metrics:
            self.real_time_metrics[key] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'last_updated': time.time()
            }
        
        data = self.real_time_metrics[key]
        data['count'] += 1
        data['sum'] += metric.value
        data['min'] = min(data['min'], metric.value)
        data['max'] = max(data['max'], metric.value)
        data['last_updated'] = time.time()
    
    def _update_real_time_event(self, event: AnalyticsEventData) -> None:
        """Update real-time event aggregations."""
        key = f"event_{event.event_type.value}"
        
        if key not in self.real_time_metrics:
            self.real_time_metrics[key] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'last_updated': time.time()
            }
        
        data = self.real_time_metrics[key]
        data['count'] += 1
        data['last_updated'] = time.time()
    
    def _update_real_time_performance(self, metrics: PerformanceMetrics) -> None:
        """Update real-time performance aggregations."""
        performance_keys = [
            'response_time', 'throughput', 'error_rate',
            'cpu_usage', 'memory_usage', 'disk_usage', 'network_io'
        ]
        
        for key in performance_keys:
            value = getattr(metrics, key)
            full_key = f"performance_{key}"
            
            if full_key not in self.real_time_metrics:
                self.real_time_metrics[full_key] = {
                    'count': 0,
                    'sum': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'last_updated': time.time()
                }
            
            data = self.real_time_metrics[full_key]
            data['count'] += 1
            data['sum'] += value
            data['min'] = min(data['min'], value)
            data['max'] = max(data['max'], value)
            data['last_updated'] = time.time()
    
    def _update_real_time_business(self, metrics: BusinessMetrics) -> None:
        """Update real-time business aggregations."""
        business_keys = [
            'total_users', 'active_users', 'total_videos_processed',
            'total_revenue', 'conversion_rate', 'churn_rate', 'customer_satisfaction'
        ]
        
        for key in business_keys:
            value = getattr(metrics, key)
            full_key = f"business_{key}"
            
            if full_key not in self.real_time_metrics:
                self.real_time_metrics[full_key] = {
                    'count': 0,
                    'sum': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'last_updated': time.time()
                }
            
            data = self.real_time_metrics[full_key]
            data['count'] += 1
            data['sum'] += value
            data['min'] = min(data['min'], value)
            data['max'] = max(data['max'], value)
            data['last_updated'] = time.time()
    
    def _update_windowed_data(self, metric: MetricData) -> None:
        """Update windowed data for time-based analysis."""
        current_time = time.time()
        
        for window_name, window_seconds in self.time_windows.items():
            window_key = f"{metric.name}_{window_name}"
            window_data = self.windowed_data[window_name][window_key]
            
            # Add current metric
            window_data.append({
                'value': metric.value,
                'timestamp': current_time,
                'labels': metric.labels
            })
            
            # Remove old data outside window
            cutoff_time = current_time - window_seconds
            while window_data and window_data[0]['timestamp'] < cutoff_time:
                window_data.popleft()
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics."""
        current_time = time.time()
        
        # Clean up old metrics (older than 1 hour)
        cutoff_time = current_time - 3600
        cleaned_metrics = {}
        
        for key, data in self.real_time_metrics.items():
            if data['last_updated'] > cutoff_time:
                cleaned_metrics[key] = data.copy()
                cleaned_metrics[key]['average'] = data['sum'] / data['count'] if data['count'] > 0 else 0
        
        return cleaned_metrics
    
    def get_windowed_metrics(self, window: str = '1h') -> Dict[str, Any]:
        """Get windowed metrics for a specific time window."""
        if window not in self.time_windows:
            raise ValueError(f"Invalid window: {window}")
        
        window_data = self.windowed_data[window]
        result = {}
        
        for metric_name, data in window_data.items():
            if not data:
                continue
            
            values = [item['value'] for item in data]
            result[metric_name] = {
                'count': len(values),
                'sum': sum(values),
                'average': statistics.mean(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'median': statistics.median(values) if values else 0,
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0
            }
        
        return result
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        return {
            'real_time_metrics': self.get_real_time_metrics(),
            'windowed_metrics': {
                window: self.get_windowed_metrics(window)
                for window in self.time_windows.keys()
            },
            'buffer_sizes': {
                'metrics': len(self.metrics_buffer),
                'events': len(self.events_buffer),
                'performance': len(self.performance_buffer),
                'business': len(self.business_buffer)
            },
            'timestamp': datetime.utcnow().isoformat()
        }

# =============================================================================
# ANALYTICS PROCESSOR
# =============================================================================

class AnalyticsProcessor:
    """Advanced analytics data processor."""
    
    def __init__(self, collector: AnalyticsCollector):
        self.collector = collector
        self.processors = {
            'trend_analysis': self._process_trend_analysis,
            'anomaly_detection': self._process_anomaly_detection,
            'correlation_analysis': self._process_correlation_analysis,
            'predictive_analysis': self._process_predictive_analysis
        }
    
    async def process_analytics(self, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Process analytics data."""
        if analysis_type not in self.processors:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        try:
            result = await self.processors[analysis_type](**kwargs)
            logger.info(f"Analytics processing completed", analysis_type=analysis_type)
            return result
        except Exception as e:
            logger.error(f"Analytics processing failed", analysis_type=analysis_type, error=str(e))
            raise
    
    async def _process_trend_analysis(self, metric_name: str, window: str = '24h') -> Dict[str, Any]:
        """Process trend analysis."""
        windowed_data = self.collector.get_windowed_metrics(window)
        
        if metric_name not in windowed_data:
            return {'error': f'Metric {metric_name} not found in window {window}'}
        
        data = windowed_data[metric_name]
        values = [item['value'] for item in self.collector.windowed_data[window][metric_name]]
        
        if len(values) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Calculate trend
        x = list(range(len(values)))
        y = values
        
        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'metric_name': metric_name,
            'window': window,
            'trend_slope': slope,
            'trend_intercept': intercept,
            'r_squared': r_squared,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'data_points': len(values),
            'current_value': values[-1] if values else 0,
            'predicted_next': slope * len(values) + intercept if len(values) > 0 else 0
        }
    
    async def _process_anomaly_detection(self, metric_name: str, window: str = '1h', threshold: float = 2.0) -> Dict[str, Any]:
        """Process anomaly detection."""
        windowed_data = self.collector.get_windowed_metrics(window)
        
        if metric_name not in windowed_data:
            return {'error': f'Metric {metric_name} not found in window {window}'}
        
        data = windowed_data[metric_name]
        values = [item['value'] for item in self.collector.windowed_data[window][metric_name]]
        
        if len(values) < 3:
            return {'error': 'Insufficient data for anomaly detection'}
        
        # Calculate Z-scores
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_dev == 0:
            return {'error': 'No variance in data for anomaly detection'}
        
        z_scores = [(value - mean) / std_dev for value in values]
        anomalies = []
        
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > threshold:
                anomalies.append({
                    'index': i,
                    'value': values[i],
                    'z_score': z_score,
                    'timestamp': self.collector.windowed_data[window][metric_name][i]['timestamp']
                })
        
        return {
            'metric_name': metric_name,
            'window': window,
            'threshold': threshold,
            'mean': mean,
            'std_dev': std_dev,
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'anomaly_rate': len(anomalies) / len(values) * 100
        }
    
    async def _process_correlation_analysis(self, metric1: str, metric2: str, window: str = '1h') -> Dict[str, Any]:
        """Process correlation analysis."""
        windowed_data = self.collector.get_windowed_metrics(window)
        
        if metric1 not in windowed_data or metric2 not in windowed_data:
            return {'error': f'One or both metrics not found in window {window}'}
        
        data1 = [item['value'] for item in self.collector.windowed_data[window][metric1]]
        data2 = [item['value'] for item in self.collector.windowed_data[window][metric2]]
        
        if len(data1) != len(data2) or len(data1) < 2:
            return {'error': 'Insufficient or mismatched data for correlation analysis'}
        
        # Calculate Pearson correlation coefficient
        n = len(data1)
        sum1 = sum(data1)
        sum2 = sum(data2)
        sum1_sq = sum(x ** 2 for x in data1)
        sum2_sq = sum(x ** 2 for x in data2)
        sum_xy = sum(data1[i] * data2[i] for i in range(n))
        
        numerator = n * sum_xy - sum1 * sum2
        denominator = ((n * sum1_sq - sum1 ** 2) * (n * sum2_sq - sum2 ** 2)) ** 0.5
        
        correlation = numerator / denominator if denominator != 0 else 0
        
        # Determine correlation strength
        if abs(correlation) >= 0.8:
            strength = 'strong'
        elif abs(correlation) >= 0.5:
            strength = 'moderate'
        elif abs(correlation) >= 0.3:
            strength = 'weak'
        else:
            strength = 'negligible'
        
        return {
            'metric1': metric1,
            'metric2': metric2,
            'window': window,
            'correlation_coefficient': correlation,
            'correlation_strength': strength,
            'data_points': n,
            'interpretation': self._interpret_correlation(correlation)
        }
    
    async def _process_predictive_analysis(self, metric_name: str, window: str = '24h', forecast_periods: int = 10) -> Dict[str, Any]:
        """Process predictive analysis using simple moving average."""
        windowed_data = self.collector.get_windowed_metrics(window)
        
        if metric_name not in windowed_data:
            return {'error': f'Metric {metric_name} not found in window {window}'}
        
        values = [item['value'] for item in self.collector.windowed_data[window][metric_name]]
        
        if len(values) < 3:
            return {'error': 'Insufficient data for predictive analysis'}
        
        # Simple moving average forecast
        window_size = min(5, len(values) // 2)
        if window_size < 2:
            window_size = 2
        
        # Calculate moving averages
        moving_averages = []
        for i in range(window_size - 1, len(values)):
            window_values = values[i - window_size + 1:i + 1]
            moving_averages.append(statistics.mean(window_values))
        
        # Forecast future values
        last_ma = moving_averages[-1] if moving_averages else values[-1]
        forecast = [last_ma] * forecast_periods
        
        # Calculate trend
        if len(moving_averages) >= 2:
            trend = moving_averages[-1] - moving_averages[-2]
            for i in range(forecast_periods):
                forecast[i] = last_ma + (trend * (i + 1))
        
        return {
            'metric_name': metric_name,
            'window': window,
            'forecast_periods': forecast_periods,
            'historical_data': values,
            'moving_averages': moving_averages,
            'forecast': forecast,
            'current_value': values[-1],
            'predicted_value': forecast[0],
            'trend': 'increasing' if len(moving_averages) >= 2 and moving_averages[-1] > moving_averages[-2] else 'decreasing'
        }
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient."""
        if correlation > 0.8:
            return "Strong positive correlation"
        elif correlation > 0.5:
            return "Moderate positive correlation"
        elif correlation > 0.3:
            return "Weak positive correlation"
        elif correlation > -0.3:
            return "No significant correlation"
        elif correlation > -0.5:
            return "Weak negative correlation"
        elif correlation > -0.8:
            return "Moderate negative correlation"
        else:
            return "Strong negative correlation"

# =============================================================================
# ANALYTICS DASHBOARD
# =============================================================================

class AnalyticsDashboard:
    """Real-time analytics dashboard."""
    
    def __init__(self, collector: AnalyticsCollector, processor: AnalyticsProcessor):
        self.collector = collector
        self.processor = processor
        self.dashboard_data = {}
        self.last_update = time.time()
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        current_time = time.time()
        
        # Update dashboard data if needed (every 30 seconds)
        if current_time - self.last_update > 30:
            await self._update_dashboard_data()
            self.last_update = current_time
        
        return self.dashboard_data
    
    async def _update_dashboard_data(self) -> None:
        """Update dashboard data."""
        try:
            # Get real-time metrics
            real_time_metrics = self.collector.get_real_time_metrics()
            
            # Get windowed metrics
            windowed_metrics = {}
            for window in ['1m', '5m', '15m', '1h', '24h']:
                windowed_metrics[window] = self.collector.get_windowed_metrics(window)
            
            # Get analytics summary
            analytics_summary = self.collector.get_analytics_summary()
            
            # Update dashboard data
            self.dashboard_data = {
                'real_time_metrics': real_time_metrics,
                'windowed_metrics': windowed_metrics,
                'analytics_summary': analytics_summary,
                'last_updated': datetime.utcnow().isoformat(),
                'status': 'healthy'
            }
            
            logger.info("Dashboard data updated successfully")
            
        except Exception as e:
            logger.error("Failed to update dashboard data", error=str(e))
            self.dashboard_data['status'] = 'error'
            self.dashboard_data['error'] = str(e)
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance-specific dashboard data."""
        try:
            # Get performance metrics
            performance_metrics = self.collector.get_windowed_metrics('1h')
            
            # Filter performance-related metrics
            performance_data = {}
            for key, data in performance_metrics.items():
                if key.startswith('performance_'):
                    performance_data[key] = data
            
            return {
                'performance_metrics': performance_data,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'healthy'
            }
            
        except Exception as e:
            logger.error("Failed to get performance dashboard", error=str(e))
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'error'
            }
    
    async def get_business_dashboard(self) -> Dict[str, Any]:
        """Get business-specific dashboard data."""
        try:
            # Get business metrics
            business_metrics = self.collector.get_windowed_metrics('24h')
            
            # Filter business-related metrics
            business_data = {}
            for key, data in business_metrics.items():
                if key.startswith('business_'):
                    business_data[key] = data
            
            return {
                'business_metrics': business_data,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'healthy'
            }
            
        except Exception as e:
            logger.error("Failed to get business dashboard", error=str(e))
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'error'
            }

# =============================================================================
# GLOBAL ANALYTICS INSTANCE
# =============================================================================

# Global analytics instances
analytics_collector = AnalyticsCollector()
analytics_processor = AnalyticsProcessor(analytics_collector)
analytics_dashboard = AnalyticsDashboard(analytics_collector, analytics_processor)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MetricType',
    'AnalyticsEvent',
    'MetricData',
    'AnalyticsEventData',
    'PerformanceMetrics',
    'BusinessMetrics',
    'AnalyticsCollector',
    'AnalyticsProcessor',
    'AnalyticsDashboard',
    'analytics_collector',
    'analytics_processor',
    'analytics_dashboard'
]





























