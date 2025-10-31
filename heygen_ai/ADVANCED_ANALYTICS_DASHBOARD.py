#!/usr/bin/env python3
"""
📊 HeyGen AI - Advanced Analytics & Insights Dashboard
=====================================================

This module implements a comprehensive analytics and insights dashboard that
provides real-time monitoring, predictive analytics, and intelligent insights
for the HeyGen AI system.
"""

import asyncio
import logging
import time
import json
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import hashlib
import secrets
import base64
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import aiohttp
import asyncio
from aiohttp import web, WSMsgType
import ssl
import certifi
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Metric types for analytics"""
    PERFORMANCE = "performance"
    USAGE = "usage"
    ERROR = "error"
    SECURITY = "security"
    COST = "cost"
    QUALITY = "quality"
    PREDICTIVE = "predictive"
    CUSTOM = "custom"

class ChartType(str, Enum):
    """Chart types for visualization"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    AREA = "area"
    CANDLESTICK = "candlestick"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"

class AlertLevel(str, Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class InsightType(str, Enum):
    """Insight types"""
    TREND = "trend"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    OPTIMIZATION = "optimization"

@dataclass
class MetricData:
    """Metric data point"""
    timestamp: datetime
    metric_type: MetricType
    metric_name: str
    value: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChartConfig:
    """Chart configuration"""
    chart_id: str
    chart_type: ChartType
    title: str
    x_axis: str
    y_axis: str
    data_source: str
    filters: Dict[str, Any] = field(default_factory=dict)
    styling: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 60  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    metric_name: str
    condition: str
    threshold: float
    alert_level: AlertLevel
    message: str
    enabled: bool = True
    cooldown: int = 300  # seconds
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Insight:
    """Analytics insight"""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    confidence: float
    impact: str
    recommendations: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataCollector:
    """Data collection engine"""
    
    def __init__(self):
        self.metrics_buffer: List[MetricData] = []
        self.buffer_size = 10000
        self.collection_interval = 1.0  # seconds
        self.initialized = False
    
    async def initialize(self):
        """Initialize data collector"""
        self.initialized = True
        logger.info("✅ Data Collector initialized")
    
    async def collect_metric(self, metric: MetricData):
        """Collect metric data"""
        if not self.initialized:
            return False
        
        try:
            # Add to buffer
            self.metrics_buffer.append(metric)
            
            # Flush buffer if full
            if len(self.metrics_buffer) >= self.buffer_size:
                await self._flush_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to collect metric: {e}")
            return False
    
    async def _flush_buffer(self):
        """Flush metrics buffer"""
        if not self.metrics_buffer:
            return
        
        try:
            # Process metrics (simplified)
            logger.info(f"Flushing {len(self.metrics_buffer)} metrics")
            
            # Clear buffer
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"❌ Failed to flush buffer: {e}")
    
    async def get_metrics(self, metric_name: str, start_time: datetime, 
                         end_time: datetime) -> List[MetricData]:
        """Get metrics for time range"""
        # Simplified implementation
        return [m for m in self.metrics_buffer 
                if m.metric_name == metric_name and 
                start_time <= m.timestamp <= end_time]

class AnomalyDetector:
    """Anomaly detection engine"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.initialized = False
    
    async def initialize(self):
        """Initialize anomaly detector"""
        self.initialized = True
        logger.info("✅ Anomaly Detector initialized")
    
    async def detect_anomalies(self, metrics: List[MetricData]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics"""
        if not self.initialized or not metrics:
            return []
        
        try:
            # Prepare data
            values = np.array([m.value for m in metrics])
            timestamps = [m.timestamp for m in metrics]
            
            if len(values) < 10:
                return []
            
            # Reshape for sklearn
            X = values.reshape(-1, 1)
            
            # Scale data
            X_scaled = self.scaler.fit_transform(X)
            
            # Detect anomalies
            anomaly_scores = self.isolation_forest.fit_predict(X_scaled)
            
            # Find anomalies
            anomalies = []
            for i, (score, metric) in enumerate(zip(anomaly_scores, metrics)):
                if score == -1:  # Anomaly detected
                    anomalies.append({
                        'timestamp': metric.timestamp,
                        'value': metric.value,
                        'metric_name': metric.metric_name,
                        'anomaly_score': abs(score),
                        'severity': self._calculate_severity(metric.value, values)
                    })
            
            logger.info(f"✅ Detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
            return []
    
    def _calculate_severity(self, value: float, all_values: np.ndarray) -> str:
        """Calculate anomaly severity"""
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        
        if std_val == 0:
            return "low"
        
        z_score = abs(value - mean_val) / std_val
        
        if z_score > 3:
            return "high"
        elif z_score > 2:
            return "medium"
        else:
            return "low"

class TrendAnalyzer:
    """Trend analysis engine"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize trend analyzer"""
        self.initialized = True
        logger.info("✅ Trend Analyzer initialized")
    
    async def analyze_trends(self, metrics: List[MetricData]) -> List[Dict[str, Any]]:
        """Analyze trends in metrics"""
        if not self.initialized or not metrics:
            return []
        
        try:
            # Group metrics by name
            metric_groups = {}
            for metric in metrics:
                if metric.metric_name not in metric_groups:
                    metric_groups[metric.metric_name] = []
                metric_groups[metric.metric_name].append(metric)
            
            trends = []
            
            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) < 5:
                    continue
                
                # Sort by timestamp
                metric_list.sort(key=lambda x: x.timestamp)
                
                # Extract values and timestamps
                values = [m.value for m in metric_list]
                timestamps = [m.timestamp for m in metric_list]
                
                # Calculate trend
                trend_info = self._calculate_trend(values, timestamps)
                
                if trend_info:
                    trends.append({
                        'metric_name': metric_name,
                        'trend_direction': trend_info['direction'],
                        'trend_strength': trend_info['strength'],
                        'change_percentage': trend_info['change_percentage'],
                        'confidence': trend_info['confidence'],
                        'time_range': {
                            'start': timestamps[0],
                            'end': timestamps[-1]
                        }
                    })
            
            logger.info(f"✅ Analyzed trends for {len(trends)} metrics")
            return trends
            
        except Exception as e:
            logger.error(f"❌ Trend analysis failed: {e}")
            return []
    
    def _calculate_trend(self, values: List[float], timestamps: List[datetime]) -> Optional[Dict[str, Any]]:
        """Calculate trend for a series of values"""
        if len(values) < 3:
            return None
        
        try:
            # Simple linear regression
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Calculate R-squared for confidence
            y_pred = np.polyval([slope, np.mean(y) - slope * np.mean(x)], x)
            r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            # Determine direction and strength
            if abs(slope) < 0.01:
                direction = "stable"
                strength = "weak"
            elif slope > 0:
                direction = "increasing"
                strength = "strong" if abs(slope) > 0.1 else "moderate"
            else:
                direction = "decreasing"
                strength = "strong" if abs(slope) > 0.1 else "moderate"
            
            # Calculate percentage change
            change_percentage = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
            
            return {
                'direction': direction,
                'strength': strength,
                'change_percentage': change_percentage,
                'confidence': max(0, min(1, r_squared))
            }
            
        except Exception:
            return None

class PredictiveAnalytics:
    """Predictive analytics engine"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize predictive analytics"""
        self.initialized = True
        logger.info("✅ Predictive Analytics initialized")
    
    async def generate_predictions(self, metrics: List[MetricData], 
                                 prediction_horizon: int = 24) -> List[Dict[str, Any]]:
        """Generate predictions for metrics"""
        if not self.initialized or not metrics:
            return []
        
        try:
            # Group metrics by name
            metric_groups = {}
            for metric in metrics:
                if metric.metric_name not in metric_groups:
                    metric_groups[metric.metric_name] = []
                metric_groups[metric.metric_name].append(metric)
            
            predictions = []
            
            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) < 10:
                    continue
                
                # Sort by timestamp
                metric_list.sort(key=lambda x: x.timestamp)
                
                # Extract values
                values = [m.value for m in metric_list]
                
                # Generate prediction (simplified)
                prediction = self._simple_forecast(values, prediction_horizon)
                
                if prediction:
                    predictions.append({
                        'metric_name': metric_name,
                        'current_value': values[-1],
                        'predicted_values': prediction['values'],
                        'predicted_timestamps': prediction['timestamps'],
                        'confidence': prediction['confidence'],
                        'trend': prediction['trend']
                    })
            
            logger.info(f"✅ Generated predictions for {len(predictions)} metrics")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Predictive analytics failed: {e}")
            return []
    
    def _simple_forecast(self, values: List[float], horizon: int) -> Optional[Dict[str, Any]]:
        """Simple forecasting method"""
        if len(values) < 3:
            return None
        
        try:
            # Calculate trend
            x = np.arange(len(values))
            y = np.array(values)
            slope, intercept = np.polyfit(x, y, 1)
            
            # Generate future values
            future_x = np.arange(len(values), len(values) + horizon)
            future_values = slope * future_x + intercept
            
            # Generate timestamps (simplified)
            last_time = datetime.now()
            future_timestamps = [
                last_time + timedelta(hours=i) for i in range(1, horizon + 1)
            ]
            
            # Calculate confidence based on recent variance
            recent_values = values[-min(10, len(values)):]
            variance = np.var(recent_values)
            confidence = max(0.1, min(0.9, 1 - (variance / np.mean(recent_values))))
            
            # Determine trend
            trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            
            return {
                'values': future_values.tolist(),
                'timestamps': future_timestamps,
                'confidence': confidence,
                'trend': trend
            }
            
        except Exception:
            return None

class ChartGenerator:
    """Chart generation engine"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize chart generator"""
        self.initialized = True
        logger.info("✅ Chart Generator initialized")
    
    async def generate_chart(self, config: ChartConfig, data: List[MetricData]) -> str:
        """Generate chart HTML"""
        if not self.initialized:
            return ""
        
        try:
            if config.chart_type == ChartType.LINE:
                return await self._generate_line_chart(config, data)
            elif config.chart_type == ChartType.BAR:
                return await self._generate_bar_chart(config, data)
            elif config.chart_type == ChartType.PIE:
                return await self._generate_pie_chart(config, data)
            elif config.chart_type == ChartType.SCATTER:
                return await self._generate_scatter_chart(config, data)
            elif config.chart_type == ChartType.HEATMAP:
                return await self._generate_heatmap(config, data)
            else:
                return await self._generate_line_chart(config, data)
                
        except Exception as e:
            logger.error(f"❌ Chart generation failed: {e}")
            return ""
    
    async def _generate_line_chart(self, config: ChartConfig, data: List[MetricData]) -> str:
        """Generate line chart"""
        if not data:
            return ""
        
        # Prepare data
        timestamps = [m.timestamp for m in data]
        values = [m.value for m in data]
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name=config.title,
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis,
            yaxis_title=config.y_axis,
            template='plotly_white',
            height=400
        )
        
        # Convert to HTML
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    async def _generate_bar_chart(self, config: ChartConfig, data: List[MetricData]) -> str:
        """Generate bar chart"""
        if not data:
            return ""
        
        # Group data by metric name
        metric_groups = {}
        for metric in data:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric)
        
        # Create figure
        fig = go.Figure()
        
        for metric_name, metric_list in metric_groups.items():
            values = [m.value for m in metric_list]
            timestamps = [m.timestamp.strftime('%Y-%m-%d %H:%M') for m in metric_list]
            
            fig.add_trace(go.Bar(
                x=timestamps,
                y=values,
                name=metric_name
            ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis,
            yaxis_title=config.y_axis,
            template='plotly_white',
            height=400
        )
        
        # Convert to HTML
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    async def _generate_pie_chart(self, config: ChartConfig, data: List[MetricData]) -> str:
        """Generate pie chart"""
        if not data:
            return ""
        
        # Group data by metric name
        metric_groups = {}
        for metric in data:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = 0
            metric_groups[metric.metric_name] += metric.value
        
        # Create figure
        fig = go.Figure(data=[go.Pie(
            labels=list(metric_groups.keys()),
            values=list(metric_groups.values()),
            hole=0.3
        )])
        
        # Update layout
        fig.update_layout(
            title=config.title,
            template='plotly_white',
            height=400
        )
        
        # Convert to HTML
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    async def _generate_scatter_chart(self, config: ChartConfig, data: List[MetricData]) -> str:
        """Generate scatter chart"""
        if not data:
            return ""
        
        # Prepare data
        timestamps = [m.timestamp for m in data]
        values = [m.value for m in data]
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='markers',
            name=config.title,
            marker=dict(size=8, color='#1f77b4')
        ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis,
            yaxis_title=config.y_axis,
            template='plotly_white',
            height=400
        )
        
        # Convert to HTML
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    async def _generate_heatmap(self, config: ChartConfig, data: List[MetricData]) -> str:
        """Generate heatmap"""
        if not data:
            return ""
        
        # Group data by hour and metric name
        heatmap_data = {}
        for metric in data:
            hour = metric.timestamp.hour
            if hour not in heatmap_data:
                heatmap_data[hour] = {}
            if metric.metric_name not in heatmap_data[hour]:
                heatmap_data[hour][metric.metric_name] = []
            heatmap_data[hour][metric.metric_name].append(metric.value)
        
        # Calculate averages
        z_data = []
        y_labels = []
        x_labels = list(heatmap_data.keys())
        
        for metric_name in set(m.metric_name for m in data):
            y_labels.append(metric_name)
            row = []
            for hour in sorted(x_labels):
                if metric_name in heatmap_data[hour]:
                    avg_value = np.mean(heatmap_data[hour][metric_name])
                    row.append(avg_value)
                else:
                    row.append(0)
            z_data.append(row)
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis'
        ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title='Hour',
            yaxis_title='Metric',
            template='plotly_white',
            height=400
        )
        
        # Convert to HTML
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)

class AdvancedAnalyticsDashboard:
    """Main analytics dashboard system"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.anomaly_detector = AnomalyDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.predictive_analytics = PredictiveAnalytics()
        self.chart_generator = ChartGenerator()
        self.alerts: Dict[str, Alert] = {}
        self.insights: List[Insight] = []
        self.charts: Dict[str, ChartConfig] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize analytics dashboard"""
        try:
            logger.info("📊 Initializing Advanced Analytics Dashboard...")
            
            # Initialize components
            await self.data_collector.initialize()
            await self.anomaly_detector.initialize()
            await self.trend_analyzer.initialize()
            await self.predictive_analytics.initialize()
            await self.chart_generator.initialize()
            
            # Initialize default charts
            await self._initialize_default_charts()
            
            self.initialized = True
            logger.info("✅ Advanced Analytics Dashboard initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Analytics Dashboard: {e}")
            raise
    
    async def _initialize_default_charts(self):
        """Initialize default charts"""
        default_charts = [
            ChartConfig(
                chart_id="performance_overview",
                chart_type=ChartType.LINE,
                title="Performance Overview",
                x_axis="Time",
                y_axis="Performance Score",
                data_source="performance_metrics"
            ),
            ChartConfig(
                chart_id="usage_breakdown",
                chart_type=ChartType.PIE,
                title="Usage Breakdown",
                x_axis="",
                y_axis="",
                data_source="usage_metrics"
            ),
            ChartConfig(
                chart_id="error_trends",
                chart_type=ChartType.BAR,
                title="Error Trends",
                x_axis="Time",
                y_axis="Error Count",
                data_source="error_metrics"
            ),
            ChartConfig(
                chart_id="system_health",
                chart_type=ChartType.HEATMAP,
                title="System Health Heatmap",
                x_axis="Hour",
                y_axis="Component",
                data_source="health_metrics"
            )
        ]
        
        for chart in default_charts:
            self.charts[chart.chart_id] = chart
    
    async def collect_metric(self, metric_name: str, value: float, 
                           metric_type: MetricType = MetricType.PERFORMANCE,
                           unit: str = "", tags: Dict[str, str] = None) -> bool:
        """Collect metric data"""
        if not self.initialized:
            return False
        
        try:
            metric = MetricData(
                timestamp=datetime.now(),
                metric_type=metric_type,
                metric_name=metric_name,
                value=value,
                unit=unit,
                tags=tags or {}
            )
            
            success = await self.data_collector.collect_metric(metric)
            
            if success:
                # Check alerts
                await self._check_alerts(metric)
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to collect metric {metric_name}: {e}")
            return False
    
    async def _check_alerts(self, metric: MetricData):
        """Check alerts for metric"""
        for alert in self.alerts.values():
            if not alert.enabled or alert.metric_name != metric.metric_name:
                continue
            
            # Check cooldown
            if alert.last_triggered and \
               (datetime.now() - alert.last_triggered).seconds < alert.cooldown:
                continue
            
            # Check condition
            if self._evaluate_condition(metric.value, alert.condition, alert.threshold):
                await self._trigger_alert(alert, metric)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return value == threshold
        elif condition == "not_equals":
            return value != threshold
        else:
            return False
    
    async def _trigger_alert(self, alert: Alert, metric: MetricData):
        """Trigger alert"""
        alert.last_triggered = datetime.now()
        
        logger.warning(f"🚨 Alert triggered: {alert.message} (Value: {metric.value})")
        
        # Generate insight
        insight = Insight(
            insight_id=str(uuid.uuid4()),
            insight_type=InsightType.ANOMALY,
            title=f"Alert: {alert.metric_name}",
            description=alert.message,
            confidence=0.9,
            impact=alert.alert_level.value,
            recommendations=[f"Check {alert.metric_name} configuration"],
            related_metrics=[alert.metric_name]
        )
        
        self.insights.append(insight)
    
    async def create_alert(self, alert: Alert) -> bool:
        """Create new alert"""
        try:
            self.alerts[alert.alert_id] = alert
            logger.info(f"✅ Alert created: {alert.alert_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create alert: {e}")
            return False
    
    async def generate_insights(self, time_range_hours: int = 24) -> List[Insight]:
        """Generate insights from recent data"""
        if not self.initialized:
            return []
        
        try:
            # Get recent metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_range_hours)
            
            # This is simplified - in real implementation, get from database
            recent_metrics = self.data_collector.metrics_buffer
            
            insights = []
            
            # Detect anomalies
            anomalies = await self.anomaly_detector.detect_anomalies(recent_metrics)
            for anomaly in anomalies:
                insight = Insight(
                    insight_id=str(uuid.uuid4()),
                    insight_type=InsightType.ANOMALY,
                    title=f"Anomaly detected in {anomaly['metric_name']}",
                    description=f"Unusual value detected: {anomaly['value']}",
                    confidence=anomaly['anomaly_score'],
                    impact=anomaly['severity'],
                    recommendations=["Investigate root cause", "Check system health"],
                    related_metrics=[anomaly['metric_name']]
                )
                insights.append(insight)
            
            # Analyze trends
            trends = await self.trend_analyzer.analyze_trends(recent_metrics)
            for trend in trends:
                insight = Insight(
                    insight_id=str(uuid.uuid4()),
                    insight_type=InsightType.TREND,
                    title=f"Trend detected in {trend['metric_name']}",
                    description=f"Metric is {trend['trend_direction']} with {trend['change_percentage']:.1f}% change",
                    confidence=trend['confidence'],
                    impact="medium",
                    recommendations=["Monitor closely", "Consider optimization"],
                    related_metrics=[trend['metric_name']]
                )
                insights.append(insight)
            
            # Generate predictions
            predictions = await self.predictive_analytics.generate_predictions(recent_metrics)
            for prediction in predictions:
                insight = Insight(
                    insight_id=str(uuid.uuid4()),
                    insight_type=InsightType.PREDICTION,
                    title=f"Prediction for {prediction['metric_name']}",
                    description=f"Predicted trend: {prediction['trend']}",
                    confidence=prediction['confidence'],
                    impact="low",
                    recommendations=["Plan accordingly", "Monitor predictions"],
                    related_metrics=[prediction['metric_name']]
                )
                insights.append(insight)
            
            # Store insights
            self.insights.extend(insights)
            
            logger.info(f"✅ Generated {len(insights)} insights")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate insights: {e}")
            return []
    
    async def generate_chart(self, chart_id: str, time_range_hours: int = 24) -> str:
        """Generate chart HTML"""
        if not self.initialized or chart_id not in self.charts:
            return ""
        
        try:
            chart_config = self.charts[chart_id]
            
            # Get data for time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_range_hours)
            
            # This is simplified - in real implementation, get from database
            data = self.data_collector.metrics_buffer
            
            # Filter data by time range
            filtered_data = [m for m in data if start_time <= m.timestamp <= end_time]
            
            # Generate chart
            chart_html = await self.chart_generator.generate_chart(chart_config, filtered_data)
            
            return chart_html
            
        except Exception as e:
            logger.error(f"❌ Failed to generate chart {chart_id}: {e}")
            return ""
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        if not self.initialized:
            return {}
        
        try:
            # Generate insights
            insights = await self.generate_insights()
            
            # Get recent metrics summary
            recent_metrics = self.data_collector.metrics_buffer[-100:]  # Last 100 metrics
            
            # Calculate summary statistics
            if recent_metrics:
                metric_summary = {}
                for metric in recent_metrics:
                    if metric.metric_name not in metric_summary:
                        metric_summary[metric.metric_name] = []
                    metric_summary[metric.metric_name].append(metric.value)
                
                summary_stats = {}
                for name, values in metric_summary.items():
                    summary_stats[name] = {
                        'current': values[-1] if values else 0,
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            else:
                summary_stats = {}
            
            return {
                'summary_stats': summary_stats,
                'insights': [insight.__dict__ for insight in insights[-10:]],  # Last 10 insights
                'alerts': [alert.__dict__ for alert in self.alerts.values() if alert.enabled],
                'charts': list(self.charts.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get dashboard data: {e}")
            return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'initialized': self.initialized,
            'data_collector_ready': self.data_collector.initialized,
            'anomaly_detector_ready': self.anomaly_detector.initialized,
            'trend_analyzer_ready': self.trend_analyzer.initialized,
            'predictive_analytics_ready': self.predictive_analytics.initialized,
            'chart_generator_ready': self.chart_generator.initialized,
            'total_metrics': len(self.data_collector.metrics_buffer),
            'total_insights': len(self.insights),
            'active_alerts': len([a for a in self.alerts.values() if a.enabled]),
            'timestamp': datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown analytics dashboard"""
        self.initialized = False
        logger.info("✅ Advanced Analytics Dashboard shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the advanced analytics dashboard"""
    print("📊 HeyGen AI - Advanced Analytics Dashboard Demo")
    print("=" * 60)
    
    # Initialize system
    dashboard = AdvancedAnalyticsDashboard()
    
    try:
        # Initialize the system
        print("\n🚀 Initializing Advanced Analytics Dashboard...")
        await dashboard.initialize()
        print("✅ Advanced Analytics Dashboard initialized successfully")
        
        # Get system status
        print("\n📊 System Status:")
        status = await dashboard.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Collect some sample metrics
        print("\n📈 Collecting Sample Metrics...")
        
        # Performance metrics
        for i in range(50):
            await dashboard.collect_metric(
                "cpu_usage", 
                np.random.normal(60, 10), 
                MetricType.PERFORMANCE, 
                "%"
            )
            await dashboard.collect_metric(
                "memory_usage", 
                np.random.normal(70, 15), 
                MetricType.PERFORMANCE, 
                "%"
            )
            await dashboard.collect_metric(
                "response_time", 
                np.random.normal(100, 20), 
                MetricType.PERFORMANCE, 
                "ms"
            )
            
            # Add some anomalies
            if i % 10 == 0:
                await dashboard.collect_metric(
                    "cpu_usage", 
                    np.random.normal(90, 5), 
                    MetricType.PERFORMANCE, 
                    "%"
                )
        
        print("  ✅ Collected 150 sample metrics")
        
        # Create an alert
        print("\n🚨 Creating Alert...")
        
        alert = Alert(
            alert_id="cpu_high_alert",
            metric_name="cpu_usage",
            condition="greater_than",
            threshold=80.0,
            alert_level=AlertLevel.WARNING,
            message="CPU usage is above 80%",
            enabled=True
        )
        
        await dashboard.create_alert(alert)
        print("  ✅ Alert created")
        
        # Generate insights
        print("\n🔍 Generating Insights...")
        insights = await dashboard.generate_insights()
        print(f"  ✅ Generated {len(insights)} insights")
        
        # Show insights
        for insight in insights[:5]:  # Show first 5 insights
            print(f"    - {insight.title}: {insight.description}")
        
        # Generate charts
        print("\n📊 Generating Charts...")
        
        chart_ids = ["performance_overview", "usage_breakdown", "error_trends", "system_health"]
        for chart_id in chart_ids:
            chart_html = await dashboard.generate_chart(chart_id)
            if chart_html:
                print(f"  ✅ Generated chart: {chart_id}")
            else:
                print(f"  ❌ Failed to generate chart: {chart_id}")
        
        # Get dashboard data
        print("\n📋 Dashboard Data:")
        dashboard_data = await dashboard.get_dashboard_data()
        
        print(f"  Summary Stats: {len(dashboard_data.get('summary_stats', {}))} metrics")
        print(f"  Insights: {len(dashboard_data.get('insights', []))}")
        print(f"  Active Alerts: {len(dashboard_data.get('alerts', []))}")
        print(f"  Available Charts: {len(dashboard_data.get('charts', []))}")
        
        # Show summary stats
        summary_stats = dashboard_data.get('summary_stats', {})
        if summary_stats:
            print(f"\n  📊 Metric Summary:")
            for metric_name, stats in summary_stats.items():
                print(f"    {metric_name}:")
                print(f"      Current: {stats['current']:.2f}")
                print(f"      Average: {stats['average']:.2f}")
                print(f"      Min: {stats['min']:.2f}")
                print(f"      Max: {stats['max']:.2f}")
                print(f"      Count: {stats['count']}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Shutdown
        await dashboard.shutdown()
        print("\n✅ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())


