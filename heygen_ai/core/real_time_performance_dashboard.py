"""
Real-Time Performance Dashboard for HeyGen AI Enterprise

This module provides a comprehensive real-time performance dashboard with:
- Live performance monitoring and visualization
- Interactive performance charts and graphs
- Real-time alerting and notifications
- Performance trend analysis and forecasting
- Custom dashboard configurations
- Web-based interface for remote monitoring
- Performance data export and reporting
"""

import logging
import os
import time
import gc
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Dashboard and visualization libraries
try:
    import dash
    from dash import dcc, html, Input, Output, callback_context
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    warnings.warn("dash and plotly not available. Install for dashboard functionality.")

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    warnings.warn("gradio not available. Install for alternative dashboard interface.")

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for real-time performance dashboard."""
    
    # Dashboard settings
    enable_web_dashboard: bool = True
    enable_gradio_interface: bool = True
    dashboard_port: int = 8050
    dashboard_host: str = "localhost"
    
    # Monitoring settings
    update_interval: float = 1.0  # 1 second
    max_data_points: int = 1000
    enable_real_time_updates: bool = True
    enable_historical_data: bool = True
    
    # Visualization settings
    enable_interactive_charts: bool = True
    enable_performance_metrics: bool = True
    enable_system_monitoring: bool = True
    enable_gpu_monitoring: bool = True
    
    # Alerting settings
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "gpu_utilization": 90.0,
        "gpu_temperature": 80.0
    })
    
    # Export settings
    enable_data_export: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "html"])
    auto_export_interval: int = 300  # 5 minutes


class PerformanceDataCollector:
    """Collects and manages performance data for the dashboard."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.performance_data = defaultdict(lambda: deque(maxlen=config.max_data_points))
        self.system_metrics = defaultdict(lambda: deque(maxlen=config.max_data_points))
        self.gpu_metrics = defaultdict(lambda: deque(maxlen=config.max_data_points))
        self.alert_history = deque(maxlen=1000)
        
    def collect_performance_data(self, model_name: str, metrics: Dict[str, Any]):
        """Collect performance data for a specific model."""
        try:
            timestamp = time.time()
            
            # Store performance metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.performance_data[f"{model_name}_{metric_name}"].append({
                        "timestamp": timestamp,
                        "value": value,
                        "model": model_name
                    })
            
            # Check for alerts
            self._check_alerts(model_name, metrics)
            
        except Exception as e:
            logger.error(f"Performance data collection failed: {e}")
    
    def collect_system_metrics(self, metrics: Dict[str, Any]):
        """Collect system-level performance metrics."""
        try:
            timestamp = time.time()
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.system_metrics[metric_name].append({
                        "timestamp": timestamp,
                        "value": value
                    })
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def collect_gpu_metrics(self, gpu_id: str, metrics: Dict[str, Any]):
        """Collect GPU-specific performance metrics."""
        try:
            timestamp = time.time()
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.gpu_metrics[f"gpu_{gpu_id}_{metric_name}"].append({
                        "timestamp": timestamp,
                        "value": value,
                        "gpu_id": gpu_id
                    })
            
        except Exception as e:
            logger.error(f"GPU metrics collection failed: {e}")
    
    def _check_alerts(self, model_name: str, metrics: Dict[str, Any]):
        """Check for performance alerts."""
        try:
            for metric_name, threshold in self.config.alert_thresholds.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, (int, float)) and value > threshold:
                        alert = {
                            "timestamp": time.time(),
                            "type": "performance_alert",
                            "model": model_name,
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold,
                            "severity": "high" if value > threshold * 1.2 else "medium"
                        }
                        
                        self.alert_history.append(alert)
                        logger.warning(f"Performance alert: {model_name} {metric_name} = {value} (threshold: {threshold})")
            
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            summary = {
                "timestamp": time.time(),
                "total_models": len(set(data[0]["model"] for data in self.performance_data.values() if data)),
                "total_metrics": len(self.performance_data),
                "system_metrics_count": len(self.system_metrics),
                "gpu_metrics_count": len(self.gpu_metrics),
                "alerts_count": len(self.alert_history),
                "recent_alerts": list(self.alert_history)[-10:] if self.alert_history else []
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {"error": str(e)}
    
    def get_metric_data(self, metric_name: str, window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metric data within a time window."""
        try:
            cutoff_time = time.time() - (window_minutes * 60)
            
            if metric_name in self.performance_data:
                data = self.performance_data[metric_name]
            elif metric_name in self.system_metrics:
                data = self.system_metrics[metric_name]
            elif metric_name in self.gpu_metrics:
                data = self.gpu_metrics[metric_name]
            else:
                return []
            
            # Filter by time window
            filtered_data = [
                entry for entry in data
                if entry["timestamp"] > cutoff_time
            ]
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Metric data retrieval failed: {e}")
            return []


class DashboardVisualizer:
    """Creates and manages dashboard visualizations."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.chart_templates = self._create_chart_templates()
        
    def _create_chart_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create chart templates for different metric types."""
        return {
            "line_chart": {
                "layout": {
                    "title": "Performance Metrics Over Time",
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Value"},
                    "template": "plotly_white"
                }
            },
            "bar_chart": {
                "layout": {
                    "title": "Performance Comparison",
                    "xaxis": {"title": "Models"},
                    "yaxis": {"title": "Performance"},
                    "template": "plotly_white"
                }
            },
            "gauge_chart": {
                "layout": {
                    "title": "Current Performance",
                    "template": "plotly_white"
                }
            }
        }
    
    def create_performance_chart(self, metric_name: str, data: List[Dict[str, Any]], 
                                chart_type: str = "line") -> go.Figure:
        """Create a performance chart."""
        try:
            if not data:
                return go.Figure()
            
            # Extract data
            timestamps = [entry["timestamp"] for entry in data]
            values = [entry["value"] for entry in data]
            
            # Convert timestamps to datetime
            datetime_timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
            
            if chart_type == "line":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=datetime_timestamps,
                    y=values,
                    mode='lines+markers',
                    name=metric_name,
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"{metric_name} Over Time",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    template="plotly_white",
                    hovermode='x unified'
                )
                
            elif chart_type == "bar":
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=datetime_timestamps,
                    y=values,
                    name=metric_name,
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title=f"{metric_name} Distribution",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    template="plotly_white"
                )
                
            else:
                fig = go.Figure()
            
            return fig
            
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")
            return go.Figure()
    
    def create_system_overview_chart(self, system_data: Dict[str, List[Dict[str, Any]]]) -> go.Figure:
        """Create a system overview chart."""
        try:
            if not system_data:
                return go.Figure()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("CPU Usage", "Memory Usage", "GPU Utilization", "GPU Temperature"),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # CPU Usage Gauge
            if "cpu_usage" in system_data and system_data["cpu_usage"]:
                cpu_value = system_data["cpu_usage"][-1]["value"]
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=cpu_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CPU Usage %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "red"}]}
                ), row=1, col=1)
            
            # Memory Usage Gauge
            if "memory_usage" in system_data and system_data["memory_usage"]:
                memory_value = system_data["memory_usage"][-1]["value"]
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=memory_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Memory Usage %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 85], 'color': "yellow"},
                                    {'range': [85, 100], 'color': "red"}]}
                ), row=1, col=2)
            
            # GPU Utilization Gauge
            if "gpu_utilization" in system_data and system_data["gpu_utilization"]:
                gpu_value = system_data["gpu_utilization"][-1]["value"]
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=gpu_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "GPU Utilization %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkred"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "red"}]}
                ), row=2, col=1)
            
            # GPU Temperature Gauge
            if "gpu_temperature" in system_data and system_data["gpu_temperature"]:
                temp_value = system_data["gpu_temperature"][-1]["value"]
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=temp_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "GPU Temperature °C"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkorange"},
                           'steps': [{'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "red"}]}
                ), row=2, col=2)
            
            fig.update_layout(
                title="System Performance Overview",
                template="plotly_white",
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"System overview chart creation failed: {e}")
            return go.Figure()
    
    def create_alert_summary_chart(self, alerts: List[Dict[str, Any]]) -> go.Figure:
        """Create an alert summary chart."""
        try:
            if not alerts:
                return go.Figure()
            
            # Count alerts by type and severity
            alert_counts = defaultdict(lambda: defaultdict(int))
            for alert in alerts:
                alert_type = alert.get("type", "unknown")
                severity = alert.get("severity", "unknown")
                alert_counts[alert_type][severity] += 1
            
            # Create bar chart
            fig = go.Figure()
            
            for alert_type, severities in alert_counts.items():
                for severity, count in severities.items():
                    fig.add_trace(go.Bar(
                        name=f"{alert_type}_{severity}",
                        x=[alert_type],
                        y=[count],
                        text=[count],
                        textposition='auto',
                        marker_color='red' if severity == 'high' else 'orange' if severity == 'medium' else 'yellow'
                    ))
            
            fig.update_layout(
                title="Performance Alerts Summary",
                xaxis_title="Alert Type",
                yaxis_title="Count",
                template="plotly_white",
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Alert summary chart creation failed: {e}")
            return go.Figure()


class WebDashboard:
    """Web-based dashboard using Dash."""
    
    def __init__(self, config: DashboardConfig, data_collector: PerformanceDataCollector,
                 visualizer: DashboardVisualizer):
        self.config = config
        self.data_collector = data_collector
        self.visualizer = visualizer
        
        if not DASH_AVAILABLE:
            logger.error("Dash not available. Web dashboard disabled.")
            return
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout."""
        try:
            self.app.layout = html.Div([
                html.H1("HeyGen AI Enterprise - Performance Dashboard", 
                       style={'textAlign': 'center', 'color': 'darkblue'}),
                
                # System Overview
                html.Div([
                    html.H2("System Performance Overview"),
                    dcc.Graph(id='system-overview-chart'),
                    dcc.Interval(
                        id='system-interval',
                        interval=self.config.update_interval * 1000,
                        n_intervals=0
                    )
                ]),
                
                # Performance Metrics
                html.Div([
                    html.H2("Performance Metrics"),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[
                            {'label': 'Inference Time', 'value': 'inference_time'},
                            {'label': 'Memory Usage', 'value': 'memory_usage'},
                            {'label': 'GPU Utilization', 'value': 'gpu_utilization'},
                            {'label': 'Throughput', 'value': 'throughput'}
                        ],
                        value='inference_time'
                    ),
                    dcc.Graph(id='performance-chart'),
                    dcc.Interval(
                        id='performance-interval',
                        interval=self.config.update_interval * 1000,
                        n_intervals=0
                    )
                ]),
                
                # Alerts
                html.Div([
                    html.H2("Performance Alerts"),
                    dcc.Graph(id='alerts-chart'),
                    html.Div(id='alerts-list')
                ]),
                
                # Controls
                html.Div([
                    html.H3("Dashboard Controls"),
                    html.Button("Export Data", id='export-btn', n_clicks=0),
                    html.Button("Refresh", id='refresh-btn', n_clicks=0),
                    dcc.Interval(
                        id='refresh-interval',
                        interval=5000,  # 5 seconds
                        n_intervals=0
                    )
                ])
            ])
            
        except Exception as e:
            logger.error(f"Dashboard layout setup failed: {e}")
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        try:
            @self.app.callback(
                Output('system-overview-chart', 'figure'),
                Input('system-interval', 'n_intervals')
            )
            def update_system_overview(n):
                try:
                    # Get system metrics
                    system_data = {}
                    for metric_name in ['cpu_usage', 'memory_usage', 'gpu_utilization', 'gpu_temperature']:
                        data = self.data_collector.get_metric_data(metric_name, window_minutes=5)
                        if data:
                            system_data[metric_name] = data
                    
                    return self.visualizer.create_system_overview_chart(system_data)
                    
                except Exception as e:
                    logger.error(f"System overview update failed: {e}")
                    return go.Figure()
            
            @self.app.callback(
                Output('performance-chart', 'figure'),
                [Input('performance-interval', 'n_intervals'),
                 Input('metric-dropdown', 'value')]
            )
            def update_performance_chart(n, metric_name):
                try:
                    if not metric_name:
                        return go.Figure()
                    
                    # Get performance data
                    data = self.data_collector.get_metric_data(metric_name, window_minutes=30)
                    
                    return self.visualizer.create_performance_chart(metric_name, data)
                    
                except Exception as e:
                    logger.error(f"Performance chart update failed: {e}")
                    return go.Figure()
            
            @self.app.callback(
                [Output('alerts-chart', 'figure'),
                 Output('alerts-list', 'children')],
                Input('refresh-interval', 'n_intervals')
            )
            def update_alerts(n):
                try:
                    # Get recent alerts
                    summary = self.data_collector.get_performance_summary()
                    recent_alerts = summary.get("recent_alerts", [])
                    
                    # Create alert chart
                    alert_chart = self.visualizer.create_alert_summary_chart(recent_alerts)
                    
                    # Create alert list
                    alert_list = []
                    for alert in recent_alerts[-5:]:  # Show last 5 alerts
                        alert_list.append(html.Div([
                            html.Span(f"{alert.get('type', 'Unknown')}: ", 
                                    style={'fontWeight': 'bold'}),
                            html.Span(f"{alert.get('description', 'No description')}"),
                            html.Br(),
                            html.Small(f"Severity: {alert.get('severity', 'Unknown')} | "
                                     f"Time: {datetime.fromtimestamp(alert.get('timestamp', 0))}")
                        ], style={'margin': '10px', 'padding': '10px', 'border': '1px solid #ddd'}))
                    
                    return alert_chart, alert_list
                    
                except Exception as e:
                    logger.error(f"Alerts update failed: {e}")
                    return go.Figure(), []
            
            @self.app.callback(
                Output('export-btn', 'n_clicks'),
                Input('export-btn', 'n_clicks')
            )
            def export_data(n_clicks):
                if n_clicks and n_clicks > 0:
                    try:
                        self._export_dashboard_data()
                    except Exception as e:
                        logger.error(f"Data export failed: {e}")
                return 0
            
        except Exception as e:
            logger.error(f"Dashboard callbacks setup failed: {e}")
    
    def _export_dashboard_data(self):
        """Export dashboard data."""
        try:
            # Get all data
            summary = self.data_collector.get_performance_summary()
            
            # Create export directory
            export_dir = Path("dashboard_exports")
            export_dir.mkdir(exist_ok=True)
            
            # Export timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export JSON
            json_file = export_dir / f"dashboard_data_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to {json_file}")
            
        except Exception as e:
            logger.error(f"Dashboard data export failed: {e}")
    
    def run(self, debug: bool = False):
        """Run the web dashboard."""
        try:
            if not DASH_AVAILABLE:
                logger.error("Cannot run web dashboard: Dash not available")
                return
            
            self.app.run_server(
                debug=debug,
                host=self.config.dashboard_host,
                port=self.config.dashboard_port
            )
            
        except Exception as e:
            logger.error(f"Web dashboard run failed: {e}")


class GradioDashboard:
    """Alternative dashboard using Gradio."""
    
    def __init__(self, config: DashboardConfig, data_collector: PerformanceDataCollector,
                 visualizer: DashboardVisualizer):
        self.config = config
        self.data_collector = data_collector
        self.visualizer = visualizer
        
        if not GRADIO_AVAILABLE:
            logger.error("Gradio not available. Gradio dashboard disabled.")
            return
        
        self.interface = None
        self.setup_interface()
        
    def setup_interface(self):
        """Setup Gradio interface."""
        try:
            with gr.Blocks(title="HeyGen AI Performance Dashboard") as interface:
                gr.Markdown("# 🚀 HeyGen AI Enterprise - Performance Dashboard")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## System Overview")
                        system_chart = gr.Plot(label="System Performance")
                        
                        gr.Markdown("## Performance Metrics")
                        metric_dropdown = gr.Dropdown(
                            choices=["inference_time", "memory_usage", "gpu_utilization", "throughput"],
                            value="inference_time",
                            label="Select Metric"
                        )
                        performance_chart = gr.Plot(label="Performance Over Time")
                        
                    with gr.Column():
                        gr.Markdown("## Performance Alerts")
                        alerts_chart = gr.Plot(label="Alerts Summary")
                        alerts_list = gr.JSON(label="Recent Alerts")
                        
                        gr.Markdown("## Dashboard Controls")
                        refresh_btn = gr.Button("🔄 Refresh Dashboard")
                        export_btn = gr.Button("📊 Export Data")
                
                # Setup event handlers
                refresh_btn.click(
                    fn=self._refresh_dashboard,
                    outputs=[system_chart, performance_chart, alerts_chart, alerts_list]
                )
                
                metric_dropdown.change(
                    fn=self._update_performance_chart,
                    inputs=[metric_dropdown],
                    outputs=[performance_chart]
                )
                
                export_btn.click(
                    fn=self._export_data,
                    outputs=[]
                )
                
                # Auto-refresh
                interface.load(
                    fn=self._refresh_dashboard,
                    outputs=[system_chart, performance_chart, alerts_chart, alerts_list]
                )
            
            self.interface = interface
            
        except Exception as e:
            logger.error(f"Gradio interface setup failed: {e}")
    
    def _refresh_dashboard(self):
        """Refresh all dashboard components."""
        try:
            # System overview
            system_data = {}
            for metric_name in ['cpu_usage', 'memory_usage', 'gpu_utilization', 'gpu_temperature']:
                data = self.data_collector.get_metric_data(metric_name, window_minutes=5)
                if data:
                    system_data[metric_name] = data
            
            system_chart = self.visualizer.create_system_overview_chart(system_data)
            
            # Performance chart (default metric)
            default_data = self.data_collector.get_metric_data("inference_time", window_minutes=30)
            performance_chart = self.visualizer.create_performance_chart("inference_time", default_data)
            
            # Alerts
            summary = self.data_collector.get_performance_summary()
            recent_alerts = summary.get("recent_alerts", [])
            alerts_chart = self.visualizer.create_alert_summary_chart(recent_alerts)
            alerts_list = recent_alerts
            
            return system_chart, performance_chart, alerts_chart, alerts_list
            
        except Exception as e:
            logger.error(f"Dashboard refresh failed: {e}")
            return None, None, None, None
    
    def _update_performance_chart(self, metric_name: str):
        """Update performance chart for specific metric."""
        try:
            data = self.data_collector.get_metric_data(metric_name, window_minutes=30)
            return self.visualizer.create_performance_chart(metric_name, data)
            
        except Exception as e:
            logger.error(f"Performance chart update failed: {e}")
            return None
    
    def _export_data(self):
        """Export dashboard data."""
        try:
            summary = self.data_collector.get_performance_summary()
            
            # Create export directory
            export_dir = Path("dashboard_exports")
            export_dir.mkdir(exist_ok=True)
            
            # Export timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export JSON
            json_file = export_dir / f"gradio_dashboard_data_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Gradio dashboard data exported to {json_file}")
            
        except Exception as e:
            logger.error(f"Gradio dashboard data export failed: {e}")
    
    def launch(self, **kwargs):
        """Launch the Gradio dashboard."""
        try:
            if not self.interface:
                logger.error("Cannot launch Gradio dashboard: interface not setup")
                return
            
            self.interface.launch(
                server_name=self.config.dashboard_host,
                server_port=self.config.dashboard_port,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Gradio dashboard launch failed: {e}")


class RealTimePerformanceDashboard:
    """Main dashboard system orchestrating all components."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.dashboard")
        
        # Initialize components
        self.data_collector = PerformanceDataCollector(config)
        self.visualizer = DashboardVisualizer(config)
        
        # Dashboard interfaces
        self.web_dashboard = None
        self.gradio_dashboard = None
        
        if config.enable_web_dashboard:
            self.web_dashboard = WebDashboard(config, self.data_collector, self.visualizer)
        
        if config.enable_gradio_interface:
            self.gradio_dashboard = GradioDashboard(config, self.data_collector, self.visualizer)
        
        # Dashboard state
        self.dashboard_active = False
        self.auto_refresh_thread = None
        
    def start_dashboard(self, interface: str = "web"):
        """Start the performance dashboard."""
        try:
            if interface == "web" and self.web_dashboard:
                self.logger.info("Starting web dashboard...")
                self.web_dashboard.run(debug=False)
            elif interface == "gradio" and self.gradio_dashboard:
                self.logger.info("Starting Gradio dashboard...")
                self.gradio_dashboard.launch()
            else:
                self.logger.error(f"Dashboard interface '{interface}' not available")
                
        except Exception as e:
            self.logger.error(f"Dashboard start failed: {e}")
    
    def collect_model_performance(self, model_name: str, metrics: Dict[str, Any]):
        """Collect performance data for a model."""
        self.data_collector.collect_performance_data(model_name, metrics)
    
    def collect_system_metrics(self, metrics: Dict[str, Any]):
        """Collect system performance metrics."""
        self.data_collector.collect_system_metrics(metrics)
    
    def collect_gpu_metrics(self, gpu_id: str, metrics: Dict[str, Any]):
        """Collect GPU performance metrics."""
        self.data_collector.collect_gpu_metrics(gpu_id, metrics)
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary."""
        return self.data_collector.get_performance_summary()
    
    def export_dashboard_data(self, format: str = "json") -> str:
        """Export dashboard data in specified format."""
        try:
            summary = self.data_collector.get_performance_summary()
            
            # Create export directory
            export_dir = Path("dashboard_exports")
            export_dir.mkdir(exist_ok=True)
            
            # Export timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == "json":
                export_file = export_dir / f"dashboard_data_{timestamp}.json"
                with open(export_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
            
            elif format == "csv":
                export_file = export_dir / f"dashboard_data_{timestamp}.csv"
                # TODO: Implement CSV export
                pass
            
            elif format == "html":
                export_file = export_dir / f"dashboard_data_{timestamp}.html"
                # TODO: Implement HTML export
                pass
            
            logger.info(f"Dashboard data exported to {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"Dashboard data export failed: {e}")
            return ""


# Factory functions
def create_performance_dashboard(config: Optional[DashboardConfig] = None) -> RealTimePerformanceDashboard:
    """Create a performance dashboard."""
    if config is None:
        config = DashboardConfig()
    
    return RealTimePerformanceDashboard(config)


def create_web_dashboard_config() -> DashboardConfig:
    """Create configuration for web-based dashboard."""
    return DashboardConfig(
        enable_web_dashboard=True,
        enable_gradio_interface=False,
        dashboard_port=8050,
        dashboard_host="localhost",
        update_interval=1.0,
        enable_real_time_updates=True
    )


def create_gradio_dashboard_config() -> DashboardConfig:
    """Create configuration for Gradio dashboard."""
    return DashboardConfig(
        enable_web_dashboard=False,
        enable_gradio_interface=True,
        dashboard_port=7860,
        dashboard_host="localhost",
        update_interval=2.0,
        enable_real_time_updates=True
    )


if __name__ == "__main__":
    # Test the performance dashboard
    config = create_web_dashboard_config()
    dashboard = create_performance_dashboard(config)
    
    # Collect some sample data
    dashboard.collect_model_performance("test_model", {
        "inference_time": 25.5,
        "memory_usage": 1024.0,
        "gpu_utilization": 85.0
    })
    
    dashboard.collect_system_metrics({
        "cpu_usage": 65.0,
        "memory_usage": 75.0,
        "gpu_utilization": 80.0,
        "gpu_temperature": 70.0
    })
    
    # Get summary
    summary = dashboard.get_dashboard_summary()
    print(f"Dashboard summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Start dashboard
    print("Starting web dashboard...")
    dashboard.start_dashboard("web")
