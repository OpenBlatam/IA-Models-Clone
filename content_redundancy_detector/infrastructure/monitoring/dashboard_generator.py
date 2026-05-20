"""
Advanced Dashboard Generator - Real-time monitoring dashboards
Production-ready dashboard generation and visualization
"""

import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    title: str
    refresh_interval: int = 30  # seconds
    theme: str = "dark"
    layout: str = "grid"
    widgets: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Widget:
    """Dashboard widget definition"""
    id: str
    type: str
    title: str
    data_source: str
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)

class DashboardGenerator:
    """Advanced dashboard generation system"""
    
    def __init__(self):
        self.dashboards: Dict[str, DashboardConfig] = {}
        self.widgets: Dict[str, Widget] = {}
        self.data_sources: Dict[str, Any] = {}
        
        # Register default widgets
        self._register_default_widgets()

    def _register_default_widgets(self):
        """Register default dashboard widgets"""
        # System metrics widget
        self.add_widget(Widget(
            id="system_metrics",
            type="metrics_grid",
            title="System Metrics",
            data_source="system_metrics",
            config={
                "metrics": ["cpu", "memory", "disk", "network"],
                "show_trends": True,
                "update_interval": 10
            }
        ))
        
        # Performance chart widget
        self.add_widget(Widget(
            id="performance_chart",
            type="line_chart",
            title="Performance Trends",
            data_source="performance_metrics",
            config={
                "metrics": ["response_time", "throughput", "error_rate"],
                "time_range": "1h",
                "show_legend": True
            }
        ))
        
        # Health status widget
        self.add_widget(Widget(
            id="health_status",
            type="health_grid",
            title="Service Health",
            data_source="health_metrics",
            config={
                "show_uptime": True,
                "show_version": True,
                "color_coding": True
            }
        ))
        
        # Alerts widget
        self.add_widget(Widget(
            id="alerts",
            type="alerts_list",
            title="Active Alerts",
            data_source="alert_metrics",
            config={
                "max_items": 10,
                "show_timestamps": True,
                "group_by_level": True
            }
        ))

    def add_dashboard(self, name: str, config: DashboardConfig):
        """Add a dashboard configuration"""
        self.dashboards[name] = config

    def add_widget(self, widget: Widget):
        """Add a widget"""
        self.widgets[widget.id] = widget

    def register_data_source(self, name: str, source: Any):
        """Register a data source"""
        self.data_sources[name] = source

    def generate_dashboard(self, name: str) -> Dict[str, Any]:
        """Generate dashboard data"""
        if name not in self.dashboards:
            return {"error": f"Dashboard '{name}' not found"}
        
        config = self.dashboards[name]
        
        # Generate widget data
        widget_data = {}
        for widget_config in config.widgets:
            widget_id = widget_config.get("id")
            if widget_id and widget_id in self.widgets:
                widget = self.widgets[widget_id]
                data = self._generate_widget_data(widget)
                widget_data[widget_id] = data
        
        return {
            "title": config.title,
            "refresh_interval": config.refresh_interval,
            "theme": config.theme,
            "layout": config.layout,
            "timestamp": time.time(),
            "widgets": widget_data
        }

    def _generate_widget_data(self, widget: Widget) -> Dict[str, Any]:
        """Generate data for a specific widget"""
        data_source = self.data_sources.get(widget.data_source)
        if not data_source:
            return {"error": f"Data source '{widget.data_source}' not found"}
        
        if widget.type == "metrics_grid":
            return self._generate_metrics_grid(widget, data_source)
        elif widget.type == "line_chart":
            return self._generate_line_chart(widget, data_source)
        elif widget.type == "health_grid":
            return self._generate_health_grid(widget, data_source)
        elif widget.type == "alerts_list":
            return self._generate_alerts_list(widget, data_source)
        elif widget.type == "gauge":
            return self._generate_gauge(widget, data_source)
        elif widget.type == "table":
            return self._generate_table(widget, data_source)
        else:
            return {"error": f"Unknown widget type: {widget.type}"}

    def _generate_metrics_grid(self, widget: Widget, data_source: Any) -> Dict[str, Any]:
        """Generate metrics grid widget data"""
        config = widget.config
        metrics = config.get("metrics", [])
        
        data = {
            "type": "metrics_grid",
            "title": widget.title,
            "metrics": []
        }
        
        for metric_name in metrics:
            if hasattr(data_source, 'get_metric'):
                value = data_source.get_metric(metric_name)
            elif hasattr(data_source, 'get_stats'):
                stats = data_source.get_stats()
                value = stats.get(metric_name, 0)
            else:
                value = 0
            
            # Format value based on metric type
            if metric_name in ["memory", "disk"]:
                formatted_value = f"{value:.1f}%"
            elif metric_name in ["cpu"]:
                formatted_value = f"{value:.1f}%"
            else:
                formatted_value = str(value)
            
            data["metrics"].append({
                "name": metric_name,
                "value": formatted_value,
                "raw_value": value
            })
        
        return data

    def _generate_line_chart(self, widget: Widget, data_source: Any) -> Dict[str, Any]:
        """Generate line chart widget data"""
        config = widget.config
        metrics = config.get("metrics", [])
        time_range = config.get("time_range", "1h")
        
        data = {
            "type": "line_chart",
            "title": widget.title,
            "series": [],
            "x_axis": "time",
            "y_axis": "value"
        }
        
        # Calculate time range
        end_time = time.time()
        if time_range == "1h":
            start_time = end_time - 3600
        elif time_range == "6h":
            start_time = end_time - 21600
        elif time_range == "24h":
            start_time = end_time - 86400
        else:
            start_time = end_time - 3600
        
        for metric_name in metrics:
            if hasattr(data_source, 'get_metric_history'):
                history = data_source.get_metric_history(metric_name, start_time, end_time)
            else:
                # Generate mock data
                history = self._generate_mock_time_series(start_time, end_time, 60)
            
            data["series"].append({
                "name": metric_name,
                "data": history,
                "color": self._get_metric_color(metric_name)
            })
        
        return data

    def _generate_health_grid(self, widget: Widget, data_source: Any) -> Dict[str, Any]:
        """Generate health grid widget data"""
        config = widget.config
        
        if hasattr(data_source, 'get_health_status'):
            health_data = data_source.get_health_status()
        else:
            health_data = {"status": "unknown"}
        
        data = {
            "type": "health_grid",
            "title": widget.title,
            "status": health_data.get("status", "unknown"),
            "uptime": health_data.get("uptime", 0),
            "version": health_data.get("version", "unknown"),
            "checks": health_data.get("checks", [])
        }
        
        return data

    def _generate_alerts_list(self, widget: Widget, data_source: Any) -> Dict[str, Any]:
        """Generate alerts list widget data"""
        config = widget.config
        max_items = config.get("max_items", 10)
        
        if hasattr(data_source, 'get_active_alerts'):
            alerts = data_source.get_active_alerts()
        else:
            alerts = []
        
        # Limit number of alerts
        alerts = alerts[:max_items]
        
        data = {
            "type": "alerts_list",
            "title": widget.title,
            "alerts": []
        }
        
        for alert in alerts:
            data["alerts"].append({
                "id": alert.id,
                "level": alert.level.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "rule_name": alert.rule_name
            })
        
        return data

    def _generate_gauge(self, widget: Widget, data_source: Any) -> Dict[str, Any]:
        """Generate gauge widget data"""
        config = widget.config
        metric_name = config.get("metric", "value")
        min_value = config.get("min_value", 0)
        max_value = config.get("max_value", 100)
        
        if hasattr(data_source, 'get_metric'):
            value = data_source.get_metric(metric_name)
        else:
            value = 0
        
        # Calculate percentage
        percentage = ((value - min_value) / (max_value - min_value)) * 100
        percentage = max(0, min(100, percentage))
        
        data = {
            "type": "gauge",
            "title": widget.title,
            "value": value,
            "percentage": percentage,
            "min_value": min_value,
            "max_value": max_value,
            "unit": config.get("unit", "")
        }
        
        return data

    def _generate_table(self, widget: Widget, data_source: Any) -> Dict[str, Any]:
        """Generate table widget data"""
        config = widget.config
        columns = config.get("columns", [])
        
        if hasattr(data_source, 'get_table_data'):
            table_data = data_source.get_table_data()
        else:
            table_data = []
        
        data = {
            "type": "table",
            "title": widget.title,
            "columns": columns,
            "rows": table_data
        }
        
        return data

    def _generate_mock_time_series(self, start_time: float, end_time: float, points: int) -> List[Dict[str, Any]]:
        """Generate mock time series data"""
        import random
        
        data = []
        time_step = (end_time - start_time) / points
        
        for i in range(points):
            timestamp = start_time + (i * time_step)
            value = random.uniform(0, 100)
            
            data.append({
                "timestamp": timestamp,
                "value": value
            })
        
        return data

    def _get_metric_color(self, metric_name: str) -> str:
        """Get color for metric"""
        colors = {
            "cpu": "#ff6b6b",
            "memory": "#4ecdc4",
            "disk": "#45b7d1",
            "network": "#96ceb4",
            "response_time": "#feca57",
            "throughput": "#ff9ff3",
            "error_rate": "#ff3838"
        }
        return colors.get(metric_name, "#95a5a6")

    def generate_html_dashboard(self, name: str) -> str:
        """Generate HTML dashboard"""
        dashboard_data = self.generate_dashboard(name)
        
        if "error" in dashboard_data:
            return f"<html><body><h1>Error: {dashboard_data['error']}</h1></body></html>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard_data['title']}</title>
            <meta http-equiv="refresh" content="{dashboard_data['refresh_interval']}">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #1a1a1a;
                    color: #ffffff;
                }}
                .dashboard {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .widget {{
                    background-color: #2d2d2d;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }}
                .widget h3 {{
                    margin-top: 0;
                    color: #4ecdc4;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                }}
                .metric {{
                    background-color: #3d3d3d;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4ecdc4;
                }}
                .metric-name {{
                    font-size: 14px;
                    color: #cccccc;
                    margin-top: 5px;
                }}
                .alerts-list {{
                    max-height: 400px;
                    overflow-y: auto;
                }}
                .alert {{
                    background-color: #3d3d3d;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #ff6b6b;
                }}
                .alert.warning {{
                    border-left-color: #ffaa00;
                }}
                .alert.error {{
                    border-left-color: #ff6b6b;
                }}
                .alert.critical {{
                    border-left-color: #ff0000;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>{dashboard_data['title']}</h1>
                    <p>Last updated: {datetime.fromtimestamp(dashboard_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
        """
        
        # Generate widget HTML
        for widget_id, widget_data in dashboard_data.get("widgets", {}).items():
            html += self._generate_widget_html(widget_data)
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html

    def _generate_widget_html(self, widget_data: Dict[str, Any]) -> str:
        """Generate HTML for a widget"""
        widget_type = widget_data.get("type", "unknown")
        
        if widget_type == "metrics_grid":
            return self._generate_metrics_grid_html(widget_data)
        elif widget_type == "alerts_list":
            return self._generate_alerts_list_html(widget_data)
        elif widget_type == "health_grid":
            return self._generate_health_grid_html(widget_data)
        else:
            return f'<div class="widget"><h3>{widget_data.get("title", "Unknown Widget")}</h3><p>Widget type not supported: {widget_type}</p></div>'

    def _generate_metrics_grid_html(self, widget_data: Dict[str, Any]) -> str:
        """Generate HTML for metrics grid widget"""
        html = f'<div class="widget"><h3>{widget_data.get("title", "Metrics")}</h3><div class="metrics-grid">'
        
        for metric in widget_data.get("metrics", []):
            html += f'''
                <div class="metric">
                    <div class="metric-value">{metric.get("value", "0")}</div>
                    <div class="metric-name">{metric.get("name", "Unknown")}</div>
                </div>
            '''
        
        html += '</div></div>'
        return html

    def _generate_alerts_list_html(self, widget_data: Dict[str, Any]) -> str:
        """Generate HTML for alerts list widget"""
        html = f'<div class="widget"><h3>{widget_data.get("title", "Alerts")}</h3><div class="alerts-list">'
        
        alerts = widget_data.get("alerts", [])
        if not alerts:
            html += '<p>No active alerts</p>'
        else:
            for alert in alerts:
                level = alert.get("level", "info")
                html += f'''
                    <div class="alert {level}">
                        <strong>{alert.get("rule_name", "Unknown")}</strong><br>
                        {alert.get("message", "No message")}<br>
                        <small>{datetime.fromtimestamp(alert.get("timestamp", 0)).strftime('%H:%M:%S')}</small>
                    </div>
                '''
        
        html += '</div></div>'
        return html

    def _generate_health_grid_html(self, widget_data: Dict[str, Any]) -> str:
        """Generate HTML for health grid widget"""
        status = widget_data.get("status", "unknown")
        uptime = widget_data.get("uptime", 0)
        version = widget_data.get("version", "unknown")
        
        status_color = {
            "healthy": "#4ecdc4",
            "degraded": "#ffaa00",
            "unhealthy": "#ff6b6b",
            "critical": "#ff0000"
        }.get(status, "#cccccc")
        
        html = f'''
            <div class="widget">
                <h3>{widget_data.get("title", "Health Status")}</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div class="metric">
                        <div class="metric-value" style="color: {status_color};">{status.upper()}</div>
                        <div class="metric-name">Status</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{uptime:.1f}s</div>
                        <div class="metric-name">Uptime</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{version}</div>
                        <div class="metric-name">Version</div>
                    </div>
                </div>
            </div>
        '''
        
        return html

    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards"""
        return [
            {
                "name": name,
                "title": config.title,
                "refresh_interval": config.refresh_interval,
                "widget_count": len(config.widgets)
            }
            for name, config in self.dashboards.items()
        ]





