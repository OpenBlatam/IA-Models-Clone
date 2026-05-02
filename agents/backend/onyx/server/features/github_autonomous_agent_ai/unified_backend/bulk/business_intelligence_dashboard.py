"""
BUL Business Intelligence Dashboard
==================================

Advanced business intelligence dashboard for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
import yaml
import sqlite3
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BIMetricType(Enum):
    """Business Intelligence metric types."""
    KPI = "kpi"
    TREND = "trend"
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    FORECAST = "forecast"
    BENCHMARK = "benchmark"

class BIVisualizationType(Enum):
    """Business Intelligence visualization types."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    DASHBOARD = "dashboard"
    TABLE = "table"
    GAUGE = "gauge"

@dataclass
class BIMetric:
    """Business Intelligence metric definition."""
    id: str
    name: str
    description: str
    metric_type: BIMetricType
    data_source: str
    calculation: str
    unit: str
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    created_at: datetime = None

@dataclass
class BIDashboard:
    """Business Intelligence dashboard definition."""
    id: str
    name: str
    description: str
    metrics: List[str]
    layout: Dict[str, Any]
    filters: List[Dict[str, Any]]
    refresh_interval: int = 300  # 5 minutes
    created_at: datetime = None

class BusinessIntelligenceDashboard:
    """Advanced business intelligence dashboard for BUL system."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.metrics = {}
        self.dashboards = {}
        self.data_sources = {}
        self.init_bi_environment()
        self.load_metrics()
        self.load_dashboards()
        self.setup_data_sources()
    
    def init_bi_environment(self):
        """Initialize business intelligence environment."""
        print("📊 Initializing business intelligence environment...")
        
        # Create BI directories
        self.bi_dir = Path("business_intelligence")
        self.bi_dir.mkdir(exist_ok=True)
        
        self.dashboards_dir = Path("bi_dashboards")
        self.dashboards_dir.mkdir(exist_ok=True)
        
        self.reports_dir = Path("bi_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.init_bi_database()
        
        print("✅ Business intelligence environment initialized")
    
    def init_bi_database(self):
        """Initialize business intelligence database."""
        conn = sqlite3.connect("business_intelligence.db")
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bi_metrics (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                metric_type TEXT,
                data_source TEXT,
                calculation TEXT,
                unit TEXT,
                target_value REAL,
                threshold_warning REAL,
                threshold_critical REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bi_dashboards (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                metrics TEXT,
                layout TEXT,
                filters TEXT,
                refresh_interval INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bi_data_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_id TEXT,
                value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (metric_id) REFERENCES bi_metrics (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_metrics(self):
        """Load existing business intelligence metrics."""
        conn = sqlite3.connect("business_intelligence.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM bi_metrics")
        rows = cursor.fetchall()
        
        for row in rows:
            metric = BIMetric(
                id=row[0],
                name=row[1],
                description=row[2],
                metric_type=BIMetricType(row[3]),
                data_source=row[4],
                calculation=row[5],
                unit=row[6],
                target_value=row[7],
                threshold_warning=row[8],
                threshold_critical=row[9],
                created_at=datetime.fromisoformat(row[10])
            )
            self.metrics[metric.id] = metric
        
        conn.close()
        
        # Create default metrics if none exist
        if not self.metrics:
            self.create_default_metrics()
        
        print(f"✅ Loaded {len(self.metrics)} business intelligence metrics")
    
    def load_dashboards(self):
        """Load existing dashboards."""
        conn = sqlite3.connect("business_intelligence.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM bi_dashboards")
        rows = cursor.fetchall()
        
        for row in rows:
            dashboard = BIDashboard(
                id=row[0],
                name=row[1],
                description=row[2],
                metrics=json.loads(row[3]),
                layout=json.loads(row[4]),
                filters=json.loads(row[5]),
                refresh_interval=row[6],
                created_at=datetime.fromisoformat(row[7])
            )
            self.dashboards[dashboard.id] = dashboard
        
        conn.close()
        
        # Create default dashboard if none exist
        if not self.dashboards:
            self.create_default_dashboard()
        
        print(f"✅ Loaded {len(self.dashboards)} dashboards")
    
    def setup_data_sources(self):
        """Setup data sources for business intelligence."""
        # Analytics database
        if Path("analytics.db").exists():
            self.data_sources["analytics"] = "analytics.db"
        
        # ML models database
        if Path("ml_models").exists():
            self.data_sources["ml_models"] = "ml_models"
        
        # Processing jobs database
        if Path("data_processing").exists():
            self.data_sources["data_processing"] = "data_processing"
        
        # Notifications database
        if Path("notifications.db").exists():
            self.data_sources["notifications"] = "notifications.db"
        
        print(f"✅ Setup {len(self.data_sources)} data sources")
    
    def create_default_metrics(self):
        """Create default business intelligence metrics."""
        default_metrics = [
            {
                'id': 'total_documents_generated',
                'name': 'Total Documents Generated',
                'description': 'Total number of documents generated by the system',
                'metric_type': BIMetricType.KPI,
                'data_source': 'analytics',
                'calculation': 'SELECT COUNT(*) FROM documents',
                'unit': 'documents',
                'target_value': 1000
            },
            {
                'id': 'average_processing_time',
                'name': 'Average Processing Time',
                'description': 'Average time to process document generation requests',
                'metric_type': BIMetricType.KPI,
                'data_source': 'analytics',
                'calculation': 'SELECT AVG(processing_time) FROM documents',
                'unit': 'seconds',
                'target_value': 30,
                'threshold_warning': 45,
                'threshold_critical': 60
            },
            {
                'id': 'success_rate',
                'name': 'Success Rate',
                'description': 'Percentage of successful document generation requests',
                'metric_type': BIMetricType.KPI,
                'data_source': 'analytics',
                'calculation': 'SELECT (COUNT(CASE WHEN status = "completed" THEN 1 END) * 100.0 / COUNT(*)) FROM requests',
                'unit': '%',
                'target_value': 95,
                'threshold_warning': 90,
                'threshold_critical': 85
            },
            {
                'id': 'business_area_distribution',
                'name': 'Business Area Distribution',
                'description': 'Distribution of requests by business area',
                'metric_type': BIMetricType.DISTRIBUTION,
                'data_source': 'analytics',
                'calculation': 'SELECT business_area, COUNT(*) FROM requests GROUP BY business_area',
                'unit': 'requests'
            },
            {
                'id': 'daily_requests_trend',
                'name': 'Daily Requests Trend',
                'description': 'Daily trend of document generation requests',
                'metric_type': BIMetricType.TREND,
                'data_source': 'analytics',
                'calculation': 'SELECT DATE(timestamp) as date, COUNT(*) FROM requests GROUP BY DATE(timestamp) ORDER BY date',
                'unit': 'requests'
            },
            {
                'id': 'ml_model_accuracy',
                'name': 'ML Model Accuracy',
                'description': 'Average accuracy of machine learning models',
                'metric_type': BIMetricType.KPI,
                'data_source': 'ml_models',
                'calculation': 'SELECT AVG(accuracy) FROM models',
                'unit': '%',
                'target_value': 90
            },
            {
                'id': 'data_processing_jobs',
                'name': 'Data Processing Jobs',
                'description': 'Number of data processing jobs completed',
                'metric_type': BIMetricType.KPI,
                'data_source': 'data_processing',
                'calculation': 'SELECT COUNT(*) FROM jobs WHERE status = "completed"',
                'unit': 'jobs'
            },
            {
                'id': 'notification_delivery_rate',
                'name': 'Notification Delivery Rate',
                'description': 'Percentage of successfully delivered notifications',
                'metric_type': BIMetricType.KPI,
                'data_source': 'notifications',
                'calculation': 'SELECT (COUNT(CASE WHEN status = "sent" THEN 1 END) * 100.0 / COUNT(*)) FROM notifications',
                'unit': '%',
                'target_value': 98
            }
        ]
        
        for metric_data in default_metrics:
            self.create_metric(
                metric_id=metric_data['id'],
                name=metric_data['name'],
                description=metric_data['description'],
                metric_type=metric_data['metric_type'],
                data_source=metric_data['data_source'],
                calculation=metric_data['calculation'],
                unit=metric_data['unit'],
                target_value=metric_data.get('target_value'),
                threshold_warning=metric_data.get('threshold_warning'),
                threshold_critical=metric_data.get('threshold_critical')
            )
    
    def create_default_dashboard(self):
        """Create default business intelligence dashboard."""
        dashboard = self.create_dashboard(
            dashboard_id="executive_dashboard",
            name="Executive Dashboard",
            description="High-level business intelligence dashboard for executives",
            metrics=[
                "total_documents_generated",
                "average_processing_time",
                "success_rate",
                "ml_model_accuracy",
                "notification_delivery_rate"
            ],
            layout={
                "rows": 2,
                "columns": 3,
                "widgets": [
                    {"metric": "total_documents_generated", "position": [0, 0], "size": [1, 1]},
                    {"metric": "average_processing_time", "position": [0, 1], "size": [1, 1]},
                    {"metric": "success_rate", "position": [0, 2], "size": [1, 1]},
                    {"metric": "ml_model_accuracy", "position": [1, 0], "size": [1, 1]},
                    {"metric": "notification_delivery_rate", "position": [1, 1], "size": [1, 1]},
                    {"metric": "business_area_distribution", "position": [1, 2], "size": [1, 1]}
                ]
            }
        )
        
        return dashboard
    
    def create_metric(self, metric_id: str, name: str, description: str,
                     metric_type: BIMetricType, data_source: str, calculation: str,
                     unit: str, target_value: Optional[float] = None,
                     threshold_warning: Optional[float] = None,
                     threshold_critical: Optional[float] = None) -> BIMetric:
        """Create a new business intelligence metric."""
        metric = BIMetric(
            id=metric_id,
            name=name,
            description=description,
            metric_type=metric_type,
            data_source=data_source,
            calculation=calculation,
            unit=unit,
            target_value=target_value,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            created_at=datetime.now()
        )
        
        self.metrics[metric_id] = metric
        
        # Save to database
        conn = sqlite3.connect("business_intelligence.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO bi_metrics 
            (id, name, description, metric_type, data_source, calculation, unit, 
             target_value, threshold_warning, threshold_critical, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (metric_id, name, description, metric_type.value, data_source, calculation, unit,
              target_value, threshold_warning, threshold_critical, metric.created_at.isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created business intelligence metric: {name}")
        return metric
    
    def create_dashboard(self, dashboard_id: str, name: str, description: str,
                        metrics: List[str], layout: Dict[str, Any],
                        filters: List[Dict[str, Any]] = None,
                        refresh_interval: int = 300) -> BIDashboard:
        """Create a new business intelligence dashboard."""
        dashboard = BIDashboard(
            id=dashboard_id,
            name=name,
            description=description,
            metrics=metrics,
            layout=layout,
            filters=filters or [],
            refresh_interval=refresh_interval,
            created_at=datetime.now()
        )
        
        self.dashboards[dashboard_id] = dashboard
        
        # Save to database
        conn = sqlite3.connect("business_intelligence.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO bi_dashboards 
            (id, name, description, metrics, layout, filters, refresh_interval, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (dashboard_id, name, description, json.dumps(metrics), json.dumps(layout),
              json.dumps(filters or []), refresh_interval, dashboard.created_at.isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created business intelligence dashboard: {name}")
        return dashboard
    
    def calculate_metric(self, metric_id: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate a business intelligence metric."""
        if metric_id not in self.metrics:
            raise ValueError(f"Metric {metric_id} not found")
        
        metric = self.metrics[metric_id]
        
        try:
            # Get data source
            data_source = self.data_sources.get(metric.data_source)
            if not data_source:
                raise ValueError(f"Data source {metric.data_source} not found")
            
            # Execute calculation
            if data_source.endswith('.db'):
                # SQLite database
                conn = sqlite3.connect(data_source)
                cursor = conn.cursor()
                
                # Apply filters to calculation
                calculation = metric.calculation
                if filters:
                    # Add WHERE clause if not present
                    if 'WHERE' not in calculation.upper():
                        calculation += ' WHERE 1=1'
                    
                    for key, value in filters.items():
                        calculation += f" AND {key} = '{value}'"
                
                cursor.execute(calculation)
                result = cursor.fetchall()
                conn.close()
                
                # Process result based on metric type
                if metric.metric_type == BIMetricType.KPI:
                    value = result[0][0] if result else 0
                elif metric.metric_type == BIMetricType.DISTRIBUTION:
                    value = dict(result)
                elif metric.metric_type == BIMetricType.TREND:
                    value = [{'date': row[0], 'value': row[1]} for row in result]
                else:
                    value = result
            
            else:
                # File-based data source
                value = self._calculate_from_file(data_source, metric, filters)
            
            # Determine status based on thresholds
            status = "normal"
            if metric.threshold_critical is not None and isinstance(value, (int, float)):
                if value <= metric.threshold_critical:
                    status = "critical"
                elif metric.threshold_warning is not None and value <= metric.threshold_warning:
                    status = "warning"
            
            # Store data point
            self._store_data_point(metric_id, value, filters)
            
            return {
                'metric_id': metric_id,
                'name': metric.name,
                'value': value,
                'unit': metric.unit,
                'status': status,
                'target_value': metric.target_value,
                'threshold_warning': metric.threshold_warning,
                'threshold_critical': metric.threshold_critical,
                'calculated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating metric {metric_id}: {e}")
            return {
                'metric_id': metric_id,
                'name': metric.name,
                'value': None,
                'error': str(e),
                'calculated_at': datetime.now().isoformat()
            }
    
    def _calculate_from_file(self, data_source: str, metric: BIMetric, filters: Dict[str, Any]) -> Any:
        """Calculate metric from file-based data source."""
        # This would implement file-based calculations
        # For now, return a placeholder
        return 0
    
    def _store_data_point(self, metric_id: str, value: Any, metadata: Dict[str, Any] = None):
        """Store a data point for a metric."""
        conn = sqlite3.connect("business_intelligence.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bi_data_points (metric_id, value, metadata)
            VALUES (?, ?, ?)
        ''', (metric_id, value, json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()
    
    def generate_dashboard(self, dashboard_id: str, output_format: str = "html") -> str:
        """Generate a business intelligence dashboard."""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.dashboards[dashboard_id]
        
        print(f"📊 Generating dashboard: {dashboard.name}")
        
        # Calculate all metrics
        metric_results = {}
        for metric_id in dashboard.metrics:
            metric_results[metric_id] = self.calculate_metric(metric_id)
        
        # Generate visualization based on output format
        if output_format == "html":
            return self._generate_html_dashboard(dashboard, metric_results)
        elif output_format == "json":
            return json.dumps({
                'dashboard': {
                    'id': dashboard.id,
                    'name': dashboard.name,
                    'description': dashboard.description,
                    'generated_at': datetime.now().isoformat()
                },
                'metrics': metric_results
            }, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_html_dashboard(self, dashboard: BIDashboard, metric_results: Dict[str, Any]) -> str:
        """Generate HTML dashboard."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{dashboard.name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .dashboard {{ display: grid; grid-template-columns: repeat({dashboard.layout.get('columns', 3)}, 1fr); gap: 20px; }}
        .metric-card {{ background: #f5f5f5; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-unit {{ color: #666; }}
        .metric-status {{ margin-top: 10px; }}
        .status-normal {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-critical {{ color: #dc3545; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>{dashboard.name}</h1>
    <p>{dashboard.description}</p>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="dashboard">
"""
        
        # Generate metric cards
        for metric_id, result in metric_results.items():
            if result.get('error'):
                continue
            
            value = result['value']
            unit = result['unit']
            status = result.get('status', 'normal')
            
            # Format value
            if isinstance(value, (int, float)):
                if value >= 1000000:
                    formatted_value = f"{value/1000000:.1f}M"
                elif value >= 1000:
                    formatted_value = f"{value/1000:.1f}K"
                else:
                    formatted_value = f"{value:.1f}"
            else:
                formatted_value = str(value)
            
            html += f"""
        <div class="metric-card">
            <h3>{result['name']}</h3>
            <div class="metric-value">{formatted_value}</div>
            <div class="metric-unit">{unit}</div>
            <div class="metric-status status-{status}">
                Status: {status.title()}
            </div>
"""
            
            if result.get('target_value'):
                html += f"<div>Target: {result['target_value']} {unit}</div>"
            
            html += "</div>"
        
        html += """
    </div>
    
    <script>
        // Add any interactive features here
        console.log('Dashboard loaded successfully');
    </script>
</body>
</html>
"""
        
        return html
    
    def generate_visualization(self, metric_id: str, visualization_type: BIVisualizationType,
                              output_path: str = None) -> str:
        """Generate a visualization for a metric."""
        if metric_id not in self.metrics:
            raise ValueError(f"Metric {metric_id} not found")
        
        metric = self.metrics[metric_id]
        result = self.calculate_metric(metric_id)
        
        if result.get('error'):
            raise ValueError(f"Error calculating metric: {result['error']}")
        
        value = result['value']
        
        # Generate visualization based on type
        if visualization_type == BIVisualizationType.LINE_CHART:
            return self._generate_line_chart(metric, value, output_path)
        elif visualization_type == BIVisualizationType.BAR_CHART:
            return self._generate_bar_chart(metric, value, output_path)
        elif visualization_type == BIVisualizationType.PIE_CHART:
            return self._generate_pie_chart(metric, value, output_path)
        elif visualization_type == BIVisualizationType.GAUGE:
            return self._generate_gauge_chart(metric, value, output_path)
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")
    
    def _generate_line_chart(self, metric: BIMetric, value: Any, output_path: str = None) -> str:
        """Generate line chart visualization."""
        if not isinstance(value, list):
            return "Line chart requires trend data"
        
        # Create line chart using plotly
        fig = go.Figure()
        
        dates = [item['date'] for item in value]
        values = [item['value'] for item in value]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=metric.name,
            line=dict(color='#007bff', width=2)
        ))
        
        fig.update_layout(
            title=metric.name,
            xaxis_title='Date',
            yaxis_title=metric.unit,
            template='plotly_white'
        )
        
        if output_path:
            fig.write_html(output_path)
            return output_path
        else:
            return fig.to_html(include_plotlyjs='cdn')
    
    def _generate_bar_chart(self, metric: BIMetric, value: Any, output_path: str = None) -> str:
        """Generate bar chart visualization."""
        if not isinstance(value, dict):
            return "Bar chart requires distribution data"
        
        # Create bar chart using plotly
        fig = go.Figure()
        
        categories = list(value.keys())
        values_list = list(value.values())
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values_list,
            name=metric.name,
            marker_color='#007bff'
        ))
        
        fig.update_layout(
            title=metric.name,
            xaxis_title='Category',
            yaxis_title=metric.unit,
            template='plotly_white'
        )
        
        if output_path:
            fig.write_html(output_path)
            return output_path
        else:
            return fig.to_html(include_plotlyjs='cdn')
    
    def _generate_pie_chart(self, metric: BIMetric, value: Any, output_path: str = None) -> str:
        """Generate pie chart visualization."""
        if not isinstance(value, dict):
            return "Pie chart requires distribution data"
        
        # Create pie chart using plotly
        fig = go.Figure()
        
        labels = list(value.keys())
        values_list = list(value.values())
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values_list,
            name=metric.name
        ))
        
        fig.update_layout(
            title=metric.name,
            template='plotly_white'
        )
        
        if output_path:
            fig.write_html(output_path)
            return output_path
        else:
            return fig.to_html(include_plotlyjs='cdn')
    
    def _generate_gauge_chart(self, metric: BIMetric, value: Any, output_path: str = None) -> str:
        """Generate gauge chart visualization."""
        if not isinstance(value, (int, float)):
            return "Gauge chart requires numeric value"
        
        # Create gauge chart using plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': metric.name},
            delta={'reference': metric.target_value or 0},
            gauge={
                'axis': {'range': [None, metric.target_value * 1.2 or 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, metric.threshold_warning or 0], 'color': "lightgray"},
                    {'range': [metric.threshold_warning or 0, metric.threshold_critical or 0], 'color': "yellow"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': metric.target_value or 0
                }
            }
        ))
        
        fig.update_layout(
            template='plotly_white'
        )
        
        if output_path:
            fig.write_html(output_path)
            return output_path
        else:
            return fig.to_html(include_plotlyjs='cdn')
    
    def get_metric_history(self, metric_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical data for a metric."""
        conn = sqlite3.connect("business_intelligence.db")
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT value, timestamp, metadata FROM bi_data_points 
            WHERE metric_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (metric_id, start_date.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'value': row[0],
                'timestamp': row[1],
                'metadata': json.loads(row[2]) if row[2] else {}
            })
        
        return history
    
    def generate_bi_report(self) -> str:
        """Generate business intelligence report."""
        report = f"""
BUL Business Intelligence Report
===============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

METRICS
-------
Total Metrics: {len(self.metrics)}
"""
        
        for metric_id, metric in self.metrics.items():
            report += f"""
{metric.name} ({metric_id}):
  Type: {metric.metric_type.value}
  Data Source: {metric.data_source}
  Unit: {metric.unit}
  Target: {metric.target_value or 'N/A'}
  Created: {metric.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report += f"""
DASHBOARDS
----------
Total Dashboards: {len(self.dashboards)}
"""
        
        for dashboard_id, dashboard in self.dashboards.items():
            report += f"""
{dashboard.name} ({dashboard_id}):
  Metrics: {len(dashboard.metrics)}
  Refresh Interval: {dashboard.refresh_interval}s
  Created: {dashboard.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report += f"""
DATA SOURCES
------------
Total Data Sources: {len(self.data_sources)}
"""
        
        for source_name, source_path in self.data_sources.items():
            report += f"  {source_name}: {source_path}\n"
        
        return report

def main():
    """Main business intelligence dashboard function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Business Intelligence Dashboard")
    parser.add_argument("--create-metric", help="Create a new BI metric")
    parser.add_argument("--create-dashboard", help="Create a new BI dashboard")
    parser.add_argument("--calculate-metric", help="Calculate a BI metric")
    parser.add_argument("--generate-dashboard", help="Generate a BI dashboard")
    parser.add_argument("--generate-visualization", help="Generate a visualization")
    parser.add_argument("--list-metrics", action="store_true", help="List all BI metrics")
    parser.add_argument("--list-dashboards", action="store_true", help="List all dashboards")
    parser.add_argument("--metric-history", help="Get metric history")
    parser.add_argument("--report", action="store_true", help="Generate BI report")
    parser.add_argument("--name", help="Name for metric/dashboard")
    parser.add_argument("--description", help="Description for metric/dashboard")
    parser.add_argument("--metric-type", choices=['kpi', 'trend', 'comparison', 'distribution', 'correlation', 'forecast', 'benchmark'],
                       help="Metric type")
    parser.add_argument("--data-source", help="Data source for metric")
    parser.add_argument("--calculation", help="Calculation for metric")
    parser.add_argument("--unit", help="Unit for metric")
    parser.add_argument("--target-value", type=float, help="Target value for metric")
    parser.add_argument("--visualization-type", choices=['line_chart', 'bar_chart', 'pie_chart', 'scatter_plot', 'heatmap', 'gauge'],
                       help="Visualization type")
    parser.add_argument("--output-format", choices=['html', 'json'], default='html', help="Output format")
    parser.add_argument("--output-path", help="Output file path")
    parser.add_argument("--days", type=int, default=30, help="Number of days for history")
    
    args = parser.parse_args()
    
    bi_dashboard = BusinessIntelligenceDashboard()
    
    print("📊 BUL Business Intelligence Dashboard")
    print("=" * 50)
    
    if args.create_metric:
        if not all([args.name, args.description, args.metric_type, args.data_source, args.calculation, args.unit]):
            print("❌ Error: --name, --description, --metric-type, --data-source, --calculation, and --unit are required")
            return 1
        
        metric = bi_dashboard.create_metric(
            metric_id=args.create_metric,
            name=args.name,
            description=args.description,
            metric_type=BIMetricType(args.metric_type),
            data_source=args.data_source,
            calculation=args.calculation,
            unit=args.unit,
            target_value=args.target_value
        )
        print(f"✅ Created metric: {metric.name}")
    
    elif args.create_dashboard:
        if not all([args.name, args.description]):
            print("❌ Error: --name and --description are required")
            return 1
        
        # Create dashboard with default metrics
        dashboard = bi_dashboard.create_dashboard(
            dashboard_id=args.create_dashboard,
            name=args.name,
            description=args.description,
            metrics=list(bi_dashboard.metrics.keys())[:6],  # First 6 metrics
            layout={"rows": 2, "columns": 3}
        )
        print(f"✅ Created dashboard: {dashboard.name}")
    
    elif args.calculate_metric:
        result = bi_dashboard.calculate_metric(args.calculate_metric)
        print(f"📊 Metric: {result['name']}")
        print(f"   Value: {result['value']} {result['unit']}")
        print(f"   Status: {result.get('status', 'N/A')}")
        if result.get('target_value'):
            print(f"   Target: {result['target_value']} {result['unit']}")
    
    elif args.generate_dashboard:
        dashboard_html = bi_dashboard.generate_dashboard(args.generate_dashboard, args.output_format)
        
        if args.output_path:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            print(f"✅ Dashboard generated: {args.output_path}")
        else:
            print("📊 Dashboard HTML generated")
    
    elif args.generate_visualization:
        if not args.visualization_type:
            print("❌ Error: --visualization-type is required")
            return 1
        
        visualization = bi_dashboard.generate_visualization(
            args.generate_visualization,
            BIVisualizationType(args.visualization_type),
            args.output_path
        )
        
        if args.output_path:
            print(f"✅ Visualization generated: {args.output_path}")
        else:
            print("📊 Visualization generated")
    
    elif args.list_metrics:
        metrics = bi_dashboard.metrics
        if metrics:
            print(f"\n📊 Business Intelligence Metrics ({len(metrics)}):")
            print("-" * 60)
            for metric_id, metric in metrics.items():
                print(f"{metric.name} ({metric_id}):")
                print(f"  Type: {metric.metric_type.value}")
                print(f"  Data Source: {metric.data_source}")
                print(f"  Unit: {metric.unit}")
                print()
        else:
            print("No metrics found.")
    
    elif args.list_dashboards:
        dashboards = bi_dashboard.dashboards
        if dashboards:
            print(f"\n📊 Business Intelligence Dashboards ({len(dashboards)}):")
            print("-" * 60)
            for dashboard_id, dashboard in dashboards.items():
                print(f"{dashboard.name} ({dashboard_id}):")
                print(f"  Metrics: {len(dashboard.metrics)}")
                print(f"  Refresh Interval: {dashboard.refresh_interval}s")
                print()
        else:
            print("No dashboards found.")
    
    elif args.metric_history:
        history = bi_dashboard.get_metric_history(args.metric_history, args.days)
        if history:
            print(f"\n📈 Metric History: {args.metric_history} (Last {args.days} days)")
            print("-" * 60)
            for data_point in history[:10]:  # Show last 10 points
                print(f"{data_point['timestamp']}: {data_point['value']}")
        else:
            print("No history found for metric.")
    
    elif args.report:
        report = bi_dashboard.generate_bi_report()
        print(report)
        
        # Save report
        report_file = f"bi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        print(f"📊 Metrics: {len(bi_dashboard.metrics)}")
        print(f"📊 Dashboards: {len(bi_dashboard.dashboards)}")
        print(f"📊 Data Sources: {len(bi_dashboard.data_sources)}")
        print(f"\n💡 Use --list-metrics to see all metrics")
        print(f"💡 Use --create-dashboard to create a new dashboard")
        print(f"💡 Use --generate-dashboard to generate a dashboard")
        print(f"💡 Use --report to generate BI report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
