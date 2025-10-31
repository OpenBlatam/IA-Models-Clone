"""
AI History Comparison System - Visualization Engine

This module provides comprehensive data visualization capabilities,
interactive dashboards, and chart generation for the AI History Comparison system.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import io
from pathlib import Path

# Data processing
import pandas as pd
import numpy as np

# Visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.utils
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .ai_history_analyzer import AIHistoryAnalyzer, MetricType
from .advanced_ml_engine import ml_engine

logger = logging.getLogger(__name__)

class ChartType(Enum):
    """Types of charts available"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"
    AREA = "area"
    RADAR = "radar"
    TREEMAP = "treemap"

class DashboardType(Enum):
    """Types of dashboards available"""
    OVERVIEW = "overview"
    TRENDS = "trends"
    QUALITY = "quality"
    COMPARISONS = "comparisons"
    ANOMALIES = "anomalies"
    CLUSTERING = "clustering"
    PERFORMANCE = "performance"
    CUSTOM = "custom"

@dataclass
class ChartConfig:
    """Configuration for chart generation"""
    chart_type: ChartType
    title: str
    x_axis: str
    y_axis: str
    width: int = 800
    height: int = 600
    colors: Optional[List[str]] = None
    show_legend: bool = True
    show_grid: bool = True
    theme: str = "default"

@dataclass
class DashboardConfig:
    """Configuration for dashboard generation"""
    dashboard_type: DashboardType
    title: str
    charts: List[ChartConfig]
    layout: str = "grid"  # grid, single, custom
    refresh_interval: int = 300  # seconds
    filters: Optional[Dict[str, Any]] = None

@dataclass
class VisualizationData:
    """Data structure for visualization"""
    labels: List[str]
    values: List[float]
    timestamps: Optional[List[datetime]] = None
    categories: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class VisualizationEngine:
    """
    Advanced visualization engine for AI History Comparison data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize visualization engine"""
        self.config = config or {}
        self.chart_cache: Dict[str, str] = {}
        self.dashboard_cache: Dict[str, Dict] = {}
        
        # Set up matplotlib style
        if HAS_MATPLOTLIB:
            plt.style.use('default')
            sns.set_palette("husl") if HAS_SEABORN else None
        
        # Color palettes
        self.color_palettes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
            "qualitative": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"],
            "sequential": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"],
            "diverging": ["#8e0152", "#c51b7d", "#de77ae", "#f1b6da", "#fde0ef", "#e6f5d0", "#b8e186", "#7fbc41"]
        }
        
        logger.info("Visualization Engine initialized")

    def create_trend_chart(self, data: List[Tuple[datetime, float]], 
                          title: str = "Trend Analysis",
                          metric_name: str = "Metric") -> Dict[str, Any]:
        """Create a trend line chart"""
        if not data:
            return {"error": "No data available for trend chart"}
        
        try:
            # Prepare data
            timestamps = [point[0] for point in data]
            values = [point[1] for point in data]
            
            if HAS_PLOTLY:
                return self._create_plotly_trend_chart(timestamps, values, title, metric_name)
            elif HAS_MATPLOTLIB:
                return self._create_matplotlib_trend_chart(timestamps, values, title, metric_name)
            else:
                return {"error": "No visualization libraries available"}
                
        except Exception as e:
            logger.error(f"Error creating trend chart: {e}")
            return {"error": str(e)}

    def _create_plotly_trend_chart(self, timestamps: List[datetime], values: List[float], 
                                  title: str, metric_name: str) -> Dict[str, Any]:
        """Create trend chart using Plotly"""
        fig = go.Figure()
        
        # Add main trend line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name=metric_name,
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        # Add trend line if enough data points
        if len(values) > 2:
            # Calculate trend line
            x_numeric = np.arange(len(values))
            z = np.polyfit(x_numeric, values, 1)
            p = np.poly1d(z)
            trend_values = p(x_numeric)
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=trend_values,
                mode='lines',
                name='Trend',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=metric_name,
            hovermode='x unified',
            showlegend=True,
            width=800,
            height=600,
            template="plotly_white"
        )
        
        # Convert to JSON
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return {"type": "plotly", "data": json.loads(chart_json)}

    def _create_matplotlib_trend_chart(self, timestamps: List[datetime], values: List[float], 
                                      title: str, metric_name: str) -> Dict[str, Any]:
        """Create trend chart using Matplotlib"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot main line
        ax.plot(timestamps, values, 'o-', color='#1f77b4', linewidth=2, markersize=6, label=metric_name)
        
        # Add trend line if enough data points
        if len(values) > 2:
            x_numeric = np.arange(len(values))
            z = np.polyfit(x_numeric, values, 1)
            p = np.poly1d(z)
            trend_values = p(x_numeric)
            ax.plot(timestamps, trend_values, '--', color='#ff7f0e', linewidth=2, label='Trend')
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return {"type": "matplotlib", "data": f"data:image/png;base64,{img_base64}"}

    def create_quality_distribution_chart(self, entries: List[Dict[str, Any]], 
                                        metric: str = "readability_score") -> Dict[str, Any]:
        """Create quality distribution chart"""
        if not entries:
            return {"error": "No data available for distribution chart"}
        
        try:
            # Extract metric values
            values = [entry.get(metric, 0) for entry in entries if entry.get(metric) is not None]
            
            if not values:
                return {"error": f"No valid {metric} data found"}
            
            if HAS_PLOTLY:
                return self._create_plotly_distribution_chart(values, metric)
            elif HAS_MATPLOTLIB:
                return self._create_matplotlib_distribution_chart(values, metric)
            else:
                return {"error": "No visualization libraries available"}
                
        except Exception as e:
            logger.error(f"Error creating distribution chart: {e}")
            return {"error": str(e)}

    def _create_plotly_distribution_chart(self, values: List[float], metric: str) -> Dict[str, Any]:
        """Create distribution chart using Plotly"""
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=20,
            name=metric,
            marker_color='#1f77b4',
            opacity=0.7
        ))
        
        # Add mean line
        mean_val = np.mean(values)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}"
        )
        
        # Update layout
        fig.update_layout(
            title=f"Distribution of {metric.replace('_', ' ').title()}",
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title="Frequency",
            showlegend=False,
            width=800,
            height=600,
            template="plotly_white"
        )
        
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return {"type": "plotly", "data": json.loads(chart_json)}

    def _create_matplotlib_distribution_chart(self, values: List[float], metric: str) -> Dict[str, Any]:
        """Create distribution chart using Matplotlib"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(values, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        
        # Formatting
        ax.set_title(f"Distribution of {metric.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return {"type": "matplotlib", "data": f"data:image/png;base64,{img_base64}"}

    def create_comparison_chart(self, comparison_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison chart for multiple entries"""
        if not comparison_data:
            return {"error": "No comparison data available"}
        
        try:
            # Prepare data
            labels = [item.get('label', f'Entry {i}') for i, item in enumerate(comparison_data)]
            metrics = ['readability_score', 'sentiment_score', 'complexity_score', 'topic_diversity']
            
            if HAS_PLOTLY:
                return self._create_plotly_comparison_chart(labels, metrics, comparison_data)
            elif HAS_MATPLOTLIB:
                return self._create_matplotlib_comparison_chart(labels, metrics, comparison_data)
            else:
                return {"error": "No visualization libraries available"}
                
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return {"error": str(e)}

    def _create_plotly_comparison_chart(self, labels: List[str], metrics: List[str], 
                                       comparison_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison chart using Plotly"""
        fig = go.Figure()
        
        colors = self.color_palettes["default"]
        
        for i, metric in enumerate(metrics):
            values = [item.get(metric, 0) for item in comparison_data]
            
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=labels,
                y=values,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title="Content Quality Comparison",
            xaxis_title="Entries",
            yaxis_title="Score",
            barmode='group',
            width=800,
            height=600,
            template="plotly_white"
        )
        
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return {"type": "plotly", "data": json.loads(chart_json)}

    def _create_matplotlib_comparison_chart(self, labels: List[str], metrics: List[str], 
                                          comparison_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison chart using Matplotlib"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(labels))
        width = 0.2
        colors = self.color_palettes["default"]
        
        for i, metric in enumerate(metrics):
            values = [item.get(metric, 0) for item in comparison_data]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), 
                  color=colors[i % len(colors)])
        
        ax.set_title("Content Quality Comparison", fontsize=14, fontweight='bold')
        ax.set_xlabel("Entries", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return {"type": "matplotlib", "data": f"data:image/png;base64,{img_base64}"}

    def create_anomaly_chart(self, entries: List[Dict[str, Any]], 
                           anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create anomaly detection visualization"""
        if not entries:
            return {"error": "No data available for anomaly chart"}
        
        try:
            # Prepare data
            timestamps = [entry.get('timestamp', datetime.now()) for entry in entries]
            if isinstance(timestamps[0], str):
                timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
            
            readability_scores = [entry.get('readability_score', 0) for entry in entries]
            anomaly_flags = [entry.get('id') in [a.get('entry_id') for a in anomalies] for entry in entries]
            
            if HAS_PLOTLY:
                return self._create_plotly_anomaly_chart(timestamps, readability_scores, anomaly_flags)
            elif HAS_MATPLOTLIB:
                return self._create_matplotlib_anomaly_chart(timestamps, readability_scores, anomaly_flags)
            else:
                return {"error": "No visualization libraries available"}
                
        except Exception as e:
            logger.error(f"Error creating anomaly chart: {e}")
            return {"error": str(e)}

    def _create_plotly_anomaly_chart(self, timestamps: List[datetime], values: List[float], 
                                    anomaly_flags: List[bool]) -> Dict[str, Any]:
        """Create anomaly chart using Plotly"""
        fig = go.Figure()
        
        # Normal points
        normal_timestamps = [ts for ts, flag in zip(timestamps, anomaly_flags) if not flag]
        normal_values = [val for val, flag in zip(values, anomaly_flags) if not flag]
        
        # Anomaly points
        anomaly_timestamps = [ts for ts, flag in zip(timestamps, anomaly_flags) if flag]
        anomaly_values = [val for val, flag in zip(values, anomaly_flags) if flag]
        
        # Add normal points
        fig.add_trace(go.Scatter(
            x=normal_timestamps,
            y=normal_values,
            mode='markers',
            name='Normal',
            marker=dict(color='#1f77b4', size=8)
        ))
        
        # Add anomaly points
        if anomaly_timestamps:
            fig.add_trace(go.Scatter(
                x=anomaly_timestamps,
                y=anomaly_values,
                mode='markers',
                name='Anomaly',
                marker=dict(color='#d62728', size=12, symbol='x')
            ))
        
        fig.update_layout(
            title="Anomaly Detection in Content Quality",
            xaxis_title="Time",
            yaxis_title="Readability Score",
            width=800,
            height=600,
            template="plotly_white"
        )
        
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return {"type": "plotly", "data": json.loads(chart_json)}

    def _create_matplotlib_anomaly_chart(self, timestamps: List[datetime], values: List[float], 
                                        anomaly_flags: List[bool]) -> Dict[str, Any]:
        """Create anomaly chart using Matplotlib"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normal points
        normal_timestamps = [ts for ts, flag in zip(timestamps, anomaly_flags) if not flag]
        normal_values = [val for val, flag in zip(values, anomaly_flags) if not flag]
        
        # Anomaly points
        anomaly_timestamps = [ts for ts, flag in zip(timestamps, anomaly_flags) if flag]
        anomaly_values = [val for val, flag in zip(values, anomaly_flags) if flag]
        
        # Plot normal points
        ax.scatter(normal_timestamps, normal_values, color='#1f77b4', s=50, label='Normal', alpha=0.7)
        
        # Plot anomaly points
        if anomaly_timestamps:
            ax.scatter(anomaly_timestamps, anomaly_values, color='#d62728', s=100, 
                      marker='x', label='Anomaly', linewidth=3)
        
        ax.set_title("Anomaly Detection in Content Quality", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Readability Score", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return {"type": "matplotlib", "data": f"data:image/png;base64,{img_base64}"}

    def create_clustering_chart(self, clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create clustering visualization"""
        if not clustering_result or not clustering_result.get('clusters'):
            return {"error": "No clustering data available"}
        
        try:
            clusters = clustering_result['clusters']
            cluster_centers = clustering_result.get('cluster_centers', {})
            
            if HAS_PLOTLY:
                return self._create_plotly_clustering_chart(clusters, cluster_centers)
            elif HAS_MATPLOTLIB:
                return self._create_matplotlib_clustering_chart(clusters, cluster_centers)
            else:
                return {"error": "No visualization libraries available"}
                
        except Exception as e:
            logger.error(f"Error creating clustering chart: {e}")
            return {"error": str(e)}

    def _create_plotly_clustering_chart(self, clusters: Dict[int, List[str]], 
                                       cluster_centers: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """Create clustering chart using Plotly"""
        fig = go.Figure()
        
        colors = self.color_palettes["qualitative"]
        
        for cluster_id, entries in clusters.items():
            # Get cluster center metrics
            center = cluster_centers.get(cluster_id, {})
            readability = center.get('readability_score', 0)
            sentiment = center.get('sentiment_score', 0)
            complexity = center.get('complexity_score', 0)
            
            fig.add_trace(go.Scatter3d(
                x=[readability],
                y=[sentiment],
                z=[complexity],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=len(entries) * 2,
                    color=colors[cluster_id % len(colors)],
                    opacity=0.7
                ),
                text=[f"Cluster {cluster_id}<br>Entries: {len(entries)}"],
                hovertemplate="%{text}<br>Readability: %{x}<br>Sentiment: %{y}<br>Complexity: %{z}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Content Clustering Visualization",
            scene=dict(
                xaxis_title="Readability Score",
                yaxis_title="Sentiment Score",
                zaxis_title="Complexity Score"
            ),
            width=800,
            height=600,
            template="plotly_white"
        )
        
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return {"type": "plotly", "data": json.loads(chart_json)}

    def _create_matplotlib_clustering_chart(self, clusters: Dict[int, List[str]], 
                                          cluster_centers: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """Create clustering chart using Matplotlib"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = self.color_palettes["qualitative"]
        
        for cluster_id, entries in clusters.items():
            center = cluster_centers.get(cluster_id, {})
            readability = center.get('readability_score', 0)
            sentiment = center.get('sentiment_score', 0)
            complexity = center.get('complexity_score', 0)
            
            ax.scatter(readability, sentiment, complexity, 
                      c=colors[cluster_id % len(colors)], 
                      s=len(entries) * 20, 
                      label=f'Cluster {cluster_id}',
                      alpha=0.7)
        
        ax.set_xlabel("Readability Score")
        ax.set_ylabel("Sentiment Score")
        ax.set_zlabel("Complexity Score")
        ax.set_title("Content Clustering Visualization")
        ax.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return {"type": "matplotlib", "data": f"data:image/png;base64,{img_base64}"}

    def create_dashboard(self, dashboard_config: DashboardConfig, 
                        data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete dashboard with multiple charts"""
        try:
            dashboard = {
                "title": dashboard_config.title,
                "type": dashboard_config.dashboard_type.value,
                "layout": dashboard_config.layout,
                "refresh_interval": dashboard_config.refresh_interval,
                "charts": [],
                "created_at": datetime.now().isoformat()
            }
            
            # Generate each chart
            for chart_config in dashboard_config.charts:
                chart_data = self._generate_chart_data(chart_config, data)
                if "error" not in chart_data:
                    dashboard["charts"].append({
                        "config": asdict(chart_config),
                        "data": chart_data
                    })
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return {"error": str(e)}

    def _generate_chart_data(self, chart_config: ChartConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart data based on configuration"""
        try:
            if chart_config.chart_type == ChartType.LINE:
                return self.create_trend_chart(
                    data.get(chart_config.x_axis, []),
                    chart_config.title,
                    chart_config.y_axis
                )
            elif chart_config.chart_type == ChartType.BAR:
                return self.create_comparison_chart(
                    data.get(chart_config.x_axis, [])
                )
            elif chart_config.chart_type == ChartType.HISTOGRAM:
                return self.create_quality_distribution_chart(
                    data.get(chart_config.x_axis, []),
                    chart_config.y_axis
                )
            else:
                return {"error": f"Unsupported chart type: {chart_config.chart_type}"}
                
        except Exception as e:
            logger.error(f"Error generating chart data: {e}")
            return {"error": str(e)}

    def export_chart(self, chart_data: Dict[str, Any], format: str = "png", 
                    filename: Optional[str] = None) -> Dict[str, Any]:
        """Export chart to file"""
        try:
            if not filename:
                filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            
            file_path = Path("exports") / filename
            file_path.parent.mkdir(exist_ok=True)
            
            if chart_data.get("type") == "matplotlib" and format == "png":
                # Extract base64 data
                base64_data = chart_data["data"].split(",")[1]
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(base64_data))
                
                return {
                    "success": True,
                    "filename": filename,
                    "path": str(file_path),
                    "size": file_path.stat().st_size
                }
            else:
                return {"error": f"Export format {format} not supported for this chart type"}
                
        except Exception as e:
            logger.error(f"Error exporting chart: {e}")
            return {"error": str(e)}

    def get_available_chart_types(self) -> List[Dict[str, str]]:
        """Get list of available chart types"""
        return [
            {"value": chart_type.value, "label": chart_type.value.replace("_", " ").title()}
            for chart_type in ChartType
        ]

    def get_available_dashboard_types(self) -> List[Dict[str, str]]:
        """Get list of available dashboard types"""
        return [
            {"value": dashboard_type.value, "label": dashboard_type.value.replace("_", " ").title()}
            for dashboard_type in DashboardType
        ]

    def get_color_palettes(self) -> Dict[str, List[str]]:
        """Get available color palettes"""
        return self.color_palettes

    def clear_cache(self):
        """Clear visualization cache"""
        self.chart_cache.clear()
        self.dashboard_cache.clear()
        logger.info("Visualization cache cleared")


# Global visualization engine instance
viz_engine = VisualizationEngine()

# Convenience functions
def create_trend_chart(data: List[Tuple[datetime, float]], title: str = "Trend Analysis", metric_name: str = "Metric") -> Dict[str, Any]:
    """Create a trend line chart"""
    return viz_engine.create_trend_chart(data, title, metric_name)

def create_quality_distribution_chart(entries: List[Dict[str, Any]], metric: str = "readability_score") -> Dict[str, Any]:
    """Create quality distribution chart"""
    return viz_engine.create_quality_distribution_chart(entries, metric)

def create_comparison_chart(comparison_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create comparison chart"""
    return viz_engine.create_comparison_chart(comparison_data)

def create_anomaly_chart(entries: List[Dict[str, Any]], anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create anomaly detection visualization"""
    return viz_engine.create_anomaly_chart(entries, anomalies)

def create_clustering_chart(clustering_result: Dict[str, Any]) -> Dict[str, Any]:
    """Create clustering visualization"""
    return viz_engine.create_clustering_chart(clustering_result)

def create_dashboard(dashboard_config: DashboardConfig, data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a complete dashboard"""
    return viz_engine.create_dashboard(dashboard_config, data)



























