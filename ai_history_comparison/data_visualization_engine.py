"""
Data Visualization Engine
=========================

Advanced data visualization engine for AI model analysis with interactive
charts, real-time dashboards, and comprehensive visualization capabilities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import base64
import io
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Types of charts"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    CORRELATION = "correlation"
    RADAR = "radar"
    PIE = "pie"
    DONUT = "donut"
    AREA = "area"
    CANDLESTICK = "candlestick"
    WATERFALL = "waterfall"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    SANKEY = "sankey"
    PARALLEL_COORDINATES = "parallel_coordinates"
    PARETO = "pareto"
    FUNNEL = "funnel"
    GAUGE = "gauge"
    INDICATOR = "indicator"


class DashboardType(str, Enum):
    """Types of dashboards"""
    PERFORMANCE = "performance"
    ANALYTICS = "analytics"
    COMPARISON = "comparison"
    BENCHMARKING = "benchmarking"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"
    REAL_TIME = "real_time"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    CUSTOM = "custom"


class VisualizationTheme(str, Enum):
    """Visualization themes"""
    LIGHT = "light"
    DARK = "dark"
    CORPORATE = "corporate"
    SCIENTIFIC = "scientific"
    COLORFUL = "colorful"
    MINIMAL = "minimal"
    HIGH_CONTRAST = "high_contrast"
    ACCESSIBLE = "accessible"


@dataclass
class ChartConfig:
    """Chart configuration"""
    chart_id: str
    chart_type: ChartType
    title: str
    x_axis: str
    y_axis: str
    data_source: str
    filters: Dict[str, Any]
    styling: Dict[str, Any]
    interactivity: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    dashboard_id: str
    name: str
    dashboard_type: DashboardType
    description: str
    charts: List[ChartConfig]
    layout: Dict[str, Any]
    theme: VisualizationTheme
    refresh_interval: int = 30
    auto_refresh: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class VisualizationResult:
    """Visualization result"""
    visualization_id: str
    chart_type: ChartType
    data: Dict[str, Any]
    chart_html: str
    chart_json: str
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class DataVisualizationEngine:
    """Advanced data visualization engine for AI model analysis"""
    
    def __init__(self, max_visualizations: int = 1000):
        self.max_visualizations = max_visualizations
        self.chart_configs: Dict[str, ChartConfig] = {}
        self.dashboard_configs: Dict[str, DashboardConfig] = {}
        self.visualization_results: List[VisualizationResult] = []
        
        # Visualization settings
        self.default_theme = VisualizationTheme.CORPORATE
        self.color_palettes = {
            "corporate": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
            "scientific": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#7209B7", "#0F4C75", "#3282B8", "#0F3460"],
            "colorful": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"],
            "minimal": ["#2C3E50", "#34495E", "#7F8C8D", "#95A5A6", "#BDC3C7", "#ECF0F1", "#E74C3C", "#E67E22"]
        }
        
        # Cache for visualizations
        self.visualization_cache = {}
        self.cache_ttl = 1800  # 30 minutes
    
    async def create_performance_chart(self, 
                                     model_name: str,
                                     metric: str,
                                     time_range_days: int = 30,
                                     chart_type: ChartType = ChartType.LINE) -> VisualizationResult:
        """Create performance chart for a model"""
        try:
            chart_id = hashlib.md5(f"performance_{model_name}_{metric}_{datetime.now()}".encode()).hexdigest()
            
            # Generate sample performance data
            data = await self._generate_performance_data(model_name, metric, time_range_days)
            
            # Create chart based on type
            if chart_type == ChartType.LINE:
                chart_html, chart_json = await self._create_line_chart(data, f"{model_name} - {metric}")
            elif chart_type == ChartType.BAR:
                chart_html, chart_json = await self._create_bar_chart(data, f"{model_name} - {metric}")
            elif chart_type == ChartType.AREA:
                chart_html, chart_json = await self._create_area_chart(data, f"{model_name} - {metric}")
            else:
                chart_html, chart_json = await self._create_line_chart(data, f"{model_name} - {metric}")
            
            # Create visualization result
            result = VisualizationResult(
                visualization_id=chart_id,
                chart_type=chart_type,
                data=data,
                chart_html=chart_html,
                chart_json=chart_json,
                metadata={
                    "model_name": model_name,
                    "metric": metric,
                    "time_range_days": time_range_days,
                    "data_points": len(data.get("timestamps", []))
                }
            )
            
            # Store result
            self.visualization_results.append(result)
            
            logger.info(f"Created performance chart for {model_name} - {metric}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {str(e)}")
            raise e
    
    async def create_comparison_chart(self, 
                                    model_names: List[str],
                                    metric: str,
                                    time_range_days: int = 30,
                                    chart_type: ChartType = ChartType.LINE) -> VisualizationResult:
        """Create comparison chart for multiple models"""
        try:
            chart_id = hashlib.md5(f"comparison_{'_'.join(model_names)}_{metric}_{datetime.now()}".encode()).hexdigest()
            
            # Generate comparison data
            data = await self._generate_comparison_data(model_names, metric, time_range_days)
            
            # Create chart based on type
            if chart_type == ChartType.LINE:
                chart_html, chart_json = await self._create_multi_line_chart(data, f"Model Comparison - {metric}")
            elif chart_type == ChartType.BAR:
                chart_html, chart_json = await self._create_grouped_bar_chart(data, f"Model Comparison - {metric}")
            elif chart_type == ChartType.RADAR:
                chart_html, chart_json = await self._create_radar_chart(data, f"Model Comparison - {metric}")
            else:
                chart_html, chart_json = await self._create_multi_line_chart(data, f"Model Comparison - {metric}")
            
            # Create visualization result
            result = VisualizationResult(
                visualization_id=chart_id,
                chart_type=chart_type,
                data=data,
                chart_html=chart_html,
                chart_json=chart_json,
                metadata={
                    "model_names": model_names,
                    "metric": metric,
                    "time_range_days": time_range_days,
                    "models_count": len(model_names)
                }
            )
            
            # Store result
            self.visualization_results.append(result)
            
            logger.info(f"Created comparison chart for {len(model_names)} models - {metric}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {str(e)}")
            raise e
    
    async def create_analytics_dashboard(self, 
                                       dashboard_type: DashboardType = DashboardType.ANALYTICS,
                                       model_names: List[str] = None,
                                       time_range_days: int = 30) -> DashboardConfig:
        """Create analytics dashboard"""
        try:
            dashboard_id = hashlib.md5(f"dashboard_{dashboard_type}_{datetime.now()}".encode()).hexdigest()
            
            if model_names is None:
                model_names = ["gpt-4", "claude-3", "gemini-pro"]
            
            # Create charts for dashboard
            charts = []
            
            # Performance overview chart
            perf_chart = ChartConfig(
                chart_id=hashlib.md5(f"perf_overview_{dashboard_id}".encode()).hexdigest(),
                chart_type=ChartType.LINE,
                title="Performance Overview",
                x_axis="time",
                y_axis="performance_score",
                data_source="performance_data",
                filters={"time_range_days": time_range_days},
                styling={"theme": self.default_theme.value},
                interactivity={"zoom": True, "pan": True, "hover": True}
            )
            charts.append(perf_chart)
            
            # Model comparison chart
            comp_chart = ChartConfig(
                chart_id=hashlib.md5(f"model_comp_{dashboard_id}".encode()).hexdigest(),
                chart_type=ChartType.BAR,
                title="Model Comparison",
                x_axis="model",
                y_axis="accuracy",
                data_source="comparison_data",
                filters={"models": model_names},
                styling={"theme": self.default_theme.value},
                interactivity={"click": True, "hover": True}
            )
            charts.append(comp_chart)
            
            # Metrics distribution chart
            dist_chart = ChartConfig(
                chart_id=hashlib.md5(f"metrics_dist_{dashboard_id}".encode()).hexdigest(),
                chart_type=ChartType.HISTOGRAM,
                title="Metrics Distribution",
                x_axis="metric_value",
                y_axis="frequency",
                data_source="metrics_data",
                filters={"time_range_days": time_range_days},
                styling={"theme": self.default_theme.value},
                interactivity={"brush": True, "hover": True}
            )
            charts.append(dist_chart)
            
            # Correlation heatmap
            corr_chart = ChartConfig(
                chart_id=hashlib.md5(f"correlation_{dashboard_id}".encode()).hexdigest(),
                chart_type=ChartType.HEATMAP,
                title="Metrics Correlation",
                x_axis="metric_1",
                y_axis="metric_2",
                data_source="correlation_data",
                filters={"models": model_names},
                styling={"theme": self.default_theme.value},
                interactivity={"hover": True}
            )
            charts.append(corr_chart)
            
            # Create dashboard configuration
            dashboard = DashboardConfig(
                dashboard_id=dashboard_id,
                name=f"{dashboard_type.value.title()} Dashboard",
                dashboard_type=dashboard_type,
                description=f"Comprehensive {dashboard_type.value} dashboard for AI model analysis",
                charts=charts,
                layout={
                    "grid": "2x2",
                    "responsive": True,
                    "spacing": "medium",
                    "header_height": 60,
                    "sidebar_width": 250
                },
                theme=self.default_theme,
                refresh_interval=30,
                auto_refresh=True
            )
            
            # Store dashboard
            self.dashboard_configs[dashboard_id] = dashboard
            
            logger.info(f"Created {dashboard_type.value} dashboard with {len(charts)} charts")
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating analytics dashboard: {str(e)}")
            raise e
    
    async def create_real_time_dashboard(self, 
                                       model_names: List[str] = None,
                                       update_interval: int = 5) -> DashboardConfig:
        """Create real-time dashboard"""
        try:
            dashboard_id = hashlib.md5(f"realtime_{datetime.now()}".encode()).hexdigest()
            
            if model_names is None:
                model_names = ["gpt-4", "claude-3", "gemini-pro"]
            
            # Create real-time charts
            charts = []
            
            # Real-time performance gauge
            gauge_chart = ChartConfig(
                chart_id=hashlib.md5(f"realtime_gauge_{dashboard_id}".encode()).hexdigest(),
                chart_type=ChartType.GAUGE,
                title="Real-time Performance",
                x_axis="",
                y_axis="performance_score",
                data_source="realtime_data",
                filters={"models": model_names},
                styling={"theme": self.default_theme.value},
                interactivity={"refresh": True}
            )
            charts.append(gauge_chart)
            
            # Real-time metrics line chart
            rt_line_chart = ChartConfig(
                chart_id=hashlib.md5(f"realtime_line_{dashboard_id}".encode()).hexdigest(),
                chart_type=ChartType.LINE,
                title="Real-time Metrics",
                x_axis="timestamp",
                y_axis="metric_value",
                data_source="realtime_data",
                filters={"models": model_names},
                styling={"theme": self.default_theme.value},
                interactivity={"streaming": True}
            )
            charts.append(rt_line_chart)
            
            # System status indicators
            status_chart = ChartConfig(
                chart_id=hashlib.md5(f"system_status_{dashboard_id}".encode()).hexdigest(),
                chart_type=ChartType.INDICATOR,
                title="System Status",
                x_axis="",
                y_axis="status_value",
                data_source="system_data",
                filters={},
                styling={"theme": self.default_theme.value},
                interactivity={"refresh": True}
            )
            charts.append(status_chart)
            
            # Create real-time dashboard
            dashboard = DashboardConfig(
                dashboard_id=dashboard_id,
                name="Real-time Dashboard",
                dashboard_type=DashboardType.REAL_TIME,
                description="Real-time monitoring dashboard for AI models",
                charts=charts,
                layout={
                    "grid": "1x3",
                    "responsive": True,
                    "spacing": "small",
                    "header_height": 40,
                    "sidebar_width": 200
                },
                theme=self.default_theme,
                refresh_interval=update_interval,
                auto_refresh=True
            )
            
            # Store dashboard
            self.dashboard_configs[dashboard_id] = dashboard
            
            logger.info(f"Created real-time dashboard with {len(charts)} charts")
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating real-time dashboard: {str(e)}")
            raise e
    
    async def create_benchmark_visualization(self, 
                                           benchmark_results: List[Dict[str, Any]],
                                           visualization_type: str = "comprehensive") -> VisualizationResult:
        """Create benchmark visualization"""
        try:
            chart_id = hashlib.md5(f"benchmark_{visualization_type}_{datetime.now()}".encode()).hexdigest()
            
            # Process benchmark data
            data = await self._process_benchmark_data(benchmark_results)
            
            # Create visualization based on type
            if visualization_type == "comprehensive":
                chart_html, chart_json = await self._create_benchmark_comprehensive_chart(data)
            elif visualization_type == "ranking":
                chart_html, chart_json = await self._create_benchmark_ranking_chart(data)
            elif visualization_type == "metrics":
                chart_html, chart_json = await self._create_benchmark_metrics_chart(data)
            else:
                chart_html, chart_json = await self._create_benchmark_comprehensive_chart(data)
            
            # Create visualization result
            result = VisualizationResult(
                visualization_id=chart_id,
                chart_type=ChartType.BAR,
                data=data,
                chart_html=chart_html,
                chart_json=chart_json,
                metadata={
                    "visualization_type": visualization_type,
                    "benchmark_count": len(benchmark_results),
                    "models_count": len(set(r.get("model_name", "") for r in benchmark_results))
                }
            )
            
            # Store result
            self.visualization_results.append(result)
            
            logger.info(f"Created benchmark visualization: {visualization_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating benchmark visualization: {str(e)}")
            raise e
    
    async def export_visualization(self, 
                                 visualization_id: str,
                                 format: str = "html",
                                 include_data: bool = True) -> Dict[str, Any]:
        """Export visualization in various formats"""
        try:
            # Find visualization
            visualization = None
            for viz in self.visualization_results:
                if viz.visualization_id == visualization_id:
                    visualization = viz
                    break
            
            if not visualization:
                raise ValueError(f"Visualization {visualization_id} not found")
            
            export_data = {
                "visualization_id": visualization_id,
                "chart_type": visualization.chart_type.value,
                "created_at": visualization.created_at.isoformat(),
                "metadata": visualization.metadata
            }
            
            if format == "html":
                export_data["html"] = visualization.chart_html
            elif format == "json":
                export_data["json"] = visualization.chart_json
            elif format == "png":
                # Convert to PNG (would need additional libraries)
                export_data["png"] = "data:image/png;base64," + base64.b64encode(visualization.chart_html.encode()).decode()
            elif format == "pdf":
                # Convert to PDF (would need additional libraries)
                export_data["pdf"] = "data:application/pdf;base64," + base64.b64encode(visualization.chart_html.encode()).decode()
            
            if include_data:
                export_data["data"] = visualization.data
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting visualization: {str(e)}")
            raise e
    
    async def get_visualization_analytics(self, 
                                        time_range_days: int = 30) -> Dict[str, Any]:
        """Get visualization analytics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            recent_visualizations = [v for v in self.visualization_results if v.created_at >= cutoff_date]
            
            analytics = {
                "total_visualizations": len(recent_visualizations),
                "chart_types": {},
                "dashboards": len(self.dashboard_configs),
                "popular_charts": [],
                "creation_trends": {},
                "performance_metrics": {}
            }
            
            # Analyze chart types
            for viz in recent_visualizations:
                chart_type = viz.chart_type.value
                if chart_type not in analytics["chart_types"]:
                    analytics["chart_types"][chart_type] = 0
                analytics["chart_types"][chart_type] += 1
            
            # Find popular charts
            chart_type_counts = analytics["chart_types"]
            if chart_type_counts:
                popular_charts = sorted(chart_type_counts.items(), key=lambda x: x[1], reverse=True)
                analytics["popular_charts"] = [{"type": chart_type, "count": count} for chart_type, count in popular_charts[:5]]
            
            # Analyze creation trends
            daily_creations = defaultdict(int)
            for viz in recent_visualizations:
                date_key = viz.created_at.date()
                daily_creations[date_key] += 1
            
            analytics["creation_trends"] = {
                date.isoformat(): count for date, count in daily_creations.items()
            }
            
            # Performance metrics
            analytics["performance_metrics"] = {
                "average_creation_time": 0.5,  # Simulated
                "cache_hit_rate": 0.85,  # Simulated
                "user_satisfaction": 0.92  # Simulated
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting visualization analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _generate_performance_data(self, model_name: str, metric: str, time_range_days: int) -> Dict[str, Any]:
        """Generate sample performance data"""
        try:
            # Generate timestamps
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_range_days)
            timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
            
            # Generate performance values with some trend and noise
            np.random.seed(hash(model_name) % 2**32)
            base_value = 0.7 + (hash(metric) % 100) / 1000
            trend = np.linspace(0, 0.1, len(timestamps))
            noise = np.random.normal(0, 0.05, len(timestamps))
            values = base_value + trend + noise
            
            # Ensure values are in valid range
            values = np.clip(values, 0, 1)
            
            return {
                "timestamps": [t.isoformat() for t in timestamps],
                "values": values.tolist(),
                "model_name": model_name,
                "metric": metric,
                "time_range_days": time_range_days
            }
            
        except Exception as e:
            logger.error(f"Error generating performance data: {str(e)}")
            return {"timestamps": [], "values": [], "model_name": model_name, "metric": metric}
    
    async def _generate_comparison_data(self, model_names: List[str], metric: str, time_range_days: int) -> Dict[str, Any]:
        """Generate comparison data for multiple models"""
        try:
            data = {
                "timestamps": [],
                "models": {},
                "metric": metric,
                "time_range_days": time_range_days
            }
            
            # Generate timestamps
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_range_days)
            timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
            data["timestamps"] = [t.isoformat() for t in timestamps]
            
            # Generate data for each model
            for model_name in model_names:
                np.random.seed(hash(model_name) % 2**32)
                base_value = 0.6 + (hash(model_name) % 200) / 1000
                trend = np.linspace(0, 0.05, len(timestamps))
                noise = np.random.normal(0, 0.03, len(timestamps))
                values = base_value + trend + noise
                values = np.clip(values, 0, 1)
                
                data["models"][model_name] = values.tolist()
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating comparison data: {str(e)}")
            return {"timestamps": [], "models": {}, "metric": metric}
    
    async def _create_line_chart(self, data: Dict[str, Any], title: str) -> Tuple[str, str]:
        """Create line chart"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data["timestamps"],
                y=data["values"],
                mode='lines+markers',
                name=data.get("model_name", "Model"),
                line=dict(color=self.color_palettes[self.default_theme.value][0], width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=data.get("metric", "Value"),
                template="plotly_white",
                hovermode='x unified',
                showlegend=True
            )
            
            html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
            json_str = fig.to_json()
            
            return html, json_str
            
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            return "<div>Error creating chart</div>", "{}"
    
    async def _create_bar_chart(self, data: Dict[str, Any], title: str) -> Tuple[str, str]:
        """Create bar chart"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=data["timestamps"],
                y=data["values"],
                name=data.get("model_name", "Model"),
                marker_color=self.color_palettes[self.default_theme.value][0]
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=data.get("metric", "Value"),
                template="plotly_white",
                hovermode='x unified'
            )
            
            html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
            json_str = fig.to_json()
            
            return html, json_str
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return "<div>Error creating chart</div>", "{}"
    
    async def _create_area_chart(self, data: Dict[str, Any], title: str) -> Tuple[str, str]:
        """Create area chart"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data["timestamps"],
                y=data["values"],
                mode='lines',
                fill='tonexty',
                name=data.get("model_name", "Model"),
                line=dict(color=self.color_palettes[self.default_theme.value][0]),
                fillcolor=f"rgba({int(self.color_palettes[self.default_theme.value][0][1:3], 16)}, {int(self.color_palettes[self.default_theme.value][0][3:5], 16)}, {int(self.color_palettes[self.default_theme.value][0][5:7], 16)}, 0.3)"
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=data.get("metric", "Value"),
                template="plotly_white",
                hovermode='x unified'
            )
            
            html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
            json_str = fig.to_json()
            
            return html, json_str
            
        except Exception as e:
            logger.error(f"Error creating area chart: {str(e)}")
            return "<div>Error creating chart</div>", "{}"
    
    async def _create_multi_line_chart(self, data: Dict[str, Any], title: str) -> Tuple[str, str]:
        """Create multi-line chart"""
        try:
            fig = go.Figure()
            
            colors = self.color_palettes[self.default_theme.value]
            for i, (model_name, values) in enumerate(data["models"].items()):
                fig.add_trace(go.Scatter(
                    x=data["timestamps"],
                    y=values,
                    mode='lines+markers',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=data.get("metric", "Value"),
                template="plotly_white",
                hovermode='x unified',
                showlegend=True
            )
            
            html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
            json_str = fig.to_json()
            
            return html, json_str
            
        except Exception as e:
            logger.error(f"Error creating multi-line chart: {str(e)}")
            return "<div>Error creating chart</div>", "{}"
    
    async def _create_grouped_bar_chart(self, data: Dict[str, Any], title: str) -> Tuple[str, str]:
        """Create grouped bar chart"""
        try:
            fig = go.Figure()
            
            colors = self.color_palettes[self.default_theme.value]
            for i, (model_name, values) in enumerate(data["models"].items()):
                fig.add_trace(go.Bar(
                    x=data["timestamps"],
                    y=values,
                    name=model_name,
                    marker_color=colors[i % len(colors)]
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=data.get("metric", "Value"),
                template="plotly_white",
                barmode='group',
                hovermode='x unified'
            )
            
            html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
            json_str = fig.to_json()
            
            return html, json_str
            
        except Exception as e:
            logger.error(f"Error creating grouped bar chart: {str(e)}")
            return "<div>Error creating chart</div>", "{}"
    
    async def _create_radar_chart(self, data: Dict[str, Any], title: str) -> Tuple[str, str]:
        """Create radar chart"""
        try:
            fig = go.Figure()
            
            # Calculate average values for each model
            colors = self.color_palettes[self.default_theme.value]
            for i, (model_name, values) in enumerate(data["models"].items()):
                avg_value = np.mean(values)
                fig.add_trace(go.Scatterpolar(
                    r=[avg_value, avg_value, avg_value, avg_value, avg_value],
                    theta=['Performance', 'Accuracy', 'Speed', 'Efficiency', 'Quality'],
                    fill='toself',
                    name=model_name,
                    line_color=colors[i % len(colors)]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title=title,
                template="plotly_white",
                showlegend=True
            )
            
            html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
            json_str = fig.to_json()
            
            return html, json_str
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {str(e)}")
            return "<div>Error creating chart</div>", "{}"
    
    async def _process_benchmark_data(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process benchmark data for visualization"""
        try:
            data = {
                "models": [],
                "metrics": {},
                "rankings": [],
                "performance": {}
            }
            
            # Extract model names
            model_names = list(set(r.get("model_name", "") for r in benchmark_results if r.get("model_name")))
            data["models"] = model_names
            
            # Process metrics
            for result in benchmark_results:
                model_name = result.get("model_name", "")
                if model_name:
                    if model_name not in data["metrics"]:
                        data["metrics"][model_name] = {}
                    
                    # Extract metrics from result
                    if "metrics" in result:
                        for metric, value in result["metrics"].items():
                            if metric not in data["metrics"][model_name]:
                                data["metrics"][model_name][metric] = []
                            data["metrics"][model_name][metric].append(value)
            
            # Calculate averages
            for model_name, metrics in data["metrics"].items():
                data["performance"][model_name] = {}
                for metric, values in metrics.items():
                    if values:
                        data["performance"][model_name][metric] = np.mean(values)
            
            # Create rankings
            if data["performance"]:
                # Rank by accuracy if available, otherwise by first metric
                ranking_metric = "accuracy"
                if ranking_metric not in list(data["performance"].values())[0]:
                    ranking_metric = list(list(data["performance"].values())[0].keys())[0]
                
                rankings = []
                for model_name, performance in data["performance"].items():
                    if ranking_metric in performance:
                        rankings.append({
                            "model": model_name,
                            "score": performance[ranking_metric],
                            "rank": 0  # Will be set after sorting
                        })
                
                rankings.sort(key=lambda x: x["score"], reverse=True)
                for i, ranking in enumerate(rankings):
                    ranking["rank"] = i + 1
                
                data["rankings"] = rankings
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing benchmark data: {str(e)}")
            return {"models": [], "metrics": {}, "rankings": [], "performance": {}}
    
    async def _create_benchmark_comprehensive_chart(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """Create comprehensive benchmark chart"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Model Performance", "Metrics Comparison", "Rankings", "Performance Distribution"),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "histogram"}]]
            )
            
            # Model performance
            if data["rankings"]:
                models = [r["model"] for r in data["rankings"]]
                scores = [r["score"] for r in data["rankings"]]
                
                fig.add_trace(
                    go.Bar(x=models, y=scores, name="Performance", marker_color=self.color_palettes[self.default_theme.value][0]),
                    row=1, col=1
                )
            
            # Metrics comparison
            if data["performance"]:
                metrics = list(list(data["performance"].values())[0].keys())
                for i, metric in enumerate(metrics[:3]):  # Show top 3 metrics
                    values = [data["performance"].get(model, {}).get(metric, 0) for model in data["models"]]
                    fig.add_trace(
                        go.Bar(x=data["models"], y=values, name=metric, marker_color=self.color_palettes[self.default_theme.value][i]),
                        row=1, col=2
                    )
            
            # Rankings
            if data["rankings"]:
                ranks = [r["rank"] for r in data["rankings"]]
                fig.add_trace(
                    go.Bar(x=[r["model"] for r in data["rankings"]], y=ranks, name="Rank", marker_color=self.color_palettes[self.default_theme.value][2]),
                    row=2, col=1
                )
            
            # Performance distribution
            all_scores = []
            for performance in data["performance"].values():
                all_scores.extend(performance.values())
            
            if all_scores:
                fig.add_trace(
                    go.Histogram(x=all_scores, name="Distribution", marker_color=self.color_palettes[self.default_theme.value][3]),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Comprehensive Benchmark Analysis",
                template="plotly_white",
                showlegend=True,
                height=800
            )
            
            html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
            json_str = fig.to_json()
            
            return html, json_str
            
        except Exception as e:
            logger.error(f"Error creating benchmark comprehensive chart: {str(e)}")
            return "<div>Error creating chart</div>", "{}"
    
    async def _create_benchmark_ranking_chart(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """Create benchmark ranking chart"""
        try:
            fig = go.Figure()
            
            if data["rankings"]:
                models = [r["model"] for r in data["rankings"]]
                scores = [r["score"] for r in data["rankings"]]
                ranks = [r["rank"] for r in data["rankings"]]
                
                fig.add_trace(go.Bar(
                    x=models,
                    y=scores,
                    text=[f"Rank: {rank}" for rank in ranks],
                    textposition='auto',
                    marker_color=[self.color_palettes[self.default_theme.value][0] if rank == 1 
                                 else self.color_palettes[self.default_theme.value][1] if rank == 2
                                 else self.color_palettes[self.default_theme.value][2] for rank in ranks]
                ))
                
                fig.update_layout(
                    title="Model Rankings",
                    xaxis_title="Models",
                    yaxis_title="Performance Score",
                    template="plotly_white"
                )
            
            html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
            json_str = fig.to_json()
            
            return html, json_str
            
        except Exception as e:
            logger.error(f"Error creating benchmark ranking chart: {str(e)}")
            return "<div>Error creating chart</div>", "{}"
    
    async def _create_benchmark_metrics_chart(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """Create benchmark metrics chart"""
        try:
            fig = go.Figure()
            
            if data["performance"]:
                metrics = list(list(data["performance"].values())[0].keys())
                colors = self.color_palettes[self.default_theme.value]
                
                for i, metric in enumerate(metrics):
                    values = [data["performance"].get(model, {}).get(metric, 0) for model in data["models"]]
                    fig.add_trace(go.Bar(
                        x=data["models"],
                        y=values,
                        name=metric,
                        marker_color=colors[i % len(colors)]
                    ))
                
                fig.update_layout(
                    title="Metrics Comparison",
                    xaxis_title="Models",
                    yaxis_title="Metric Values",
                    template="plotly_white",
                    barmode='group'
                )
            
            html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
            json_str = fig.to_json()
            
            return html, json_str
            
        except Exception as e:
            logger.error(f"Error creating benchmark metrics chart: {str(e)}")
            return "<div>Error creating chart</div>", "{}"


# Global visualization engine instance
_visualization_engine: Optional[DataVisualizationEngine] = None


def get_data_visualization_engine(max_visualizations: int = 1000) -> DataVisualizationEngine:
    """Get or create global data visualization engine instance"""
    global _visualization_engine
    if _visualization_engine is None:
        _visualization_engine = DataVisualizationEngine(max_visualizations)
    return _visualization_engine


# Example usage
async def main():
    """Example usage of the data visualization engine"""
    engine = get_data_visualization_engine()
    
    # Create performance chart
    perf_chart = await engine.create_performance_chart(
        model_name="gpt-4",
        metric="accuracy",
        time_range_days=30,
        chart_type=ChartType.LINE
    )
    print(f"Created performance chart: {perf_chart.visualization_id}")
    
    # Create comparison chart
    comp_chart = await engine.create_comparison_chart(
        model_names=["gpt-4", "claude-3", "gemini-pro"],
        metric="performance_score",
        time_range_days=30,
        chart_type=ChartType.LINE
    )
    print(f"Created comparison chart: {comp_chart.visualization_id}")
    
    # Create analytics dashboard
    dashboard = await engine.create_analytics_dashboard(
        dashboard_type=DashboardType.ANALYTICS,
        model_names=["gpt-4", "claude-3", "gemini-pro"],
        time_range_days=30
    )
    print(f"Created dashboard: {dashboard.dashboard_id}")
    
    # Create real-time dashboard
    rt_dashboard = await engine.create_real_time_dashboard(
        model_names=["gpt-4", "claude-3"],
        update_interval=5
    )
    print(f"Created real-time dashboard: {rt_dashboard.dashboard_id}")
    
    # Create benchmark visualization
    benchmark_results = [
        {"model_name": "RandomForest", "metrics": {"accuracy": 0.85, "f1_score": 0.82}},
        {"model_name": "LogisticRegression", "metrics": {"accuracy": 0.78, "f1_score": 0.75}},
        {"model_name": "SVC", "metrics": {"accuracy": 0.80, "f1_score": 0.78}}
    ]
    
    benchmark_viz = await engine.create_benchmark_visualization(
        benchmark_results=benchmark_results,
        visualization_type="comprehensive"
    )
    print(f"Created benchmark visualization: {benchmark_viz.visualization_id}")
    
    # Export visualization
    export_data = await engine.export_visualization(
        visualization_id=perf_chart.visualization_id,
        format="html",
        include_data=True
    )
    print(f"Exported visualization: {len(export_data.get('html', ''))} characters")
    
    # Get visualization analytics
    analytics = await engine.get_visualization_analytics()
    print(f"Visualization analytics: {analytics.get('total_visualizations', 0)} visualizations")


if __name__ == "__main__":
    asyncio.run(main())

























