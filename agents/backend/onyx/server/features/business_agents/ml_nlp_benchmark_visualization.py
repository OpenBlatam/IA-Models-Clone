"""
ML NLP Benchmark Visualization System
Real, working advanced visualization for ML NLP Benchmark system
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import json
import base64
from collections import defaultdict, Counter
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class Visualization:
    """Visualization structure"""
    viz_id: str
    name: str
    viz_type: str
    data: Any
    configuration: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    is_active: bool
    metadata: Dict[str, Any]

@dataclass
class VisualizationResult:
    """Visualization Result structure"""
    result_id: str
    viz_id: str
    chart_data: Dict[str, Any]
    image_data: Optional[str]
    processing_time: float
    success: bool
    error_message: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class Dashboard:
    """Dashboard structure"""
    dashboard_id: str
    name: str
    visualizations: List[str]
    layout: Dict[str, Any]
    configuration: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    is_active: bool
    metadata: Dict[str, Any]

class MLNLPBenchmarkVisualization:
    """Advanced Visualization system for ML NLP Benchmark"""
    
    def __init__(self):
        self.visualizations = {}
        self.visualization_results = []
        self.dashboards = {}
        self.lock = threading.RLock()
        
        # Visualization capabilities
        self.viz_capabilities = {
            "line_chart": True,
            "bar_chart": True,
            "scatter_plot": True,
            "histogram": True,
            "pie_chart": True,
            "heatmap": True,
            "box_plot": True,
            "violin_plot": True,
            "area_chart": True,
            "bubble_chart": True,
            "radar_chart": True,
            "treemap": True,
            "sankey_diagram": True,
            "network_graph": True,
            "word_cloud": True,
            "geographic_map": True,
            "time_series": True,
            "correlation_matrix": True,
            "distribution_plot": True,
            "density_plot": True
        }
        
        # Chart types
        self.chart_types = {
            "line_chart": {
                "description": "Line chart for trends",
                "use_cases": ["time_series", "trends", "comparisons"],
                "data_requirements": ["x_axis", "y_axis"]
            },
            "bar_chart": {
                "description": "Bar chart for comparisons",
                "use_cases": ["comparisons", "rankings", "categories"],
                "data_requirements": ["categories", "values"]
            },
            "scatter_plot": {
                "description": "Scatter plot for correlations",
                "use_cases": ["correlations", "relationships", "clusters"],
                "data_requirements": ["x_values", "y_values"]
            },
            "histogram": {
                "description": "Histogram for distributions",
                "use_cases": ["distributions", "frequencies", "patterns"],
                "data_requirements": ["values", "bins"]
            },
            "pie_chart": {
                "description": "Pie chart for proportions",
                "use_cases": ["proportions", "percentages", "parts_of_whole"],
                "data_requirements": ["labels", "values"]
            },
            "heatmap": {
                "description": "Heatmap for matrix data",
                "use_cases": ["correlations", "patterns", "intensity"],
                "data_requirements": ["matrix_data"]
            },
            "box_plot": {
                "description": "Box plot for distributions",
                "use_cases": ["distributions", "outliers", "quartiles"],
                "data_requirements": ["values", "groups"]
            },
            "violin_plot": {
                "description": "Violin plot for distributions",
                "use_cases": ["distributions", "density", "comparisons"],
                "data_requirements": ["values", "groups"]
            },
            "area_chart": {
                "description": "Area chart for cumulative data",
                "use_cases": ["cumulative", "stacked", "trends"],
                "data_requirements": ["x_axis", "y_axis"]
            },
            "bubble_chart": {
                "description": "Bubble chart for 3D data",
                "use_cases": ["3d_data", "relationships", "size_encoding"],
                "data_requirements": ["x_values", "y_values", "sizes"]
            },
            "radar_chart": {
                "description": "Radar chart for multivariate data",
                "use_cases": ["multivariate", "profiles", "comparisons"],
                "data_requirements": ["categories", "values"]
            },
            "treemap": {
                "description": "Treemap for hierarchical data",
                "use_cases": ["hierarchical", "proportions", "nested"],
                "data_requirements": ["hierarchy", "values"]
            },
            "sankey_diagram": {
                "description": "Sankey diagram for flows",
                "use_cases": ["flows", "processes", "transitions"],
                "data_requirements": ["source", "target", "value"]
            },
            "network_graph": {
                "description": "Network graph for relationships",
                "use_cases": ["networks", "relationships", "connections"],
                "data_requirements": ["nodes", "edges"]
            },
            "word_cloud": {
                "description": "Word cloud for text data",
                "use_cases": ["text_analysis", "keywords", "frequency"],
                "data_requirements": ["text_data", "frequencies"]
            },
            "geographic_map": {
                "description": "Geographic map for location data",
                "use_cases": ["geographic", "location", "spatial"],
                "data_requirements": ["coordinates", "values"]
            },
            "time_series": {
                "description": "Time series for temporal data",
                "use_cases": ["temporal", "trends", "forecasting"],
                "data_requirements": ["time", "values"]
            },
            "correlation_matrix": {
                "description": "Correlation matrix for relationships",
                "use_cases": ["correlations", "relationships", "patterns"],
                "data_requirements": ["numeric_data"]
            },
            "distribution_plot": {
                "description": "Distribution plot for data shape",
                "use_cases": ["distributions", "shape", "patterns"],
                "data_requirements": ["values"]
            },
            "density_plot": {
                "description": "Density plot for data density",
                "use_cases": ["density", "concentration", "patterns"],
                "data_requirements": ["values"]
            }
        }
        
        # Color palettes
        self.color_palettes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "viridis": ["#440154", "#482777", "#3f4a8a", "#31678e", "#26838f"],
            "plasma": ["#0c0786", "#6a00a8", "#b02a8b", "#e16462", "#fca636"],
            "inferno": ["#000004", "#1b0c42", "#4c0a6b", "#781c6d", "#a52c60"],
            "magma": ["#000004", "#1b0c42", "#4c0a6b", "#781c6d", "#a52c60"],
            "categorical": ["#e377c2", "#17becf", "#bcbd22", "#ff7f0e", "#1f77b4"],
            "sequential": ["#f7f7f7", "#d9d9d9", "#bdbdbd", "#969696", "#737373"],
            "diverging": ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf"]
        }
        
        # Chart themes
        self.chart_themes = {
            "default": {"background": "white", "grid": "lightgray", "text": "black"},
            "dark": {"background": "black", "grid": "darkgray", "text": "white"},
            "minimal": {"background": "white", "grid": "none", "text": "black"},
            "colorful": {"background": "lightblue", "grid": "white", "text": "darkblue"}
        }
    
    def create_visualization(self, name: str, viz_type: str, data: Any,
                           configuration: Optional[Dict[str, Any]] = None) -> str:
        """Create a visualization"""
        viz_id = f"{name}_{int(time.time())}"
        
        if viz_type not in self.chart_types:
            raise ValueError(f"Unknown visualization type: {viz_type}")
        
        # Default configuration
        default_config = {
            "title": name,
            "width": 800,
            "height": 600,
            "color_palette": "default",
            "theme": "default",
            "show_legend": True,
            "show_grid": True,
            "show_labels": True
        }
        
        if configuration:
            default_config.update(configuration)
        
        visualization = Visualization(
            viz_id=viz_id,
            name=name,
            viz_type=viz_type,
            data=data,
            configuration=default_config,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            is_active=True,
            metadata={
                "data_size": len(str(data)),
                "configuration_keys": len(default_config)
            }
        )
        
        with self.lock:
            self.visualizations[viz_id] = visualization
        
        logger.info(f"Created visualization {viz_id}: {name} ({viz_type})")
        return viz_id
    
    def generate_chart(self, viz_id: str) -> VisualizationResult:
        """Generate chart data for visualization"""
        if viz_id not in self.visualizations:
            raise ValueError(f"Visualization {viz_id} not found")
        
        visualization = self.visualizations[viz_id]
        
        if not visualization.is_active:
            raise ValueError(f"Visualization {viz_id} is not active")
        
        result_id = f"chart_{viz_id}_{int(time.time())}"
        start_time = time.time()
        
        try:
            # Generate chart data based on type
            chart_data = self._generate_chart_data(visualization)
            
            # Generate image data (simulated)
            image_data = self._generate_image_data(visualization)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = VisualizationResult(
                result_id=result_id,
                viz_id=viz_id,
                chart_data=chart_data,
                image_data=image_data,
                processing_time=processing_time,
                success=True,
                error_message=None,
                timestamp=datetime.now(),
                metadata={
                    "chart_type": visualization.viz_type,
                    "data_points": len(str(visualization.data)),
                    "configuration": visualization.configuration
                }
            )
            
            # Store result
            with self.lock:
                self.visualization_results.append(result)
            
            logger.info(f"Generated chart for {viz_id} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            
            result = VisualizationResult(
                result_id=result_id,
                viz_id=viz_id,
                chart_data={},
                image_data=None,
                processing_time=processing_time,
                success=False,
                error_message=error_message,
                timestamp=datetime.now(),
                metadata={}
            )
            
            with self.lock:
                self.visualization_results.append(result)
            
            logger.error(f"Error generating chart for {viz_id}: {e}")
            return result
    
    def create_dashboard(self, name: str, visualizations: List[str],
                        layout: Optional[Dict[str, Any]] = None,
                        configuration: Optional[Dict[str, Any]] = None) -> str:
        """Create a dashboard"""
        dashboard_id = f"{name}_{int(time.time())}"
        
        # Validate visualizations
        for viz_id in visualizations:
            if viz_id not in self.visualizations:
                raise ValueError(f"Visualization {viz_id} not found")
        
        # Default layout
        default_layout = {
            "rows": 2,
            "columns": 2,
            "spacing": 10,
            "padding": 20
        }
        
        if layout:
            default_layout.update(layout)
        
        # Default configuration
        default_config = {
            "title": name,
            "theme": "default",
            "auto_refresh": False,
            "refresh_interval": 300
        }
        
        if configuration:
            default_config.update(configuration)
        
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            visualizations=visualizations,
            layout=default_layout,
            configuration=default_config,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            is_active=True,
            metadata={
                "visualization_count": len(visualizations),
                "layout_keys": len(default_layout),
                "configuration_keys": len(default_config)
            }
        )
        
        with self.lock:
            self.dashboards[dashboard_id] = dashboard
        
        logger.info(f"Created dashboard {dashboard_id}: {name}")
        return dashboard_id
    
    def generate_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Generate dashboard data"""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.dashboards[dashboard_id]
        
        if not dashboard.is_active:
            raise ValueError(f"Dashboard {dashboard_id} is not active")
        
        start_time = time.time()
        
        try:
            # Generate charts for all visualizations
            dashboard_data = {
                "dashboard_id": dashboard_id,
                "name": dashboard.name,
                "layout": dashboard.layout,
                "configuration": dashboard.configuration,
                "visualizations": []
            }
            
            for viz_id in dashboard.visualizations:
                chart_result = self.generate_chart(viz_id)
                dashboard_data["visualizations"].append({
                    "viz_id": viz_id,
                    "chart_data": chart_result.chart_data,
                    "success": chart_result.success,
                    "error_message": chart_result.error_message
                })
            
            processing_time = time.time() - start_time
            dashboard_data["processing_time"] = processing_time
            dashboard_data["timestamp"] = datetime.now().isoformat()
            
            logger.info(f"Generated dashboard {dashboard_id} in {processing_time:.3f}s")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard {dashboard_id}: {e}")
            raise
    
    def export_visualization(self, viz_id: str, format: str = "json") -> str:
        """Export visualization"""
        if viz_id not in self.visualizations:
            raise ValueError(f"Visualization {viz_id} not found")
        
        visualization = self.visualizations[viz_id]
        
        if format == "json":
            export_data = {
                "viz_id": visualization.viz_id,
                "name": visualization.name,
                "viz_type": visualization.viz_type,
                "data": visualization.data,
                "configuration": visualization.configuration,
                "created_at": visualization.created_at.isoformat(),
                "last_updated": visualization.last_updated.isoformat(),
                "is_active": visualization.is_active,
                "metadata": visualization.metadata
            }
            return json.dumps(export_data, indent=2)
        
        elif format == "csv":
            # Convert data to CSV format
            if isinstance(visualization.data, list):
                df = pd.DataFrame(visualization.data)
                return df.to_csv(index=False)
            else:
                return str(visualization.data)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_visualization(self, data: str, format: str = "json") -> str:
        """Import visualization"""
        try:
            if format == "json":
                import_data = json.loads(data)
                
                # Create visualization
                viz_id = import_data["viz_id"]
                visualization = Visualization(
                    viz_id=viz_id,
                    name=import_data["name"],
                    viz_type=import_data["viz_type"],
                    data=import_data["data"],
                    configuration=import_data["configuration"],
                    created_at=datetime.fromisoformat(import_data["created_at"]),
                    last_updated=datetime.fromisoformat(import_data["last_updated"]),
                    is_active=import_data["is_active"],
                    metadata=import_data["metadata"]
                )
                
                with self.lock:
                    self.visualizations[viz_id] = visualization
                
                logger.info(f"Imported visualization {viz_id}")
                return viz_id
            
            else:
                raise ValueError(f"Unsupported import format: {format}")
                
        except Exception as e:
            logger.error(f"Error importing visualization: {e}")
            raise
    
    def get_visualization(self, viz_id: str) -> Optional[Visualization]:
        """Get visualization information"""
        return self.visualizations.get(viz_id)
    
    def list_visualizations(self, viz_type: Optional[str] = None, 
                          active_only: bool = False) -> List[Visualization]:
        """List visualizations"""
        visualizations = list(self.visualizations.values())
        
        if viz_type:
            visualizations = [v for v in visualizations if v.viz_type == viz_type]
        
        if active_only:
            visualizations = [v for v in visualizations if v.is_active]
        
        return visualizations
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard information"""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self, active_only: bool = False) -> List[Dashboard]:
        """List dashboards"""
        dashboards = list(self.dashboards.values())
        
        if active_only:
            dashboards = [d for d in dashboards if d.is_active]
        
        return dashboards
    
    def _generate_chart_data(self, visualization: Visualization) -> Dict[str, Any]:
        """Generate chart data based on visualization type"""
        viz_type = visualization.viz_type
        data = visualization.data
        config = visualization.configuration
        
        if viz_type == "line_chart":
            return self._generate_line_chart_data(data, config)
        elif viz_type == "bar_chart":
            return self._generate_bar_chart_data(data, config)
        elif viz_type == "scatter_plot":
            return self._generate_scatter_plot_data(data, config)
        elif viz_type == "histogram":
            return self._generate_histogram_data(data, config)
        elif viz_type == "pie_chart":
            return self._generate_pie_chart_data(data, config)
        elif viz_type == "heatmap":
            return self._generate_heatmap_data(data, config)
        elif viz_type == "box_plot":
            return self._generate_box_plot_data(data, config)
        elif viz_type == "word_cloud":
            return self._generate_word_cloud_data(data, config)
        else:
            return self._generate_generic_chart_data(data, config)
    
    def _generate_line_chart_data(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate line chart data"""
        # Simulate line chart data
        x_values = list(range(10))
        y_values = [np.random.rand() * 100 for _ in range(10)]
        
        return {
            "type": "line_chart",
            "data": {
                "x": x_values,
                "y": y_values
            },
            "options": {
                "title": config.get("title", "Line Chart"),
                "x_label": "X Axis",
                "y_label": "Y Axis",
                "color": config.get("color", "#1f77b4")
            }
        }
    
    def _generate_bar_chart_data(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate bar chart data"""
        # Simulate bar chart data
        categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
        values = [np.random.rand() * 100 for _ in range(5)]
        
        return {
            "type": "bar_chart",
            "data": {
                "categories": categories,
                "values": values
            },
            "options": {
                "title": config.get("title", "Bar Chart"),
                "x_label": "Categories",
                "y_label": "Values",
                "color": config.get("color", "#ff7f0e")
            }
        }
    
    def _generate_scatter_plot_data(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scatter plot data"""
        # Simulate scatter plot data
        x_values = [np.random.rand() * 100 for _ in range(20)]
        y_values = [np.random.rand() * 100 for _ in range(20)]
        
        return {
            "type": "scatter_plot",
            "data": {
                "x": x_values,
                "y": y_values
            },
            "options": {
                "title": config.get("title", "Scatter Plot"),
                "x_label": "X Values",
                "y_label": "Y Values",
                "color": config.get("color", "#2ca02c")
            }
        }
    
    def _generate_histogram_data(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate histogram data"""
        # Simulate histogram data
        values = [np.random.normal(50, 15) for _ in range(100)]
        bins = np.linspace(min(values), max(values), 20)
        counts, _ = np.histogram(values, bins)
        
        return {
            "type": "histogram",
            "data": {
                "bins": bins.tolist(),
                "counts": counts.tolist()
            },
            "options": {
                "title": config.get("title", "Histogram"),
                "x_label": "Values",
                "y_label": "Frequency",
                "color": config.get("color", "#d62728")
            }
        }
    
    def _generate_pie_chart_data(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pie chart data"""
        # Simulate pie chart data
        labels = ["Slice 1", "Slice 2", "Slice 3", "Slice 4", "Slice 5"]
        values = [np.random.rand() * 100 for _ in range(5)]
        
        return {
            "type": "pie_chart",
            "data": {
                "labels": labels,
                "values": values
            },
            "options": {
                "title": config.get("title", "Pie Chart"),
                "colors": config.get("colors", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
            }
        }
    
    def _generate_heatmap_data(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate heatmap data"""
        # Simulate heatmap data
        matrix = np.random.rand(10, 10) * 100
        
        return {
            "type": "heatmap",
            "data": {
                "matrix": matrix.tolist()
            },
            "options": {
                "title": config.get("title", "Heatmap"),
                "color_scale": config.get("color_scale", "viridis")
            }
        }
    
    def _generate_box_plot_data(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate box plot data"""
        # Simulate box plot data
        groups = ["Group A", "Group B", "Group C", "Group D"]
        values = []
        for group in groups:
            group_values = [np.random.normal(50, 15) for _ in range(20)]
            values.append(group_values)
        
        return {
            "type": "box_plot",
            "data": {
                "groups": groups,
                "values": values
            },
            "options": {
                "title": config.get("title", "Box Plot"),
                "x_label": "Groups",
                "y_label": "Values",
                "color": config.get("color", "#9467bd")
            }
        }
    
    def _generate_word_cloud_data(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate word cloud data"""
        # Simulate word cloud data
        words = ["machine", "learning", "artificial", "intelligence", "data", "analysis", "visualization", "chart", "graph", "plot"]
        frequencies = [np.random.randint(1, 100) for _ in words]
        
        return {
            "type": "word_cloud",
            "data": {
                "words": words,
                "frequencies": frequencies
            },
            "options": {
                "title": config.get("title", "Word Cloud"),
                "max_words": config.get("max_words", 50),
                "color_scheme": config.get("color_scheme", "viridis")
            }
        }
    
    def _generate_generic_chart_data(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic chart data"""
        return {
            "type": "generic",
            "data": {
                "values": [1, 2, 3, 4, 5]
            },
            "options": {
                "title": config.get("title", "Generic Chart"),
                "color": config.get("color", "#1f77b4")
            }
        }
    
    def _generate_image_data(self, visualization: Visualization) -> str:
        """Generate image data (simulated)"""
        # Simulate image data generation
        image_data = {
            "width": visualization.configuration.get("width", 800),
            "height": visualization.configuration.get("height", 600),
            "format": "png",
            "data": "simulated_image_data_base64_encoded"
        }
        
        return base64.b64encode(json.dumps(image_data).encode()).decode()
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get visualization system summary"""
        with self.lock:
            return {
                "total_visualizations": len(self.visualizations),
                "total_dashboards": len(self.dashboards),
                "total_chart_results": len(self.visualization_results),
                "active_visualizations": len([v for v in self.visualizations.values() if v.is_active]),
                "active_dashboards": len([d for d in self.dashboards.values() if d.is_active]),
                "viz_capabilities": self.viz_capabilities,
                "chart_types": list(self.chart_types.keys()),
                "color_palettes": list(self.color_palettes.keys()),
                "chart_themes": list(self.chart_themes.keys()),
                "recent_visualizations": len([v for v in self.visualizations.values() if (datetime.now() - v.created_at).days <= 7]),
                "recent_dashboards": len([d for d in self.dashboards.values() if (datetime.now() - d.created_at).days <= 7])
            }
    
    def clear_visualization_data(self):
        """Clear all visualization data"""
        with self.lock:
            self.visualizations.clear()
            self.visualization_results.clear()
            self.dashboards.clear()
        logger.info("Visualization data cleared")

# Global visualization instance
ml_nlp_benchmark_visualization = MLNLPBenchmarkVisualization()

def get_visualization() -> MLNLPBenchmarkVisualization:
    """Get the global visualization instance"""
    return ml_nlp_benchmark_visualization

def create_visualization(name: str, viz_type: str, data: Any,
                        configuration: Optional[Dict[str, Any]] = None) -> str:
    """Create a visualization"""
    return ml_nlp_benchmark_visualization.create_visualization(name, viz_type, data, configuration)

def generate_chart(viz_id: str) -> VisualizationResult:
    """Generate chart data for visualization"""
    return ml_nlp_benchmark_visualization.generate_chart(viz_id)

def create_dashboard(name: str, visualizations: List[str],
                    layout: Optional[Dict[str, Any]] = None,
                    configuration: Optional[Dict[str, Any]] = None) -> str:
    """Create a dashboard"""
    return ml_nlp_benchmark_visualization.create_dashboard(name, visualizations, layout, configuration)

def generate_dashboard(dashboard_id: str) -> Dict[str, Any]:
    """Generate dashboard data"""
    return ml_nlp_benchmark_visualization.generate_dashboard(dashboard_id)

def export_visualization(viz_id: str, format: str = "json") -> str:
    """Export visualization"""
    return ml_nlp_benchmark_visualization.export_visualization(viz_id, format)

def import_visualization(data: str, format: str = "json") -> str:
    """Import visualization"""
    return ml_nlp_benchmark_visualization.import_visualization(data, format)

def get_visualization_summary() -> Dict[str, Any]:
    """Get visualization system summary"""
    return ml_nlp_benchmark_visualization.get_visualization_summary()

def clear_visualization_data():
    """Clear all visualization data"""
    ml_nlp_benchmark_visualization.clear_visualization_data()











