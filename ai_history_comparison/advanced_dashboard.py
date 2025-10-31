"""
Advanced Interactive Dashboard System for AI History Comparison
Sistema de dashboard interactivo avanzado para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import base64
import io
from pathlib import Path

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartType(Enum):
    """Tipos de gráficos"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    AREA = "area"
    CANDLESTICK = "candlestick"

class DashboardTheme(Enum):
    """Temas del dashboard"""
    LIGHT = "light"
    DARK = "dark"
    BLUE = "blue"
    GREEN = "green"
    PURPLE = "purple"
    CORPORATE = "corporate"

@dataclass
class DashboardWidget:
    """Widget del dashboard"""
    id: str
    title: str
    chart_type: ChartType
    data: Dict[str, Any]
    position: Tuple[int, int]
    size: Tuple[int, int]
    config: Dict[str, Any]
    refresh_interval: int = 300  # segundos
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class DashboardLayout:
    """Layout del dashboard"""
    id: str
    name: str
    description: str
    widgets: List[DashboardWidget]
    theme: DashboardTheme
    grid_size: Tuple[int, int]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedDashboard:
    """
    Dashboard interactivo avanzado
    """
    
    def __init__(
        self,
        enable_real_time: bool = True,
        enable_interactive_charts: bool = True,
        default_theme: DashboardTheme = DashboardTheme.LIGHT
    ):
        self.enable_real_time = enable_real_time
        self.enable_interactive_charts = enable_interactive_charts
        self.default_theme = default_theme
        
        # Layouts del dashboard
        self.layouts: Dict[str, DashboardLayout] = {}
        
        # Widgets disponibles
        self.widgets: Dict[str, DashboardWidget] = {}
        
        # Configuración de temas
        self.themes = self._initialize_themes()
        
        # Configuración
        self.config = {
            "max_widgets_per_layout": 20,
            "default_refresh_interval": 300,
            "chart_width": 800,
            "chart_height": 600,
            "export_formats": ["png", "svg", "pdf", "html"],
            "max_data_points": 1000
        }
        
        # Inicializar layouts por defecto
        self._initialize_default_layouts()
    
    def _initialize_themes(self) -> Dict[DashboardTheme, Dict[str, Any]]:
        """Inicializar temas del dashboard"""
        return {
            DashboardTheme.LIGHT: {
                "background_color": "#ffffff",
                "text_color": "#333333",
                "primary_color": "#007bff",
                "secondary_color": "#6c757d",
                "success_color": "#28a745",
                "warning_color": "#ffc107",
                "danger_color": "#dc3545",
                "grid_color": "#e9ecef",
                "plot_style": "whitegrid"
            },
            DashboardTheme.DARK: {
                "background_color": "#1a1a1a",
                "text_color": "#ffffff",
                "primary_color": "#0d6efd",
                "secondary_color": "#6c757d",
                "success_color": "#198754",
                "warning_color": "#fd7e14",
                "danger_color": "#dc3545",
                "grid_color": "#343a40",
                "plot_style": "darkgrid"
            },
            DashboardTheme.CORPORATE: {
                "background_color": "#f8f9fa",
                "text_color": "#212529",
                "primary_color": "#0d6efd",
                "secondary_color": "#6c757d",
                "success_color": "#198754",
                "warning_color": "#ffc107",
                "danger_color": "#dc3545",
                "grid_color": "#dee2e6",
                "plot_style": "whitegrid"
            }
        }
    
    def _initialize_default_layouts(self):
        """Inicializar layouts por defecto"""
        
        # Layout principal
        main_layout = DashboardLayout(
            id="main_dashboard",
            name="Dashboard Principal",
            description="Dashboard principal con métricas clave",
            widgets=[],
            theme=self.default_theme,
            grid_size=(4, 3)
        )
        
        # Layout de análisis
        analysis_layout = DashboardLayout(
            id="analysis_dashboard",
            name="Dashboard de Análisis",
            description="Dashboard para análisis detallado",
            widgets=[],
            theme=self.default_theme,
            grid_size=(3, 4)
        )
        
        # Layout de tendencias
        trends_layout = DashboardLayout(
            id="trends_dashboard",
            name="Dashboard de Tendencias",
            description="Dashboard para análisis de tendencias",
            widgets=[],
            theme=self.default_theme,
            grid_size=(2, 3)
        )
        
        self.layouts = {
            "main": main_layout,
            "analysis": analysis_layout,
            "trends": trends_layout
        }
    
    async def create_widget(
        self,
        widget_id: str,
        title: str,
        chart_type: ChartType,
        data: Dict[str, Any],
        position: Tuple[int, int] = (0, 0),
        size: Tuple[int, int] = (1, 1),
        config: Optional[Dict[str, Any]] = None
    ) -> DashboardWidget:
        """
        Crear un widget del dashboard
        
        Args:
            widget_id: ID único del widget
            title: Título del widget
            chart_type: Tipo de gráfico
            data: Datos para el gráfico
            position: Posición en la grilla
            size: Tamaño en la grilla
            config: Configuración adicional
            
        Returns:
            Widget creado
        """
        try:
            logger.info(f"Creating widget: {widget_id}")
            
            widget_config = config or {}
            
            widget = DashboardWidget(
                id=widget_id,
                title=title,
                chart_type=chart_type,
                data=data,
                position=position,
                size=size,
                config=widget_config
            )
            
            self.widgets[widget_id] = widget
            
            logger.info(f"Widget created successfully: {widget_id}")
            return widget
            
        except Exception as e:
            logger.error(f"Error creating widget: {e}")
            raise
    
    async def generate_chart(
        self,
        widget: DashboardWidget,
        theme: Optional[DashboardTheme] = None
    ) -> str:
        """
        Generar gráfico para un widget
        
        Args:
            widget: Widget del dashboard
            theme: Tema a usar
            
        Returns:
            HTML del gráfico
        """
        try:
            theme = theme or self.default_theme
            theme_config = self.themes[theme]
            
            if self.enable_interactive_charts:
                return await self._generate_interactive_chart(widget, theme_config)
            else:
                return await self._generate_static_chart(widget, theme_config)
                
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            raise
    
    async def _generate_interactive_chart(
        self,
        widget: DashboardWidget,
        theme_config: Dict[str, Any]
    ) -> str:
        """Generar gráfico interactivo con Plotly"""
        
        data = widget.data
        chart_type = widget.chart_type
        
        if chart_type == ChartType.LINE:
            fig = self._create_line_chart(data, theme_config)
        elif chart_type == ChartType.BAR:
            fig = self._create_bar_chart(data, theme_config)
        elif chart_type == ChartType.SCATTER:
            fig = self._create_scatter_chart(data, theme_config)
        elif chart_type == ChartType.HEATMAP:
            fig = self._create_heatmap_chart(data, theme_config)
        elif chart_type == ChartType.PIE:
            fig = self._create_pie_chart(data, theme_config)
        elif chart_type == ChartType.HISTOGRAM:
            fig = self._create_histogram_chart(data, theme_config)
        elif chart_type == ChartType.BOX:
            fig = self._create_box_chart(data, theme_config)
        elif chart_type == ChartType.AREA:
            fig = self._create_area_chart(data, theme_config)
        else:
            fig = self._create_line_chart(data, theme_config)
        
        # Aplicar tema
        fig.update_layout(
            title=widget.title,
            plot_bgcolor=theme_config["background_color"],
            paper_bgcolor=theme_config["background_color"],
            font_color=theme_config["text_color"],
            width=self.config["chart_width"],
            height=self.config["chart_height"]
        )
        
        # Convertir a HTML
        html = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        return html
    
    async def _generate_static_chart(
        self,
        widget: DashboardWidget,
        theme_config: Dict[str, Any]
    ) -> str:
        """Generar gráfico estático con Matplotlib"""
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8' if theme_config["plot_style"] == "whitegrid" else 'dark_background')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data = widget.data
        chart_type = widget.chart_type
        
        if chart_type == ChartType.LINE:
            self._plot_line_chart(ax, data, theme_config)
        elif chart_type == ChartType.BAR:
            self._plot_bar_chart(ax, data, theme_config)
        elif chart_type == ChartType.SCATTER:
            self._plot_scatter_chart(ax, data, theme_config)
        elif chart_type == ChartType.HISTOGRAM:
            self._plot_histogram_chart(ax, data, theme_config)
        else:
            self._plot_line_chart(ax, data, theme_config)
        
        ax.set_title(widget.title, color=theme_config["text_color"])
        ax.set_facecolor(theme_config["background_color"])
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', 
                   facecolor=theme_config["background_color"])
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return f'<img src="data:image/png;base64,{image_base64}" alt="{widget.title}">'
    
    def _create_line_chart(self, data: Dict[str, Any], theme_config: Dict[str, Any]) -> go.Figure:
        """Crear gráfico de líneas"""
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", {})
        
        fig = go.Figure()
        
        if isinstance(y[0], list):
            # Múltiples series
            for i, series in enumerate(y):
                name = labels.get(f"series_{i}", f"Serie {i+1}")
                fig.add_trace(go.Scatter(
                    x=x,
                    y=series,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=theme_config["primary_color"])
                ))
        else:
            # Una sola serie
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=labels.get("y", "Valores"),
                line=dict(color=theme_config["primary_color"])
            ))
        
        fig.update_layout(
            xaxis_title=labels.get("x", "Tiempo"),
            yaxis_title=labels.get("y", "Valores")
        )
        
        return fig
    
    def _create_bar_chart(self, data: Dict[str, Any], theme_config: Dict[str, Any]) -> go.Figure:
        """Crear gráfico de barras"""
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", {})
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=x,
            y=y,
            name=labels.get("y", "Valores"),
            marker_color=theme_config["primary_color"]
        ))
        
        fig.update_layout(
            xaxis_title=labels.get("x", "Categorías"),
            yaxis_title=labels.get("y", "Valores")
        )
        
        return fig
    
    def _create_scatter_chart(self, data: Dict[str, Any], theme_config: Dict[str, Any]) -> go.Figure:
        """Crear gráfico de dispersión"""
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", {})
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name=labels.get("y", "Valores"),
            marker=dict(
                color=theme_config["primary_color"],
                size=8
            )
        ))
        
        fig.update_layout(
            xaxis_title=labels.get("x", "X"),
            yaxis_title=labels.get("y", "Y")
        )
        
        return fig
    
    def _create_heatmap_chart(self, data: Dict[str, Any], theme_config: Dict[str, Any]) -> go.Figure:
        """Crear mapa de calor"""
        z = data.get("z", [])
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", {})
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=labels.get("title", "Mapa de Calor"),
            xaxis_title=labels.get("x", "X"),
            yaxis_title=labels.get("y", "Y")
        )
        
        return fig
    
    def _create_pie_chart(self, data: Dict[str, Any], theme_config: Dict[str, Any]) -> go.Figure:
        """Crear gráfico de pastel"""
        labels = data.get("labels", [])
        values = data.get("values", [])
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3
        )])
        
        return fig
    
    def _create_histogram_chart(self, data: Dict[str, Any], theme_config: Dict[str, Any]) -> go.Figure:
        """Crear histograma"""
        values = data.get("values", [])
        labels = data.get("labels", {})
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=values,
            name=labels.get("x", "Distribución"),
            marker_color=theme_config["primary_color"]
        ))
        
        fig.update_layout(
            xaxis_title=labels.get("x", "Valores"),
            yaxis_title=labels.get("y", "Frecuencia")
        )
        
        return fig
    
    def _create_box_chart(self, data: Dict[str, Any], theme_config: Dict[str, Any]) -> go.Figure:
        """Crear gráfico de cajas"""
        values = data.get("values", [])
        labels = data.get("labels", [])
        
        fig = go.Figure()
        
        if isinstance(values[0], list):
            # Múltiples series
            for i, series in enumerate(values):
                name = labels[i] if i < len(labels) else f"Serie {i+1}"
                fig.add_trace(go.Box(
                    y=series,
                    name=name
                ))
        else:
            # Una sola serie
            fig.add_trace(go.Box(
                y=values,
                name=labels[0] if labels else "Distribución"
            ))
        
        return fig
    
    def _create_area_chart(self, data: Dict[str, Any], theme_config: Dict[str, Any]) -> go.Figure:
        """Crear gráfico de área"""
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", {})
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            fill='tonexty',
            name=labels.get("y", "Valores"),
            line=dict(color=theme_config["primary_color"])
        ))
        
        fig.update_layout(
            xaxis_title=labels.get("x", "Tiempo"),
            yaxis_title=labels.get("y", "Valores")
        )
        
        return fig
    
    def _plot_line_chart(self, ax, data: Dict[str, Any], theme_config: Dict[str, Any]):
        """Plotear gráfico de líneas con Matplotlib"""
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", {})
        
        if isinstance(y[0], list):
            for i, series in enumerate(y):
                name = labels.get(f"series_{i}", f"Serie {i+1}")
                ax.plot(x, series, label=name, color=theme_config["primary_color"])
        else:
            ax.plot(x, y, label=labels.get("y", "Valores"), color=theme_config["primary_color"])
        
        ax.set_xlabel(labels.get("x", "Tiempo"), color=theme_config["text_color"])
        ax.set_ylabel(labels.get("y", "Valores"), color=theme_config["text_color"])
        ax.legend()
        ax.grid(True, color=theme_config["grid_color"])
    
    def _plot_bar_chart(self, ax, data: Dict[str, Any], theme_config: Dict[str, Any]):
        """Plotear gráfico de barras con Matplotlib"""
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", {})
        
        ax.bar(x, y, color=theme_config["primary_color"])
        ax.set_xlabel(labels.get("x", "Categorías"), color=theme_config["text_color"])
        ax.set_ylabel(labels.get("y", "Valores"), color=theme_config["text_color"])
        ax.grid(True, color=theme_config["grid_color"])
    
    def _plot_scatter_chart(self, ax, data: Dict[str, Any], theme_config: Dict[str, Any]):
        """Plotear gráfico de dispersión con Matplotlib"""
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", {})
        
        ax.scatter(x, y, color=theme_config["primary_color"], alpha=0.7)
        ax.set_xlabel(labels.get("x", "X"), color=theme_config["text_color"])
        ax.set_ylabel(labels.get("y", "Y"), color=theme_config["text_color"])
        ax.grid(True, color=theme_config["grid_color"])
    
    def _plot_histogram_chart(self, ax, data: Dict[str, Any], theme_config: Dict[str, Any]):
        """Plotear histograma con Matplotlib"""
        values = data.get("values", [])
        labels = data.get("labels", {})
        
        ax.hist(values, bins=20, color=theme_config["primary_color"], alpha=0.7)
        ax.set_xlabel(labels.get("x", "Valores"), color=theme_config["text_color"])
        ax.set_ylabel(labels.get("y", "Frecuencia"), color=theme_config["text_color"])
        ax.grid(True, color=theme_config["grid_color"])
    
    async def add_widget_to_layout(
        self,
        layout_id: str,
        widget_id: str,
        position: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Agregar widget a un layout
        
        Args:
            layout_id: ID del layout
            widget_id: ID del widget
            position: Posición opcional
            
        Returns:
            True si se agregó exitosamente
        """
        try:
            if layout_id not in self.layouts:
                raise ValueError(f"Layout {layout_id} not found")
            
            if widget_id not in self.widgets:
                raise ValueError(f"Widget {widget_id} not found")
            
            layout = self.layouts[layout_id]
            widget = self.widgets[widget_id]
            
            # Verificar límite de widgets
            if len(layout.widgets) >= self.config["max_widgets_per_layout"]:
                logger.warning(f"Layout {layout_id} has reached maximum widgets")
                return False
            
            # Actualizar posición si se especifica
            if position:
                widget.position = position
            
            # Agregar widget al layout
            layout.widgets.append(widget)
            
            logger.info(f"Widget {widget_id} added to layout {layout_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding widget to layout: {e}")
            return False
    
    async def remove_widget_from_layout(self, layout_id: str, widget_id: str) -> bool:
        """Remover widget de un layout"""
        try:
            if layout_id not in self.layouts:
                return False
            
            layout = self.layouts[layout_id]
            layout.widgets = [w for w in layout.widgets if w.id != widget_id]
            
            logger.info(f"Widget {widget_id} removed from layout {layout_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing widget from layout: {e}")
            return False
    
    async def generate_dashboard_html(
        self,
        layout_id: str,
        theme: Optional[DashboardTheme] = None
    ) -> str:
        """
        Generar HTML completo del dashboard
        
        Args:
            layout_id: ID del layout
            theme: Tema a usar
            
        Returns:
            HTML del dashboard
        """
        try:
            if layout_id not in self.layouts:
                raise ValueError(f"Layout {layout_id} not found")
            
            layout = self.layouts[layout_id]
            theme = theme or layout.theme
            theme_config = self.themes[theme]
            
            # Generar HTML base
            html = self._generate_dashboard_base_html(theme_config)
            
            # Agregar widgets
            widgets_html = ""
            for widget in layout.widgets:
                widget_html = await self.generate_chart(widget, theme)
                widgets_html += f"""
                <div class="widget" style="grid-column: {widget.position[0] + 1} / span {widget.size[0]}; 
                                         grid-row: {widget.position[1] + 1} / span {widget.size[1]};">
                    <div class="widget-header">
                        <h3>{widget.title}</h3>
                        <span class="last-updated">Última actualización: {widget.last_updated.strftime('%H:%M:%S')}</span>
                    </div>
                    <div class="widget-content">
                        {widget_html}
                    </div>
                </div>
                """
            
            # Reemplazar placeholder de widgets
            html = html.replace("{{WIDGETS}}", widgets_html)
            
            # Agregar configuración de grilla
            grid_config = f"grid-template-columns: repeat({layout.grid_size[0]}, 1fr); grid-template-rows: repeat({layout.grid_size[1]}, 1fr);"
            html = html.replace("{{GRID_CONFIG}}", grid_config)
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating dashboard HTML: {e}")
            raise
    
    def _generate_dashboard_base_html(self, theme_config: Dict[str, Any]) -> str:
        """Generar HTML base del dashboard"""
        return f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dashboard Avanzado</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: {theme_config['background_color']};
                    color: {theme_config['text_color']};
                }}
                
                .dashboard {{
                    display: grid;
                    gap: 20px;
                    {{GRID_CONFIG}}
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                
                .widget {{
                    background: {theme_config['background_color']};
                    border: 1px solid {theme_config['grid_color']};
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                
                .widget-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid {theme_config['grid_color']};
                }}
                
                .widget-header h3 {{
                    margin: 0;
                    color: {theme_config['text_color']};
                }}
                
                .last-updated {{
                    font-size: 0.8em;
                    color: {theme_config['secondary_color']};
                }}
                
                .widget-content {{
                    min-height: 300px;
                }}
                
                .dashboard-header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                
                .dashboard-title {{
                    font-size: 2.5em;
                    margin: 0;
                    color: {theme_config['primary_color']};
                }}
                
                .dashboard-subtitle {{
                    font-size: 1.2em;
                    margin: 10px 0;
                    color: {theme_config['secondary_color']};
                }}
                
                .refresh-indicator {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: {theme_config['success_color']};
                    color: white;
                    padding: 10px 15px;
                    border-radius: 5px;
                    font-size: 0.9em;
                    display: none;
                }}
                
                @media (max-width: 768px) {{
                    .dashboard {{
                        grid-template-columns: 1fr;
                        grid-template-rows: auto;
                    }}
                    
                    .widget {{
                        grid-column: 1 !important;
                        grid-row: auto !important;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1 class="dashboard-title">Dashboard Avanzado</h1>
                <p class="dashboard-subtitle">Análisis de Historial de IA en Tiempo Real</p>
            </div>
            
            <div class="dashboard">
                {{WIDGETS}}
            </div>
            
            <div class="refresh-indicator" id="refreshIndicator">
                Actualizando datos...
            </div>
            
            <script>
                // Auto-refresh functionality
                function showRefreshIndicator() {{
                    document.getElementById('refreshIndicator').style.display = 'block';
                    setTimeout(() => {{
                        document.getElementById('refreshIndicator').style.display = 'none';
                    }}, 2000);
                }}
                
                // Auto-refresh every 5 minutes
                setInterval(() => {{
                    showRefreshIndicator();
                    location.reload();
                }}, 300000);
                
                // Resize charts on window resize
                window.addEventListener('resize', function() {{
                    if (typeof Plotly !== 'undefined') {{
                        Plotly.Plots.resize();
                    }}
                }});
            </script>
        </body>
        </html>
        """
    
    async def export_dashboard(
        self,
        layout_id: str,
        format: str = "html",
        filepath: Optional[str] = None
    ) -> str:
        """
        Exportar dashboard
        
        Args:
            layout_id: ID del layout
            format: Formato de exportación
            filepath: Ruta del archivo
            
        Returns:
            Ruta del archivo exportado
        """
        try:
            if format not in self.config["export_formats"]:
                raise ValueError(f"Unsupported export format: {format}")
            
            if filepath is None:
                filepath = f"exports/dashboard_{layout_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            
            # Crear directorio si no existe
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if format == "html":
                content = await self.generate_dashboard_html(layout_id)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                # Para otros formatos, generar como imagen
                await self._export_dashboard_as_image(layout_id, format, filepath)
            
            logger.info(f"Dashboard exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting dashboard: {e}")
            raise
    
    async def _export_dashboard_as_image(self, layout_id: str, format: str, filepath: str):
        """Exportar dashboard como imagen"""
        # Esta es una implementación básica
        # En una implementación real, se usaría una herramienta como Selenium o Playwright
        # para renderizar el HTML y capturar como imagen
        
        logger.warning(f"Image export for format {format} not fully implemented")
        
        # Crear un archivo placeholder
        with open(filepath, 'w') as f:
            f.write(f"Dashboard export placeholder for {layout_id}")
    
    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Obtener resumen del dashboard"""
        return {
            "total_layouts": len(self.layouts),
            "total_widgets": len(self.widgets),
            "layouts": {
                layout_id: {
                    "name": layout.name,
                    "description": layout.description,
                    "widget_count": len(layout.widgets),
                    "theme": layout.theme.value,
                    "grid_size": layout.grid_size
                }
                for layout_id, layout in self.layouts.items()
            },
            "available_themes": [theme.value for theme in DashboardTheme],
            "export_formats": self.config["export_formats"],
            "last_updated": datetime.now().isoformat()
        }

























