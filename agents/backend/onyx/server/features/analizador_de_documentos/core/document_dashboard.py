"""
Document Dashboard - Dashboard Visual Interactivo
==================================================

Dashboard visual para visualización de métricas y análisis.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class DashboardWidget:
    """Widget del dashboard."""
    widget_id: str
    widget_type: str  # 'chart', 'metric', 'table', 'list'
    title: str
    data: Any
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Datos del dashboard."""
    dashboard_id: str
    title: str
    widgets: List[DashboardWidget]
    period: str
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DashboardGenerator:
    """Generador de dashboards."""
    
    def __init__(self, analyzer):
        """Inicializar generador."""
        self.analyzer = analyzer
    
    async def generate_dashboard(
        self,
        period: str = "daily",
        days: int = 7,
        include_widgets: Optional[List[str]] = None
    ) -> DashboardData:
        """
        Generar dashboard completo.
        
        Args:
            period: Período ('daily', 'weekly', 'monthly')
            days: Número de días
            include_widgets: Lista de widgets a incluir (None = todos)
        
        Returns:
            DashboardData con widgets
        """
        widgets = []
        
        # Obtener métricas
        metrics_dashboard = await self.analyzer.generate_metrics_dashboard(period, days)
        
        # Widget: Métricas principales
        widgets.append(DashboardWidget(
            widget_id="main_metrics",
            widget_type="metric",
            title="Métricas Principales",
            data={
                "total_documents": metrics_dashboard.total_documents,
                "total_analyses": metrics_dashboard.total_analyses,
                "avg_quality": metrics_dashboard.average_quality_score,
                "avg_grammar": metrics_dashboard.average_grammar_score,
                "avg_processing_time": metrics_dashboard.average_processing_time
            }
        ))
        
        # Widget: Tendencias de calidad
        if metrics_dashboard.trends.get("quality"):
            widgets.append(DashboardWidget(
                widget_id="quality_trend",
                widget_type="chart",
                title="Tendencia de Calidad",
                data=metrics_dashboard.trends["quality"]["daily"],
                config={"type": "line", "y_label": "Score", "x_label": "Fecha"}
            ))
        
        # Widget: Tendencias de gramática
        if metrics_dashboard.trends.get("grammar"):
            widgets.append(DashboardWidget(
                widget_id="grammar_trend",
                widget_type="chart",
                title="Tendencia de Gramática",
                data=metrics_dashboard.trends["grammar"]["daily"],
                config={"type": "line", "y_label": "Score", "x_label": "Fecha"}
            ))
        
        # Widget: Top categorías
        if metrics_dashboard.top_categories:
            widgets.append(DashboardWidget(
                widget_id="top_categories",
                widget_type="chart",
                title="Top Categorías",
                data=metrics_dashboard.top_categories,
                config={"type": "pie", "label_field": "category", "value_field": "count"}
            ))
        
        # Widget: Métricas adicionales
        if metrics_dashboard.metrics:
            widgets.append(DashboardWidget(
                widget_id="additional_metrics",
                widget_type="table",
                title="Métricas Adicionales",
                data=[
                    {
                        "name": m.metric_name,
                        "value": m.value,
                        "unit": m.unit
                    }
                    for m in metrics_dashboard.metrics
                ]
            ))
        
        # Filtrar widgets si se especifica
        if include_widgets:
            widgets = [w for w in widgets if w.widget_id in include_widgets]
        
        return DashboardData(
            dashboard_id=f"dashboard_{period}_{days}",
            title=f"Dashboard {period.capitalize()} - {days} días",
            widgets=widgets,
            period=period,
            metadata={
                "total_widgets": len(widgets),
                "data_points": metrics_dashboard.total_analyses
            }
        )
    
    def generate_html_dashboard(self, dashboard_data: DashboardData) -> str:
        """Generar HTML del dashboard."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{dashboard_data.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .widgets {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .widget {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .widget h3 {{
            margin-top: 0;
            color: #333;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            color: #333;
            font-size: 1.1em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f8f8;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>{dashboard_data.title}</h1>
            <p>Generado: {dashboard_data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="widgets">
"""
        
        for widget in dashboard_data.widgets:
            html += f"""
            <div class="widget">
                <h3>{widget.title}</h3>
"""
            
            if widget.widget_type == "metric":
                html += '<div class="metrics">'
                for key, value in widget.data.items():
                    html += f"""
                    <div class="metric">
                        <span class="metric-label">{key.replace('_', ' ').title()}:</span>
                        <span class="metric-value">{value}</span>
                    </div>
"""
                html += '</div>'
            
            elif widget.widget_type == "table":
                html += '<table>'
                if widget.data:
                    # Headers
                    html += '<thead><tr>'
                    for key in widget.data[0].keys():
                        html += f'<th>{key.replace("_", " ").title()}</th>'
                    html += '</tr></thead>'
                    
                    # Rows
                    html += '<tbody>'
                    for row in widget.data:
                        html += '<tr>'
                        for value in row.values():
                            html += f'<td>{value}</td>'
                        html += '</tr>'
                    html += '</tbody>'
                html += '</table>'
            
            elif widget.widget_type == "chart":
                chart_id = f"chart_{widget.widget_id}"
                html += f'<canvas id="{chart_id}"></canvas>'
                html += f"""
                <script>
                    const ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
                    new Chart(ctx_{chart_id}, {{
                        type: '{widget.config.get("type", "line")}',
                        data: {{
                            labels: {json.dumps(list(widget.data.keys()))},
                            datasets: [{{
                                label: '{widget.title}',
                                data: {json.dumps(list(widget.data.values()))},
                                borderColor: 'rgb(75, 192, 192)',
                                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                tension: 0.1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                legend: {{
                                    display: true
                                }}
                            }},
                            scales: {{
                                y: {{
                                    beginAtZero: true
                                }}
                            }}
                        }}
                    }});
                </script>
"""
            
            html += """
            </div>
"""
        
        html += """
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    async def save_dashboard_html(
        self,
        dashboard_data: DashboardData,
        output_path: str
    ) -> str:
        """Guardar dashboard como HTML."""
        html = self.generate_html_dashboard(dashboard_data)
        output_file = output_path if output_path.endswith('.html') else f"{output_path}.html"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_file


__all__ = [
    "DashboardGenerator",
    "DashboardData",
    "DashboardWidget"
]
















