"""
Generador de Dashboard Web
===========================

Sistema para generar dashboards HTML simples.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
    Generador de dashboards web
    
    Genera dashboards HTML interactivos para visualización
    de métricas y estadísticas.
    """
    
    @staticmethod
    def generate_dashboard(
        metrics: Dict[str, Any],
        title: str = "Analizador de Documentos - Dashboard"
    ) -> str:
        """
        Generar dashboard HTML
        
        Args:
            metrics: Métricas a mostrar
            title: Título del dashboard
        
        Returns:
            HTML del dashboard
        """
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .chart-container h2 {{
            margin-bottom: 20px;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats-grid">
            {DashboardGenerator._generate_stat_cards(metrics)}
        </div>
        
        <div class="chart-container">
            <h2>Métricas de Rendimiento</h2>
            <canvas id="performanceChart"></canvas>
        </div>
    </div>
    
    <script>
        {DashboardGenerator._generate_chart_js(metrics)}
    </script>
</body>
</html>"""
        
        return html
    
    @staticmethod
    def _generate_stat_cards(metrics: Dict[str, Any]) -> str:
        """Generar tarjetas de estadísticas"""
        cards = []
        
        performance = metrics.get("performance", {})
        if performance:
            cards.append(f"""
                <div class="stat-card">
                    <h3>Total de Peticiones</h3>
                    <div class="value">{performance.get('total_requests', 0)}</div>
                </div>
            """)
            
            cards.append(f"""
                <div class="stat-card">
                    <h3>Tasa de Éxito</h3>
                    <div class="value">{(performance.get('success_rate', 0) * 100):.1f}%</div>
                </div>
            """)
            
            cards.append(f"""
                <div class="stat-card">
                    <h3>Tiempo Promedio</h3>
                    <div class="value">{performance.get('avg_duration', 0):.2f}s</div>
                </div>
            """)
        
        return "".join(cards)
    
    @staticmethod
    def _generate_chart_js(metrics: Dict[str, Any]) -> str:
        """Generar código JavaScript para gráficos"""
        performance = metrics.get("performance", {})
        
        return f"""
        const ctx = document.getElementById('performanceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: ['P50', 'P95', 'P99'],
                datasets: [{{
                    label: 'Tiempo de Respuesta (ms)',
                    data: [
                        {performance.get('p50', 0) * 1000:.0f},
                        {performance.get('p95', 0) * 1000:.0f},
                        {performance.get('p99', 0) * 1000:.0f}
                    ],
                    borderColor: 'rgb(102, 126, 234)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        """
















