#!/usr/bin/env python3
"""
Visualizador de Resultados
Genera visualizaciones de resultados de tests
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import argparse


class ResultsVisualizer:
    """Visualizador de resultados de tests"""
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Cargar datos de resultados"""
        if not self.data_path.exists():
            return {}
        
        with open(self.data_path, 'r') as f:
            return json.load(f)
    
    def generate_html_dashboard(self) -> str:
        """Generar dashboard HTML"""
        stats = self.data.get('stats', {})
        history = stats.get('history', [])
        
        # Calcular métricas
        total = len(history)
        successful = sum(1 for r in history if r.get('success', False))
        failed = total - successful
        success_rate = (successful / total * 100) if total > 0 else 0
        
        # Tiempos
        times = [r.get('elapsed', 0) for r in history]
        avg_time = sum(times) / len(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Results Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-card .value {{
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .stat-card.danger {{
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }}
        .stat-card.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .chart-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .chart-container h2 {{
            color: #333;
            margin-bottom: 20px;
        }}
        canvas {{
            max-height: 400px;
        }}
        .timeline {{
            margin-top: 30px;
        }}
        .timeline-item {{
            display: flex;
            align-items: center;
            padding: 15px;
            margin-bottom: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .timeline-item.success {{
            border-left-color: #38ef7d;
        }}
        .timeline-item.failed {{
            border-left-color: #f45c43;
        }}
        .timeline-item .icon {{
            font-size: 24px;
            margin-right: 15px;
        }}
        .timeline-item .info {{
            flex: 1;
        }}
        .timeline-item .time {{
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Test Results Dashboard</h1>
        <p class="subtitle">Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Runs</h3>
                <div class="value">{total}</div>
            </div>
            <div class="stat-card success">
                <h3>Successful</h3>
                <div class="value">{successful}</div>
            </div>
            <div class="stat-card danger">
                <h3>Failed</h3>
                <div class="value">{failed}</div>
            </div>
            <div class="stat-card warning">
                <h3>Success Rate</h3>
                <div class="value">{success_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Avg Time</h3>
                <div class="value">{avg_time:.2f}s</div>
            </div>
            <div class="stat-card">
                <h3>Min/Max Time</h3>
                <div class="value">{min_time:.2f}s / {max_time:.2f}s</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>📈 Execution Time Trend</h2>
            <canvas id="timeChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>📊 Success Rate Over Time</h2>
            <canvas id="successChart"></canvas>
        </div>
        
        <div class="timeline">
            <h2>🕐 Recent Runs</h2>
"""
        
        # Agregar timeline de ejecuciones recientes
        for run in history[-20:]:  # Últimas 20
            status_class = 'success' if run.get('success') else 'failed'
            icon = '✅' if run.get('success') else '❌'
            timestamp = run.get('timestamp', '')
            elapsed = run.get('elapsed', 0)
            
            html += f"""
            <div class="timeline-item {status_class}">
                <div class="icon">{icon}</div>
                <div class="info">
                    <strong>Run at {timestamp[:19]}</strong>
                    <div class="time">Elapsed: {elapsed:.2f}s</div>
                </div>
            </div>
"""
        
        html += """
        </div>
    </div>
    
    <script>
        // Time Chart
        const timeCtx = document.getElementById('timeChart').getContext('2d');
        const timeData = {
            labels: """ + json.dumps([r.get('timestamp', '')[:19] for r in history[-30:]]) + """,
            datasets: [{
                label: 'Execution Time (s)',
                data: """ + json.dumps([r.get('elapsed', 0) for r in history[-30:]]) + """,
                borderColor: 'rgb(102, 126, 234)',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4
            }]
        };
        
        new Chart(timeCtx, {
            type: 'line',
            data: timeData,
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        // Success Chart
        const successCtx = document.getElementById('successChart').getContext('2d');
        const successData = {
            labels: """ + json.dumps([r.get('timestamp', '')[:19] for r in history[-30:]]) + """,
            datasets: [{
                label: 'Success (1) / Failure (0)',
                data: """ + json.dumps([1 if r.get('success') else 0 for r in history[-30:]]) + """,
                borderColor: 'rgb(56, 239, 125)',
                backgroundColor: 'rgba(56, 239, 125, 0.1)',
                tension: 0.4
            }]
        };
        
        new Chart(successCtx, {
            type: 'line',
            data: successData,
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    </script>
</body>
</html>
"""
        return html


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Visualizar resultados de tests')
    parser.add_argument('--input', type=Path, required=True,
                       help='Archivo JSON de entrada')
    parser.add_argument('--output', type=Path, default=Path('dashboard.html'),
                       help='Archivo HTML de salida')
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.input)
    html = visualizer.generate_html_dashboard()
    
    with open(args.output, 'w') as f:
        f.write(html)
    
    print(f"✅ Dashboard generado: {args.output}")
    print(f"   Abre en tu navegador para ver la visualización")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

