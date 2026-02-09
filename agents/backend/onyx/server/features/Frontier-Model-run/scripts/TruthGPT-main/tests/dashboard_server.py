#!/usr/bin/env python3
"""
Servidor de Dashboard
Servidor web para visualizar resultados de tests en tiempo real
"""

import sys
import json
from pathlib import Path
from typing import Dict
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time


class DashboardHandler(BaseHTTPRequestHandler):
    """Handler para el servidor de dashboard"""
    
    data_path = None
    last_update = None
    
    def do_GET(self):
        """Manejar requests GET"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self._generate_dashboard().encode())
        elif self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data = self._load_data()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def _load_data(self) -> Dict:
        """Cargar datos de resultados"""
        if self.data_path and self.data_path.exists():
            with open(self.data_path, 'r') as f:
                return json.load(f)
        return {'stats': {}, 'history': []}
    
    def _generate_dashboard(self) -> str:
        """Generar HTML del dashboard"""
        data = self._load_data()
        stats = data.get('stats', {})
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>TruthGPT Test Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
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
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{ color: #333; margin-bottom: 20px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .auto-refresh {{
            background: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .auto-refresh.active {{ background: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 TruthGPT Test Dashboard</h1>
        <div class="auto-refresh" id="refreshStatus">Auto-refresh: ON (cada 5s)</div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div>Total Runs</div>
                <div class="value" id="totalRuns">{stats.get('total_runs', 0)}</div>
            </div>
            <div class="stat-card">
                <div>Successful</div>
                <div class="value" id="successful">{stats.get('successful', 0)}</div>
            </div>
            <div class="stat-card">
                <div>Failed</div>
                <div class="value" id="failed">{stats.get('failed', 0)}</div>
            </div>
            <div class="stat-card">
                <div>Success Rate</div>
                <div class="value" id="successRate">
                    {(stats.get('successful', 0) / stats.get('total_runs', 1) * 100):.1f}%
                </div>
            </div>
        </div>
        
        <div style="margin-top: 30px;">
            <canvas id="trendChart"></canvas>
        </div>
    </div>
    
    <script>
        let chart = null;
        
        function loadData() {{
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {{
                    updateDashboard(data);
                }})
                .catch(error => console.error('Error:', error));
        }}
        
        function updateDashboard(data) {{
            const stats = data.stats || {{}};
            const history = data.history || [];
            
            // Actualizar estadísticas
            document.getElementById('totalRuns').textContent = stats.total_runs || 0;
            document.getElementById('successful').textContent = stats.successful || 0;
            document.getElementById('failed').textContent = stats.failed || 0;
            
            const total = stats.total_runs || 1;
            const success = stats.successful || 0;
            document.getElementById('successRate').textContent = 
                ((success / total) * 100).toFixed(1) + '%';
            
            // Actualizar gráfico
            updateChart(history);
        }}
        
        function updateChart(history) {{
            const labels = history.slice(-20).map(h => 
                new Date(h.timestamp).toLocaleTimeString()
            );
            const times = history.slice(-20).map(h => h.elapsed || 0);
            const successes = history.slice(-20).map(h => h.success ? 1 : 0);
            
            if (!chart) {{
                const ctx = document.getElementById('trendChart').getContext('2d');
                chart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: labels,
                        datasets: [
                            {{
                                label: 'Execution Time (s)',
                                data: times,
                                borderColor: 'rgb(102, 126, 234)',
                                yAxisID: 'y'
                            }},
                            {{
                                label: 'Success (1) / Failure (0)',
                                data: successes,
                                borderColor: 'rgb(56, 239, 125)',
                                yAxisID: 'y1'
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{ beginAtZero: true }},
                            y1: {{ beginAtZero: true, max: 1, position: 'right' }}
                        }}
                    }}
                }});
            }} else {{
                chart.data.labels = labels;
                chart.data.datasets[0].data = times;
                chart.data.datasets[1].data = successes;
                chart.update();
            }}
        }}
        
        // Cargar datos iniciales
        loadData();
        
        // Auto-refresh cada 5 segundos
        setInterval(loadData, 5000);
    </script>
</body>
</html>
"""
    
    def log_message(self, format, *args):
        """Suprimir logs del servidor"""
        pass


def run_server(port: int = 8080, data_path: Path = None):
    """Ejecutar servidor de dashboard"""
    DashboardHandler.data_path = data_path
    
    server = HTTPServer(('', port), DashboardHandler)
    print(f"🚀 Dashboard servidor iniciado en http://localhost:{port}")
    print(f"   Presiona Ctrl+C para detener")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo servidor...")
        server.shutdown()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Servidor de dashboard de tests')
    parser.add_argument('--port', type=int, default=8080,
                       help='Puerto del servidor')
    parser.add_argument('--data', type=Path,
                       help='Archivo JSON de datos')
    
    args = parser.parse_args()
    
    run_server(args.port, args.data)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

