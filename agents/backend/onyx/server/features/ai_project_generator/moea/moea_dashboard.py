"""
MOEA Dashboard - Dashboard interactivo mejorado
==============================================
Dashboard web simple para monitoreo del sistema MOEA
"""
import http.server
import socketserver
import json
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs


class MOEADashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Handler para el dashboard MOEA"""
    
    def __init__(self, *args, api_url="http://localhost:8000", **kwargs):
        self.api_url = api_url
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Manejar requests GET"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.serve_dashboard()
        elif path == '/api/health':
            self.serve_api_health()
        elif path == '/api/stats':
            self.serve_api_stats()
        elif path == '/api/queue':
            self.serve_api_queue()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """Servir dashboard HTML"""
        html = self.get_dashboard_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_api_health(self):
        """Servir health check"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            data = {
                "status": "ok" if response.status_code == 200 else "error",
                "code": response.status_code,
                "timestamp": datetime.now().isoformat()
            }
        except:
            data = {
                "status": "error",
                "code": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        self.send_json_response(data)
    
    def serve_api_stats(self):
        """Servir estadísticas"""
        try:
            response = requests.get(f"{self.api_url}/api/v1/stats", timeout=5)
            if response.status_code == 200:
                data = response.json()
            else:
                data = {"error": "Unable to fetch stats"}
        except:
            data = {"error": "Server unavailable"}
        
        self.send_json_response(data)
    
    def serve_api_queue(self):
        """Servir estado de cola"""
        try:
            response = requests.get(f"{self.api_url}/api/v1/queue", timeout=5)
            if response.status_code == 200:
                data = response.json()
            else:
                data = {"error": "Unable to fetch queue"}
        except:
            data = {"error": "Server unavailable"}
        
        self.send_json_response(data)
    
    def send_json_response(self, data):
        """Enviar respuesta JSON"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def get_dashboard_html(self):
        """Generar HTML del dashboard"""
        return """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOEA Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .stat-card .value {
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .status.ok { background: #10b981; color: white; }
        .status.error { background: #ef4444; color: white; }
        .queue-list {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .queue-item {
            padding: 15px;
            border-bottom: 1px solid #eee;
        }
        .queue-item:last-child { border-bottom: none; }
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 20px;
        }
        .refresh-btn:hover { background: #5568d3; }
        .timestamp {
            color: #999;
            font-size: 12px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 MOEA System Dashboard</h1>
            <p>Monitoreo en tiempo real del sistema MOEA</p>
            <div class="timestamp" id="timestamp"></div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Estado del Servidor</h3>
                <div class="value" id="health-status">Verificando...</div>
            </div>
            <div class="stat-card">
                <h3>Proyectos Procesados</h3>
                <div class="value" id="processed-count">-</div>
            </div>
            <div class="stat-card">
                <h3>En Cola</h3>
                <div class="value" id="queue-size">-</div>
            </div>
            <div class="stat-card">
                <h3>Tiempo Promedio</h3>
                <div class="value" id="avg-time">-</div>
            </div>
        </div>
        
        <div class="queue-list">
            <h3 style="margin-bottom: 15px;">Cola de Proyectos</h3>
            <div id="queue-items">Cargando...</div>
        </div>
        
        <button class="refresh-btn" onclick="refreshData()">🔄 Actualizar</button>
    </div>
    
    <script>
        function updateTimestamp() {
            document.getElementById('timestamp').textContent = 
                'Última actualización: ' + new Date().toLocaleString('es-ES');
        }
        
        async function fetchHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                const statusEl = document.getElementById('health-status');
                statusEl.innerHTML = data.status === 'ok' 
                    ? '<span class="status ok">✅ OPERATIVO</span>'
                    : '<span class="status error">❌ NO DISPONIBLE</span>';
            } catch (e) {
                document.getElementById('health-status').innerHTML = 
                    '<span class="status error">❌ ERROR</span>';
            }
        }
        
        async function fetchStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                if (!data.error) {
                    document.getElementById('processed-count').textContent = 
                        data.processed_count || 0;
                    document.getElementById('queue-size').textContent = 
                        data.queue_size || 0;
                    document.getElementById('avg-time').textContent = 
                        (data.avg_time || 0).toFixed(2) + 's';
                }
            } catch (e) {
                console.error('Error fetching stats:', e);
            }
        }
        
        async function fetchQueue() {
            try {
                const response = await fetch('/api/queue');
                const data = await response.json();
                const queueEl = document.getElementById('queue-items');
                
                if (data.error) {
                    queueEl.innerHTML = '<p>Error cargando cola</p>';
                    return;
                }
                
                if (!data.queue || data.queue.length === 0) {
                    queueEl.innerHTML = '<p>No hay proyectos en cola</p>';
                    return;
                }
                
                queueEl.innerHTML = data.queue.slice(0, 10).map(item => `
                    <div class="queue-item">
                        <strong>${item.id || 'N/A'}</strong>
                        <span class="status ${item.status}">${item.status}</span>
                        <p style="margin-top: 5px; color: #666; font-size: 14px;">
                            ${(item.description || '').substring(0, 100)}...
                        </p>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('queue-items').innerHTML = 
                    '<p>Error cargando cola</p>';
            }
        }
        
        async function refreshData() {
            updateTimestamp();
            await Promise.all([fetchHealth(), fetchStats(), fetchQueue()]);
        }
        
        // Cargar datos iniciales
        refreshData();
        
        // Auto-refresh cada 5 segundos
        setInterval(refreshData, 5000);
    </script>
</body>
</html>"""


def create_handler_class(api_url):
    """Crear clase handler con API URL"""
    class Handler(MOEADashboardHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, api_url=api_url, **kwargs)
    return Handler


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Web Dashboard")
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Puerto del dashboard'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='URL de la API MOEA'
    )
    
    args = parser.parse_args()
    
    Handler = create_handler_class(args.api_url)
    
    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        print("=" * 70)
        print("MOEA Web Dashboard".center(70))
        print("=" * 70)
        print(f"\n✅ Dashboard disponible en: http://localhost:{args.port}")
        print(f"🔗 API MOEA: {args.api_url}")
        print(f"\n📊 Presiona Ctrl+C para detener\n")
        print("=" * 70)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n⚠️  Dashboard detenido")


if __name__ == "__main__":
    main()

