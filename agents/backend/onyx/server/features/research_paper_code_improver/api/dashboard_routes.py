"""
Dashboard Routes - Rutas para dashboard web
===========================================
"""

import logging
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    """Dashboard principal"""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Research Paper Code Improver - Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            color: #333;
            margin-bottom: 10px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .stat-card .value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .actions {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .btn {
            display: inline-block;
            padding: 12px 24px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 5px;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #5568d3;
        }
        .api-link {
            color: #667eea;
            text-decoration: none;
        }
        .api-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Research Paper Code Improver</h1>
            <p>Mejora tu código usando conocimiento de papers de investigación</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Papers Indexados</h3>
                <div class="value" id="papers-count">-</div>
            </div>
            <div class="stat-card">
                <h3>Mejoras Aplicadas</h3>
                <div class="value" id="improvements-count">-</div>
            </div>
            <div class="stat-card">
                <h3>Modelos Entrenados</h3>
                <div class="value" id="models-count">-</div>
            </div>
            <div class="stat-card">
                <h3>Estado del Sistema</h3>
                <div class="value" style="font-size: 18px;" id="system-status">-</div>
            </div>
        </div>
        
        <div class="actions">
            <h2>Acciones Rápidas</h2>
            <br>
            <a href="/docs" class="btn">📚 API Documentation</a>
            <a href="/api/research-paper-code-improver/health" class="btn">💚 Health Check</a>
            <a href="/api/research-paper-code-improver/papers" class="btn">📄 Ver Papers</a>
            <a href="/api/research-paper-code-improver/vector-store/stats" class="btn">🔍 Vector Store Stats</a>
            <a href="/api/research-paper-code-improver/cache/stats" class="btn">💾 Cache Stats</a>
        </div>
        
        <div class="actions" style="margin-top: 20px;">
            <h2>Gráficos y Métricas</h2>
            <canvas id="metricsChart" style="max-height: 300px; margin-top: 20px;"></canvas>
        </div>
        
        <div class="actions" style="margin-top: 20px;">
            <h2>Endpoints Principales</h2>
            <ul style="margin-top: 15px; line-height: 2;">
                <li><a href="/docs#/Research%20Paper%20Code%20Improver/upload_paper_papers_upload_post" class="api-link">POST /api/research-paper-code-improver/papers/upload</a> - Subir PDF</li>
                <li><a href="/docs#/Research%20Paper%20Code%20Improver/process_link_papers_link_post" class="api-link">POST /api/research-paper-code-improver/papers/link</a> - Procesar link</li>
                <li><a href="/docs#/Research%20Paper%20Code%20Improver/train_model_training_train_post" class="api-link">POST /api/research-paper-code-improver/training/train</a> - Entrenar modelo</li>
                <li><a href="/docs#/Research%20Paper%20Code%20Improver/improve_code_code_improve_post" class="api-link">POST /api/research-paper-code-improver/code/improve</a> - Mejorar código</li>
                <li><a href="/docs#/Research%20Paper%20Code%20Improver/batch_improve_batch_improve_post" class="api-link">POST /api/research-paper-code-improver/batch/improve</a> - Procesamiento en lote</li>
            </ul>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Cargar estadísticas
        async function loadStats() {
            try {
                const health = await fetch('/api/research-paper-code-improver/health');
                const healthData = await health.json();
                
                document.getElementById('papers-count').textContent = 
                    healthData.paper_storage?.total_papers || 0;
                document.getElementById('system-status').textContent = 
                    healthData.status === 'healthy' ? '✅ Healthy' : '⚠️ Issues';
                
                // Cargar métricas
                try {
                    const metrics = await fetch('/api/research-paper-code-improver/metrics/stats?hours=24');
                    const metricsData = await metrics.json();
                    
                    if (metricsData.improvements) {
                        document.getElementById('improvements-count').textContent = 
                            metricsData.improvements.total || 0;
                    }
                    
                    // Actualizar gráfico si existe
                    if (window.metricsChart) {
                        updateMetricsChart(metricsData);
                    }
                } catch (e) {
                    console.warn('No se pudieron cargar métricas:', e);
                }
            } catch (error) {
                console.error('Error cargando estadísticas:', error);
            }
        }
        
        // Crear gráfico de métricas
        function createMetricsChart() {
            const ctx = document.getElementById('metricsChart');
            if (!ctx) return;
            
            window.metricsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Mejoras Aplicadas',
                        data: [],
                        borderColor: 'rgb(102, 126, 234)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        function updateMetricsChart(metricsData) {
            if (!window.metricsChart) return;
            
            // Actualizar datos del gráfico
            // Esto es un ejemplo básico
            window.metricsChart.data.labels.push(new Date().toLocaleTimeString());
            window.metricsChart.data.datasets[0].data.push(
                metricsData.improvements?.total || 0
            );
            
            // Mantener solo últimos 20 puntos
            if (window.metricsChart.data.labels.length > 20) {
                window.metricsChart.data.labels.shift();
                window.metricsChart.data.datasets[0].data.shift();
            }
            
            window.metricsChart.update();
        }
        
        // Inicializar
        loadStats();
        createMetricsChart();
        setInterval(loadStats, 30000); // Actualizar cada 30 segundos
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)

