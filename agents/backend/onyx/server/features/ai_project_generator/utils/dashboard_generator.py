"""
Dashboard Generator - Generador de Dashboard Web
================================================

Genera un dashboard web para visualizar y gestionar proyectos.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import json

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generador de dashboard web"""

    def __init__(self):
        """Inicializa el generador de dashboard"""
        pass

    async def generate_dashboard(
        self,
        output_dir: Path,
        api_url: str = "http://localhost:8020",
    ):
        """
        Genera un dashboard web completo.

        Args:
            output_dir: Directorio donde generar el dashboard
            api_url: URL de la API
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # index.html
        index_html = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Project Generator - Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-100">
    <div id="app">
        <nav class="bg-blue-600 text-white p-4">
            <div class="container mx-auto">
                <h1 class="text-2xl font-bold">AI Project Generator Dashboard</h1>
            </div>
        </nav>

        <div class="container mx-auto p-6">
            <!-- Stats Cards -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div class="bg-white p-6 rounded-lg shadow">
                    <h3 class="text-gray-500 text-sm">Total Proyectos</h3>
                    <p class="text-3xl font-bold" id="total-projects">-</p>
                </div>
                <div class="bg-white p-6 rounded-lg shadow">
                    <h3 class="text-gray-500 text-sm">En Cola</h3>
                    <p class="text-3xl font-bold" id="queue-size">-</p>
                </div>
                <div class="bg-white p-6 rounded-lg shadow">
                    <h3 class="text-gray-500 text-sm">Tasa de Éxito</h3>
                    <p class="text-3xl font-bold" id="success-rate">-</p>
                </div>
                <div class="bg-white p-6 rounded-lg shadow">
                    <h3 class="text-gray-500 text-sm">Cache Hits</h3>
                    <p class="text-3xl font-bold" id="cache-hits">-</p>
                </div>
            </div>

            <!-- Charts -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div class="bg-white p-6 rounded-lg shadow">
                    <h2 class="text-xl font-bold mb-4">Proyectos por Tipo de IA</h2>
                    <canvas id="aiTypeChart"></canvas>
                </div>
                <div class="bg-white p-6 rounded-lg shadow">
                    <h2 class="text-xl font-bold mb-4">Proyectos por Día</h2>
                    <canvas id="dailyChart"></canvas>
                </div>
            </div>

            <!-- Recent Projects -->
            <div class="bg-white p-6 rounded-lg shadow">
                <h2 class="text-xl font-bold mb-4">Proyectos Recientes</h2>
                <div id="recent-projects" class="space-y-2">
                    <p class="text-gray-500">Cargando...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_URL = '{api_url}';

        async function loadStats() {{
            try {{
                const [stats, queue, searchStats] = await Promise.all([
                    fetch(API_URL + '/api/v1/stats').then(r => r.json()),
                    fetch(API_URL + '/api/v1/queue').then(r => r.json()),
                    fetch(API_URL + '/api/v1/search/stats').then(r => r.json()),
                ]);

                document.getElementById('total-projects').textContent = stats.total_processed || 0;
                document.getElementById('queue-size').textContent = queue.queue_size || 0;
                document.getElementById('success-rate').textContent = (stats.success_rate || 0) + '%';
                document.getElementById('cache-hits').textContent = searchStats.by_ai_type?.chat || 0;
            }} catch (error) {{
                console.error('Error cargando stats:', error);
            }}
        }}

        async function loadRecentProjects() {{
            try {{
                const response = await fetch(API_URL + '/api/v1/projects?limit=10');
                const data = await response.json();
                
                const container = document.getElementById('recent-projects');
                if (data.projects && data.projects.length > 0) {{
                    container.innerHTML = data.projects.map(p => `
                        <div class="border p-4 rounded">
                            <h3 class="font-bold">${{p.id}}</h3>
                            <p class="text-sm text-gray-600">${{p.description}}</p>
                            <span class="text-xs text-blue-600">${{p.status}}</span>
                        </div>
                    `).join('');
                }} else {{
                    container.innerHTML = '<p class="text-gray-500">No hay proyectos</p>';
                }}
            }} catch (error) {{
                console.error('Error cargando proyectos:', error);
            }}
        }}

        // Cargar datos al iniciar
        loadStats();
        loadRecentProjects();
        setInterval(loadStats, 30000); // Actualizar cada 30 segundos
        setInterval(loadRecentProjects, 60000); // Actualizar cada minuto
    </script>
</body>
</html>
'''
        (output_dir / "index.html").write_text(index_html, encoding="utf-8")

        logger.info(f"Dashboard generado en: {output_dir}")


