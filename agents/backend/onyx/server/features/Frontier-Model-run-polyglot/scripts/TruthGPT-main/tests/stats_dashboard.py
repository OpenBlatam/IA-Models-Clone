#!/usr/bin/env python3
"""
Dashboard de Estadísticas
Genera dashboard completo de estadísticas de tests
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict


class StatsDashboard:
    """Generador de dashboard de estadísticas"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    def collect_all_stats(self) -> Dict:
        """Recopilar todas las estadísticas"""
        print("📊 Recopilando estadísticas completas...")
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'file_stats': self._get_file_stats(),
            'test_stats': self._get_test_stats(),
            'category_stats': self._get_category_stats(),
            'coverage_stats': self._get_coverage_stats(),
            'performance_stats': self._get_performance_stats()
        }
        
        return stats
    
    def _get_file_stats(self) -> Dict:
        """Estadísticas de archivos"""
        total_files = len(list(self.base_path.rglob('*.py')))
        test_files = len(list(self.base_path.rglob('test_*.py')))
        analyzer_files = len(list((self.base_path / 'analyzers').rglob('*.py')))
        
        return {
            'total_python_files': total_files,
            'test_files': test_files,
            'analyzer_files': analyzer_files,
            'other_files': total_files - test_files - analyzer_files
        }
    
    def _get_test_stats(self) -> Dict:
        """Estadísticas de tests"""
        unit_tests = len(list((self.base_path / 'core' / 'unit').rglob('test_*.py')))
        integration_tests = len(list((self.base_path / 'core' / 'integration').rglob('test_*.py')))
        
        return {
            'unit_tests': unit_tests,
            'integration_tests': integration_tests,
            'total_tests': unit_tests + integration_tests
        }
    
    def _get_category_stats(self) -> Dict:
        """Estadísticas por categoría"""
        categories = {}
        
        for category_dir in (self.base_path / 'analyzers').iterdir():
            if category_dir.is_dir() and category_dir.name != '__pycache__':
                files = len(list(category_dir.rglob('*.py')))
                if files > 0:
                    categories[category_dir.name] = files
        
        return categories
    
    def _get_coverage_stats(self) -> Dict:
        """Estadísticas de cobertura (simulado)"""
        return {
            'estimated_coverage': 85.0,
            'target_coverage': 90.0,
            'status': 'good'
        }
    
    def _get_performance_stats(self) -> Dict:
        """Estadísticas de rendimiento (simulado)"""
        return {
            'avg_execution_time': 45.2,
            'fastest_category': 'unit',
            'slowest_category': 'integration',
            'total_executions': 150
        }
    
    def generate_html_dashboard(self, stats: Dict) -> str:
        """Generar dashboard HTML"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Statistics Dashboard</title>
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
        h1 {{ color: #333; margin-bottom: 30px; }}
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
        }}
        .stat-card .value {{
            font-size: 36px;
            font-weight: bold;
        }}
        .chart-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Test Statistics Dashboard</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Python Files</h3>
                <div class="value">{stats['file_stats']['total_python_files']}</div>
            </div>
            <div class="stat-card">
                <h3>Test Files</h3>
                <div class="value">{stats['test_stats']['total_tests']}</div>
            </div>
            <div class="stat-card">
                <h3>Analyzer Files</h3>
                <div class="value">{stats['file_stats']['analyzer_files']}</div>
            </div>
            <div class="stat-card">
                <h3>Categories</h3>
                <div class="value">{len(stats['category_stats'])}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>📁 Files by Category</h2>
            <canvas id="categoryChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>🧪 Test Distribution</h2>
            <canvas id="testChart"></canvas>
        </div>
    </div>
    
    <script>
        // Category Chart
        const catCtx = document.getElementById('categoryChart').getContext('2d');
        const categoryData = {{
            labels: {json.dumps(list(stats['category_stats'].keys()))},
            datasets: [{{
                label: 'Files',
                data: {json.dumps(list(stats['category_stats'].values()))},
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: 'rgb(102, 126, 234)',
                borderWidth: 2
            }}]
        }};
        
        new Chart(catCtx, {{
            type: 'bar',
            data: categoryData,
            options: {{
                responsive: true,
                scales: {{
                    y: {{ beginAtZero: true }}
                }}
            }}
        }});
        
        // Test Chart
        const testCtx = document.getElementById('testChart').getContext('2d');
        const testData = {{
            labels: ['Unit Tests', 'Integration Tests'],
            datasets: [{{
                label: 'Tests',
                data: [{stats['test_stats']['unit_tests']}, {stats['test_stats']['integration_tests']}],
                backgroundColor: ['rgba(56, 239, 125, 0.6)', 'rgba(102, 126, 234, 0.6)'],
                borderColor: ['rgb(56, 239, 125)', 'rgb(102, 126, 234)'],
                borderWidth: 2
            }}]
        }};
        
        new Chart(testCtx, {{
            type: 'pie',
            data: testData,
            options: {{
                responsive: true
            }}
        }});
    </script>
</body>
</html>
"""
        return html
    
    def save_dashboard(self, stats: Dict, output_path: Path):
        """Guardar dashboard"""
        html = self.generate_html_dashboard(stats)
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"✅ Dashboard guardado en: {output_path}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar dashboard de estadísticas')
    parser.add_argument('--output', type=Path, default=Path('stats_dashboard.html'),
                       help='Archivo HTML de salida')
    parser.add_argument('--json', type=Path,
                       help='Archivo JSON de salida')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    dashboard = StatsDashboard(args.base_path)
    stats = dashboard.collect_all_stats()
    
    dashboard.save_dashboard(stats, args.output)
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Estadísticas JSON guardadas en: {args.json}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

