"""
Generador de Dashboard HTML para resultados de pruebas
Crea un dashboard interactivo y visual con los resultados
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

def generate_html_dashboard(results_file: str = "test_results.json", output_file: str = "test_dashboard.html"):
    """Genera un dashboard HTML interactivo."""
    
    # Leer resultados
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {results_file}")
        return None
    
    summary = results.get("summary", {})
    tests = results.get("tests", [])
    errors = results.get("errors", [])
    metrics = results.get("metrics", {})
    
    # Calcular estadísticas
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    success_rate = summary.get("success_rate", 0)
    duration = summary.get("duration", 0)
    
    # Generar HTML
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Pruebas - API BUL</title>
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
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-card.success {{
            border-left: 5px solid #28a745;
        }}
        
        .stat-card.error {{
            border-left: 5px solid #dc3545;
        }}
        
        .stat-card.info {{
            border-left: 5px solid #17a2b8;
        }}
        
        .stat-value {{
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .info {{ color: #17a2b8; }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .test-list {{
            display: grid;
            gap: 15px;
        }}
        
        .test-item {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s;
        }}
        
        .test-item:hover {{
            background: #e9ecef;
            transform: translateX(5px);
        }}
        
        .test-item.passed {{
            border-left-color: #28a745;
        }}
        
        .test-item.failed {{
            border-left-color: #dc3545;
        }}
        
        .test-name {{
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }}
        
        .test-duration {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .badge-success {{
            background: #28a745;
            color: white;
        }}
        
        .badge-error {{
            background: #dc3545;
            color: white;
        }}
        
        .error-list {{
            background: #fff5f5;
            border: 2px solid #fed7d7;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .error-item {{
            padding: 15px;
            margin-bottom: 10px;
            background: white;
            border-left: 4px solid #dc3545;
            border-radius: 4px;
        }}
        
        .error-test {{
            font-weight: bold;
            color: #dc3545;
            margin-bottom: 5px;
        }}
        
        .error-message {{
            color: #666;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        
        .metric-title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            color: #667eea;
            font-weight: bold;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        
        .timestamp {{
            text-align: center;
            color: #666;
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .test-item {{
                flex-direction: column;
                align-items: flex-start;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Dashboard de Pruebas</h1>
            <p>API BUL - Frontend Ready</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card success">
                <div class="stat-label">Total de Pruebas</div>
                <div class="stat-value info">{total}</div>
            </div>
            
            <div class="stat-card success">
                <div class="stat-label">Exitosas</div>
                <div class="stat-value success">{passed}</div>
            </div>
            
            <div class="stat-card error">
                <div class="stat-label">Fallidas</div>
                <div class="stat-value error">{failed}</div>
            </div>
            
            <div class="stat-card info">
                <div class="stat-label">Tasa de Éxito</div>
                <div class="stat-value info">{success_rate:.1f}%</div>
            </div>
            
            <div class="stat-card info">
                <div class="stat-label">Duración</div>
                <div class="stat-value info">{duration:.2f}s</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {success_rate}%">
                {success_rate:.1f}%
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📋 Lista de Pruebas</h2>
                <div class="test-list">
"""
    
    # Agregar cada prueba
    for test in tests:
        status_class = "passed" if test.get("passed") else "failed"
        badge_class = "badge-success" if test.get("passed") else "badge-error"
        badge_text = "PASS" if test.get("passed") else "FAIL"
        
        html += f"""
                    <div class="test-item {status_class}">
                        <div>
                            <div class="test-name">{test.get("name", "Unknown")}</div>
                            <div class="test-duration">Duración: {test.get("duration", 0):.2f}s</div>
                        </div>
                        <span class="badge {badge_class}">{badge_text}</span>
                    </div>
        """
    
    html += """
                </div>
            </div>
"""
    
    # Agregar errores si hay
    if errors:
        html += """
            <div class="section">
                <h2>❌ Errores Encontrados</h2>
                <div class="error-list">
"""
        for error in errors:
            html += f"""
                    <div class="error-item">
                        <div class="error-test">{error.get("test", "Unknown")}</div>
                        <div class="error-message">{error.get("error", "Unknown error")}</div>
                    </div>
            """
        
        html += """
                </div>
            </div>
"""
    
    # Agregar métricas si hay
    if metrics:
        html += """
            <div class="section">
                <h2>📊 Métricas Detalladas</h2>
                <div class="metrics-grid">
"""
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    html += f"""
                    <div class="metric-card">
                        <div class="metric-title">{key} - {sub_key}</div>
                        <div class="metric-value">{sub_value}</div>
                    </div>
                    """
            else:
                html += f"""
                    <div class="metric-card">
                        <div class="metric-title">{key}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                """
        
        html += """
                </div>
            </div>
"""
    
    # Footer
    timestamp = summary.get("timestamp", datetime.now().isoformat())
    html += f"""
        </div>
        
        <div class="timestamp">
            Generado el: {timestamp}
        </div>
    </div>
</body>
</html>
"""
    
    # Guardar archivo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Dashboard generado: {output_file}")
    return output_file

if __name__ == "__main__":
    import sys
    
    results_file = sys.argv[1] if len(sys.argv) > 1 else "test_results.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "test_dashboard.html"
    
    generate_html_dashboard(results_file, output_file)
































