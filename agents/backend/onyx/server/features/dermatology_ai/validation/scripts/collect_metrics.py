#!/usr/bin/env python3
"""
Script interactivo para recopilar métricas manualmente.
Útil cuando no tienes analytics configurado.
"""

import json
from pathlib import Path
from datetime import datetime

def collect_metrics():
    """Recopila métricas de forma interactiva."""
    print("\n📊 RECOPILACIÓN DE MÉTRICAS\n")
    print("=" * 50)
    
    metrics = {}
    
    # Métricas de tráfico
    print("\n🌐 TRÁFICO:")
    try:
        metrics['visits'] = int(input("  Visitas totales: ") or "0")
        metrics['unique_users'] = int(input("  Usuarios únicos: ") or "0")
        metrics['bounce_rate'] = float(input("  Tasa de rebote (%): ") or "0")
    except ValueError:
        print("  ⚠️ Usando valores por defecto: 0")
        metrics['visits'] = 0
        metrics['unique_users'] = 0
        metrics['bounce_rate'] = 0
    
    # Métricas de conversión
    print("\n📈 CONVERSIÓN:")
    try:
        metrics['uploads'] = int(input("  Subidas de foto: ") or "0")
        metrics['completions'] = int(input("  Análisis completados: ") or "0")
    except ValueError:
        print("  ⚠️ Usando valores por defecto: 0")
        metrics['uploads'] = 0
        metrics['completions'] = 0
    
    # Métricas de satisfacción
    print("\n😊 SATISFACCIÓN:")
    try:
        metrics['avg_feedback'] = float(input("  Feedback promedio (1-5): ") or "0")
        metrics['nps'] = int(input("  NPS (-100 a 100): ") or "0")
        metrics['payment_intent'] = float(input("  % que pagarían: ") or "0")
    except ValueError:
        print("  ⚠️ Usando valores por defecto: 0")
        metrics['avg_feedback'] = 0
        metrics['nps'] = 0
        metrics['payment_intent'] = 0
    
    # Métricas técnicas
    print("\n⚙️ TÉCNICAS:")
    try:
        metrics['avg_analysis_time'] = float(input("  Tiempo promedio de análisis (segundos): ") or "0")
        metrics['error_rate'] = float(input("  Tasa de error (%): ") or "0")
    except ValueError:
        print("  ⚠️ Usando valores por defecto: 0")
        metrics['avg_analysis_time'] = 0
        metrics['error_rate'] = 0
    
    # Guardar
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    metrics_file = results_dir / "metrics.json"
    metrics['last_updated'] = datetime.now().isoformat()
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Métricas guardadas en: {metrics_file}")
    print("💡 Ejecuta 'analyze_metrics.py' para ver el análisis.")

if __name__ == "__main__":
    collect_metrics()






