#!/usr/bin/env python3
"""
Script para analizar métricas de validación.
Recopila datos y genera análisis.
"""

import json
from pathlib import Path
from datetime import datetime

def load_metrics():
    """Carga métricas desde archivo JSON."""
    metrics_file = Path(__file__).parent.parent / "results" / "metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_conversion_rate(visits, uploads):
    """Calcula tasa de conversión."""
    if visits == 0:
        return 0
    return (uploads / visits) * 100

def analyze_metrics():
    """Analiza métricas y genera reporte."""
    metrics = load_metrics()
    
    if not metrics:
        print("❌ No hay métricas para analizar.")
        print("💡 Ejecuta 'collect_metrics.py' primero.")
        return
    
    print("\n📊 REPORTE DE MÉTRICAS DE VALIDACIÓN\n")
    print("=" * 50)
    
    # Métricas de tráfico
    visits = metrics.get('visits', 0)
    unique_users = metrics.get('unique_users', 0)
    bounce_rate = metrics.get('bounce_rate', 0)
    
    print(f"\n🌐 TRÁFICO:")
    print(f"  Visitas: {visits}")
    print(f"  Usuarios únicos: {unique_users}")
    print(f"  Tasa de rebote: {bounce_rate:.1f}%")
    
    # Métricas de conversión
    uploads = metrics.get('uploads', 0)
    completions = metrics.get('completions', 0)
    conversion_rate = calculate_conversion_rate(visits, uploads)
    completion_rate = calculate_conversion_rate(uploads, completions) if uploads > 0 else 0
    
    print(f"\n📈 CONVERSIÓN:")
    print(f"  Subidas de foto: {uploads}")
    print(f"  Análisis completados: {completions}")
    print(f"  Tasa de conversión (visitas → uploads): {conversion_rate:.1f}%")
    print(f"  Tasa de completación (uploads → completions): {completion_rate:.1f}%")
    
    # Métricas de satisfacción
    avg_feedback = metrics.get('avg_feedback', 0)
    nps = metrics.get('nps', 0)
    payment_intent = metrics.get('payment_intent', 0)
    
    print(f"\n😊 SATISFACCIÓN:")
    print(f"  Feedback promedio: {avg_feedback:.1f}/5")
    print(f"  NPS: {nps}")
    print(f"  % que pagarían: {payment_intent:.1f}%")
    
    # Métricas técnicas
    avg_analysis_time = metrics.get('avg_analysis_time', 0)
    error_rate = metrics.get('error_rate', 0)
    
    print(f"\n⚙️ TÉCNICAS:")
    print(f"  Tiempo promedio de análisis: {avg_analysis_time:.1f}s")
    print(f"  Tasa de error: {error_rate:.1f}%")
    
    # Calcular health score
    satisfaction_score = (avg_feedback / 5) * 100 if avg_feedback > 0 else 0
    health_score = (
        (conversion_rate * 0.3) + 
        (satisfaction_score * 0.3) + 
        (50 * 0.2) +  # Placeholder for retention
        (payment_intent * 0.2)
    )
    
    print(f"\n🎯 SCORE DE SALUD:")
    print(f"  Score: {health_score:.0f}/100")
    if health_score >= 70:
        print("  Estado: ✅ Excelente - Continúa!")
    elif health_score >= 50:
        print("  Estado: ⚠️ Bueno - Hay espacio para mejorar")
    else:
        print("  Estado: ❌ Necesita mejora - Revisa estrategia")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    recommendations = []
    
    if conversion_rate < 20:
        recommendations.append("  ⚠️ Tasa de conversión baja. Revisa UX y propuesta de valor.")
    
    if completion_rate < 50:
        recommendations.append("  ⚠️ Muchos usuarios no completan. Revisa tiempo de análisis.")
    
    if avg_feedback < 3.5:
        recommendations.append("  ⚠️ Feedback bajo. Revisa calidad de resultados.")
    
    if payment_intent < 30:
        recommendations.append("  ⚠️ Baja intención de pago. Revisa propuesta de valor.")
    
    if avg_analysis_time > 15:
        recommendations.append("  ⚠️ Análisis muy lento. Optimiza backend.")
    
    if error_rate > 5:
        recommendations.append("  ⚠️ Alta tasa de errores. Revisa estabilidad.")
    
    if not recommendations:
        recommendations.append("  ✅ Métricas en buen rango. Continúa iterando!")
    
    for rec in recommendations:
        print(rec)
    
    # Decisión
    print(f"\n🎯 DECISIÓN:")
    
    if (conversion_rate >= 20 and 
        completion_rate >= 50 and 
        avg_feedback >= 3.5 and 
        payment_intent >= 30):
        print("  ✅ CONTINUAR - Métricas positivas")
    elif (conversion_rate >= 10 and 
          avg_feedback >= 3.0):
        print("  ⚠️ ITERAR - Hay potencial pero necesita mejoras")
    else:
        print("  ❌ RECONSIDERAR - Métricas bajas, necesita pivotar")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    analyze_metrics()






