#!/usr/bin/env python3
"""
Genera un reporte completo de validación en Markdown.
"""

import json
from pathlib import Path
from datetime import datetime

def load_metrics():
    """Carga métricas."""
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

def generate_report():
    """Genera reporte de validación."""
    metrics = load_metrics()
    
    if not metrics:
        print("❌ No hay métricas. Ejecuta 'collect_metrics.py' primero.")
        return
    
    conversion_rate = calculate_conversion_rate(
        metrics.get('visits', 0), 
        metrics.get('uploads', 0)
    )
    completion_rate = calculate_conversion_rate(
        metrics.get('uploads', 0), 
        metrics.get('completions', 0)
    ) if metrics.get('uploads', 0) > 0 else 0
    
    satisfaction_score = (metrics.get('avg_feedback', 0) / 5) * 100 if metrics.get('avg_feedback', 0) > 0 else 0
    health_score = (
        (conversion_rate * 0.3) + 
        (satisfaction_score * 0.3) + 
        (50 * 0.2) +  # Placeholder for retention
        (metrics.get('payment_intent', 0) * 0.2)
    )
    
    report = f"""# 📊 Reporte de Validación - Dermatology AI

**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 Resumen Ejecutivo

Este reporte resume los resultados de la validación de la idea Dermatology AI.

**Score de Salud**: {health_score:.0f}/100

## 🌐 Métricas de Tráfico

- **Visitas**: {metrics.get('visits', 0)}
- **Usuarios únicos**: {metrics.get('unique_users', 0)}
- **Tasa de rebote**: {metrics.get('bounce_rate', 0):.1f}%

## 📊 Métricas de Conversión

- **Subidas de foto**: {metrics.get('uploads', 0)}
- **Análisis completados**: {metrics.get('completions', 0)}
- **Tasa de conversión**: {conversion_rate:.1f}%
- **Tasa de completación**: {completion_rate:.1f}%

## 😊 Métricas de Satisfacción

- **Feedback promedio**: {metrics.get('avg_feedback', 0):.1f}/5
- **NPS**: {metrics.get('nps', 0)}
- **% que pagarían**: {metrics.get('payment_intent', 0):.1f}%

## ⚙️ Métricas Técnicas

- **Tiempo promedio de análisis**: {metrics.get('avg_analysis_time', 0):.1f}s
- **Tasa de error**: {metrics.get('error_rate', 0):.1f}%

## 🎯 Análisis

### Fortalezas
- [Escribe fortalezas basadas en métricas]

### Áreas de Mejora
- [Escribe áreas de mejora basadas en métricas]

### Insights Clave
- [Insight 1]
- [Insight 2]
- [Insight 3]

## 🎯 Conclusión

[Escribe tu conclusión aquí basada en las métricas]

## 📝 Próximos Pasos

1. [ ] [Próximo paso 1]
2. [ ] [Próximo paso 2]
3. [ ] [Próximo paso 3]

## 💡 Aprendizajes Clave

- [Aprendizaje 1]
- [Aprendizaje 2]
- [Aprendizaje 3]

## 📊 Recomendaciones

{get_recommendations(metrics, conversion_rate, completion_rate)}

---
*Generado automáticamente el {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    report_file = results_dir / f"report_{datetime.now().strftime('%Y%m%d')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Reporte generado: {report_file}")
    print("💡 Edita el reporte para agregar conclusiones y próximos pasos.")

def get_recommendations(metrics, conversion_rate, completion_rate):
    """Genera recomendaciones basadas en métricas."""
    recommendations = []
    
    if conversion_rate < 20:
        recommendations.append("- ⚠️ Tasa de conversión baja. Revisa UX y propuesta de valor.")
    
    if completion_rate < 50:
        recommendations.append("- ⚠️ Muchos usuarios no completan. Revisa tiempo de análisis.")
    
    if metrics.get('avg_feedback', 0) < 3.5:
        recommendations.append("- ⚠️ Feedback bajo. Revisa calidad de resultados.")
    
    if metrics.get('payment_intent', 0) < 30:
        recommendations.append("- ⚠️ Baja intención de pago. Revisa propuesta de valor.")
    
    if metrics.get('avg_analysis_time', 0) > 15:
        recommendations.append("- ⚠️ Análisis muy lento. Optimiza backend.")
    
    if metrics.get('error_rate', 0) > 5:
        recommendations.append("- ⚠️ Alta tasa de errores. Revisa estabilidad.")
    
    if not recommendations:
        recommendations.append("- ✅ Métricas en buen rango. Continúa iterando!")
    
    return '\n'.join(recommendations)

if __name__ == "__main__":
    generate_report()






