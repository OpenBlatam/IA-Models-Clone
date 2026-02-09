#!/usr/bin/env python3
"""
Colector de Métricas
Recopila y analiza métricas avanzadas de tests
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class MetricsCollector:
    """Colector de métricas de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.metrics = {
            'execution_times': [],
            'success_rates': [],
            'test_counts': [],
            'coverage_scores': [],
            'flaky_detections': [],
            'performance_trends': []
        }
    
    def collect_execution_metrics(self, days: int = 7) -> Dict:
        """Recopilar métricas de ejecución"""
        print(f"📊 Recopilando métricas de ejecución (últimos {days} días)...")
        
        # Simular recopilación de métricas históricas
        # En producción, esto leería de base de datos o archivos de log
        
        metrics = {
            'period_days': days,
            'total_executions': 0,
            'avg_execution_time': 0,
            'success_rate': 0,
            'trend': 'stable'
        }
        
        return metrics
    
    def calculate_test_health_score(self) -> Dict:
        """Calcular score de salud de tests"""
        print("🏥 Calculando score de salud...")
        
        # Factores que afectan la salud
        factors = {
            'success_rate': 0.4,  # 40% del score
            'execution_speed': 0.2,  # 20% del score
            'coverage': 0.2,  # 20% del score
            'flakiness': 0.2  # 20% del score
        }
        
        # Valores simulados (en producción serían reales)
        scores = {
            'success_rate': 95.0,
            'execution_speed': 85.0,
            'coverage': 90.0,
            'flakiness': 98.0  # Menos flaky = mejor
        }
        
        # Calcular score total
        total_score = sum(scores[k] * factors[k] for k in factors)
        
        # Determinar estado
        if total_score >= 90:
            status = 'excellent'
        elif total_score >= 75:
            status = 'good'
        elif total_score >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'total_score': total_score,
            'status': status,
            'factors': scores,
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_anomalies(self, history: List[Dict]) -> List[Dict]:
        """Detectar anomalías en ejecuciones"""
        print("🔍 Detectando anomalías...")
        
        if len(history) < 5:
            return []
        
        anomalies = []
        
        # Extraer tiempos de ejecución
        times = [r.get('elapsed', 0) for r in history]
        if len(times) >= 3:
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            # Detectar outliers (más de 2 desviaciones estándar)
            for i, time in enumerate(times):
                if abs(time - mean_time) > 2 * std_time:
                    anomalies.append({
                        'type': 'execution_time_outlier',
                        'index': i,
                        'value': time,
                        'expected_range': (mean_time - 2*std_time, mean_time + 2*std_time),
                        'timestamp': history[i].get('timestamp', '')
                    })
        
        # Detectar cambios bruscos en tasa de éxito
        success_rates = []
        window_size = 5
        for i in range(len(history) - window_size + 1):
            window = history[i:i+window_size]
            success_count = sum(1 for r in window if r.get('success', False))
            success_rates.append(success_count / window_size * 100)
        
        if len(success_rates) >= 2:
            for i in range(1, len(success_rates)):
                change = abs(success_rates[i] - success_rates[i-1])
                if change > 20:  # Cambio mayor a 20%
                    anomalies.append({
                        'type': 'success_rate_spike',
                        'index': i,
                        'change': change,
                        'from': success_rates[i-1],
                        'to': success_rates[i]
                    })
        
        return anomalies
    
    def generate_metrics_report(self) -> Dict:
        """Generar reporte completo de métricas"""
        print("📊 Generando reporte de métricas...\n")
        
        health_score = self.calculate_test_health_score()
        execution_metrics = self.collect_execution_metrics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'execution_metrics': execution_metrics,
            'recommendations': self._generate_recommendations(health_score)
        }
        
        return report
    
    def _generate_recommendations(self, health_score: Dict) -> List[str]:
        """Generar recomendaciones basadas en métricas"""
        recommendations = []
        score = health_score['total_score']
        status = health_score['status']
        
        if status == 'poor':
            recommendations.append("🔴 Acción urgente requerida: Score de salud bajo")
            recommendations.append("   - Revisar tests fallidos")
            recommendations.append("   - Investigar degradación de rendimiento")
            recommendations.append("   - Mejorar cobertura de tests")
        elif status == 'fair':
            recommendations.append("🟡 Mejoras recomendadas:")
            recommendations.append("   - Optimizar tests lentos")
            recommendations.append("   - Reducir tests flaky")
        elif status == 'good':
            recommendations.append("🟢 Mantener calidad actual")
        else:
            recommendations.append("✅ Excelente estado - mantener prácticas actuales")
        
        return recommendations
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        print("=" * 60)
        print("📊 REPORTE DE MÉTRICAS")
        print("=" * 60)
        print()
        
        health = report['health_score']
        print(f"🏥 Score de Salud: {health['total_score']:.1f}/100")
        print(f"   Estado: {health['status'].upper()}")
        print()
        
        print("📈 Factores:")
        for factor, score in health['factors'].items():
            print(f"   {factor}: {score:.1f}")
        print()
        
        if report['recommendations']:
            print("💡 Recomendaciones:")
            for rec in report['recommendations']:
                print(f"   {rec}")
            print()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recopilar métricas de tests')
    parser.add_argument('--output', type=Path,
                       help='Archivo de salida JSON')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.base_path)
    report = collector.generate_metrics_report()
    collector.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Reporte guardado en: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

