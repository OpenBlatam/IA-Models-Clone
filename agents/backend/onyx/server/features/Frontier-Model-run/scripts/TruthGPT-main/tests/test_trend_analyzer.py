#!/usr/bin/env python3
"""
Analizador de Tendencias de Tests
Analiza tendencias a largo plazo en ejecuciones de tests
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class TestTrendAnalyzer:
    """Analizador de tendencias de tests"""
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Cargar datos históricos"""
        if not self.data_path.exists():
            return {'stats': {'history': []}}
        
        with open(self.data_path, 'r') as f:
            return json.load(f)
    
    def analyze_trends(self, days: int = 30) -> Dict:
        """Analizar tendencias"""
        print(f"📈 Analizando tendencias (últimos {days} días)...")
        
        history = self.data.get('stats', {}).get('history', [])
        
        if not history:
            return {'error': 'No hay datos históricos'}
        
        # Filtrar por período
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff]
        
        if len(recent) < 3:
            return {'error': 'Datos insuficientes para análisis de tendencias'}
        
        # Análisis de tendencias
        trends = {
            'period_days': days,
            'total_runs': len(recent),
            'success_rate_trend': self._analyze_success_rate_trend(recent),
            'execution_time_trend': self._analyze_execution_time_trend(recent),
            'failure_trend': self._analyze_failure_trend(recent),
            'predictions': self._generate_predictions(recent)
        }
        
        return trends
    
    def _analyze_success_rate_trend(self, history: List[Dict]) -> Dict:
        """Analizar tendencia de tasa de éxito"""
        success_rates = []
        
        # Calcular tasa de éxito por ventana
        window_size = max(3, len(history) // 10)
        for i in range(0, len(history) - window_size + 1, window_size):
            window = history[i:i+window_size]
            success_count = sum(1 for r in window if r.get('success', False))
            success_rates.append(success_count / len(window) * 100)
        
        if len(success_rates) < 2:
            return {'trend': 'insufficient_data'}
        
        # Determinar tendencia
        first_half = statistics.mean(success_rates[:len(success_rates)//2])
        second_half = statistics.mean(success_rates[len(success_rates)//2:])
        
        change = second_half - first_half
        
        if abs(change) < 2:
            trend = 'stable'
        elif change > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        return {
            'trend': trend,
            'current': success_rates[-1] if success_rates else 0,
            'average': statistics.mean(success_rates),
            'change': change
        }
    
    def _analyze_execution_time_trend(self, history: List[Dict]) -> Dict:
        """Analizar tendencia de tiempo de ejecución"""
        times = [r.get('elapsed', 0) for r in history]
        
        if len(times) < 2:
            return {'trend': 'insufficient_data'}
        
        # Dividir en mitades
        first_half = times[:len(times)//2]
        second_half = times[len(times)//2:]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        change_percent = ((avg_second - avg_first) / avg_first * 100) if avg_first > 0 else 0
        
        if abs(change_percent) < 5:
            trend = 'stable'
        elif change_percent > 0:
            trend = 'slowing'
        else:
            trend = 'speeding'
        
        return {
            'trend': trend,
            'current_avg': avg_second,
            'previous_avg': avg_first,
            'change_percent': change_percent
        }
    
    def _analyze_failure_trend(self, history: List[Dict]) -> Dict:
        """Analizar tendencia de fallos"""
        failures = [1 if not r.get('success', False) else 0 for r in history]
        
        if len(failures) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calcular tasa de fallos por ventana
        window_size = max(3, len(history) // 10)
        failure_rates = []
        
        for i in range(0, len(failures) - window_size + 1, window_size):
            window = failures[i:i+window_size]
            failure_rates.append(sum(window) / len(window) * 100)
        
        if len(failure_rates) < 2:
            return {'trend': 'insufficient_data'}
        
        first_half = statistics.mean(failure_rates[:len(failure_rates)//2])
        second_half = statistics.mean(failure_rates[len(failure_rates)//2:])
        
        change = second_half - first_half
        
        if abs(change) < 2:
            trend = 'stable'
        elif change > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'current_rate': failure_rates[-1] if failure_rates else 0,
            'change': change
        }
    
    def _generate_predictions(self, history: List[Dict]) -> Dict:
        """Generar predicciones basadas en tendencias"""
        if len(history) < 5:
            return {'error': 'Datos insuficientes para predicciones'}
        
        # Predicción simple basada en tendencia lineal
        recent_times = [r.get('elapsed', 0) for r in history[-10:]]
        recent_success = [1 if r.get('success') else 0 for r in history[-10:]]
        
        # Predicción de tiempo (promedio móvil)
        predicted_time = statistics.mean(recent_times)
        
        # Predicción de éxito (promedio móvil)
        predicted_success_rate = statistics.mean(recent_success) * 100
        
        return {
            'predicted_execution_time': predicted_time,
            'predicted_success_rate': predicted_success_rate,
            'confidence': 'medium' if len(history) >= 10 else 'low'
        }
    
    def generate_trend_report(self, days: int = 30) -> Dict:
        """Generar reporte de tendencias"""
        trends = self.analyze_trends(days)
        
        if 'error' in trends:
            return trends
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'period_days': days,
            'trends': trends,
            'summary': self._generate_summary(trends)
        }
        
        return report
    
    def _generate_summary(self, trends: Dict) -> str:
        """Generar resumen de tendencias"""
        success_trend = trends.get('success_rate_trend', {}).get('trend', 'unknown')
        time_trend = trends.get('execution_time_trend', {}).get('trend', 'unknown')
        failure_trend = trends.get('failure_trend', {}).get('trend', 'unknown')
        
        summary_parts = []
        
        if success_trend == 'improving':
            summary_parts.append("✅ Tasa de éxito mejorando")
        elif success_trend == 'declining':
            summary_parts.append("⚠️  Tasa de éxito disminuyendo")
        
        if time_trend == 'speeding':
            summary_parts.append("⚡ Tests ejecutándose más rápido")
        elif time_trend == 'slowing':
            summary_parts.append("🐌 Tests ejecutándose más lento")
        
        if failure_trend == 'decreasing':
            summary_parts.append("✅ Fallos disminuyendo")
        elif failure_trend == 'increasing':
            summary_parts.append("❌ Fallos aumentando")
        
        return " | ".join(summary_parts) if summary_parts else "Tendencias estables"
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        if 'error' in report:
            print(f"❌ Error: {report['error']}")
            return
        
        print("=" * 60)
        print("📈 REPORTE DE TENDENCIAS")
        print("=" * 60)
        print()
        
        trends = report['trends']
        
        print(f"📊 Período: {trends['period_days']} días")
        print(f"   Total ejecuciones: {trends['total_runs']}")
        print()
        
        # Tasa de éxito
        success = trends['success_rate_trend']
        print(f"✅ Tasa de Éxito:")
        print(f"   Tendencia: {success['trend']}")
        print(f"   Actual: {success['current']:.1f}%")
        print(f"   Promedio: {success['average']:.1f}%")
        print(f"   Cambio: {success['change']:+.1f}%")
        print()
        
        # Tiempo de ejecución
        time_trend = trends['execution_time_trend']
        print(f"⏱️  Tiempo de Ejecución:")
        print(f"   Tendencia: {time_trend['trend']}")
        print(f"   Actual: {time_trend['current_avg']:.2f}s")
        print(f"   Anterior: {time_trend['previous_avg']:.2f}s")
        print(f"   Cambio: {time_trend['change_percent']:+.1f}%")
        print()
        
        # Fallos
        failures = trends['failure_trend']
        print(f"❌ Tasa de Fallos:")
        print(f"   Tendencia: {failures['trend']}")
        print(f"   Actual: {failures['current_rate']:.1f}%")
        print(f"   Cambio: {failures['change']:+.1f}%")
        print()
        
        # Predicciones
        if 'predictions' in trends and 'error' not in trends['predictions']:
            pred = trends['predictions']
            print(f"🔮 Predicciones:")
            print(f"   Tiempo estimado: {pred['predicted_execution_time']:.2f}s")
            print(f"   Tasa de éxito estimada: {pred['predicted_success_rate']:.1f}%")
            print(f"   Confianza: {pred['confidence']}")
            print()
        
        print(f"📋 Resumen: {report['summary']}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar tendencias de tests')
    parser.add_argument('--input', type=Path, default=Path('monitor_stats.json'),
                       help='Archivo JSON de datos históricos')
    parser.add_argument('--days', type=int, default=30,
                       help='Días a analizar')
    parser.add_argument('--output', type=Path,
                       help='Archivo de salida JSON')
    
    args = parser.parse_args()
    
    analyzer = TestTrendAnalyzer(args.input)
    report = analyzer.generate_trend_report(args.days)
    analyzer.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Reporte guardado en: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

