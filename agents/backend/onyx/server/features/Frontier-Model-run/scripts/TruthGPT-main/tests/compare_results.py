#!/usr/bin/env python3
"""
Comparador de Resultados
Compara resultados de tests entre diferentes ejecuciones
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import argparse


class ResultsComparator:
    """Comparador de resultados de tests"""
    
    def __init__(self, file1: Path, file2: Path):
        self.file1 = Path(file1)
        self.file2 = Path(file2)
        self.data1 = self._load_data(file1)
        self.data2 = self._load_data(file2)
    
    def _load_data(self, file_path: Path) -> Dict:
        """Cargar datos de un archivo"""
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def compare_stats(self) -> Dict:
        """Comparar estadísticas"""
        stats1 = self.data1.get('stats', {})
        stats2 = self.data2.get('stats', {})
        
        comparison = {
            'total_runs': {
                'file1': stats1.get('total_runs', 0),
                'file2': stats2.get('total_runs', 0),
                'difference': stats2.get('total_runs', 0) - stats1.get('total_runs', 0)
            },
            'successful': {
                'file1': stats1.get('successful', 0),
                'file2': stats2.get('successful', 0),
                'difference': stats2.get('successful', 0) - stats1.get('successful', 0)
            },
            'failed': {
                'file1': stats1.get('failed', 0),
                'file2': stats2.get('failed', 0),
                'difference': stats2.get('failed', 0) - stats1.get('failed', 0)
            }
        }
        
        # Calcular tasas de éxito
        runs1 = stats1.get('total_runs', 0)
        runs2 = stats2.get('total_runs', 0)
        
        if runs1 > 0:
            success_rate1 = (stats1.get('successful', 0) / runs1) * 100
        else:
            success_rate1 = 0
        
        if runs2 > 0:
            success_rate2 = (stats2.get('successful', 0) / runs2) * 100
        else:
            success_rate2 = 0
        
        comparison['success_rate'] = {
            'file1': success_rate1,
            'file2': success_rate2,
            'difference': success_rate2 - success_rate1
        }
        
        return comparison
    
    def compare_performance(self) -> Dict:
        """Comparar rendimiento"""
        history1 = self.data1.get('stats', {}).get('history', [])
        history2 = self.data2.get('stats', {}).get('history', [])
        
        if not history1 or not history2:
            return {'error': 'Datos insuficientes para comparación'}
        
        times1 = [r.get('elapsed', 0) for r in history1]
        times2 = [r.get('elapsed', 0) for r in history2]
        
        avg1 = sum(times1) / len(times1) if times1 else 0
        avg2 = sum(times2) / len(times2) if times2 else 0
        
        min1 = min(times1) if times1 else 0
        min2 = min(times2) if times2 else 0
        
        max1 = max(times1) if times1 else 0
        max2 = max(times2) if times2 else 0
        
        return {
            'average_time': {
                'file1': avg1,
                'file2': avg2,
                'difference': avg2 - avg1,
                'percent_change': ((avg2 - avg1) / avg1 * 100) if avg1 > 0 else 0
            },
            'min_time': {
                'file1': min1,
                'file2': min2,
                'difference': min2 - min1
            },
            'max_time': {
                'file1': max1,
                'file2': max2,
                'difference': max2 - max1
            }
        }
    
    def find_regressions(self) -> List[Dict]:
        """Encontrar regresiones"""
        history1 = self.data1.get('stats', {}).get('history', [])
        history2 = self.data2.get('stats', {}).get('history', [])
        
        regressions = []
        
        # Comparar últimos runs
        recent1 = {r.get('timestamp', ''): r for r in history1[-10:]}
        recent2 = {r.get('timestamp', ''): r for r in history2[-10:]}
        
        for timestamp, run2 in recent2.items():
            if timestamp in recent1:
                run1 = recent1[timestamp]
                
                # Test que pasaba y ahora falla
                if run1.get('success') and not run2.get('success'):
                    regressions.append({
                        'type': 'new_failure',
                        'timestamp': timestamp,
                        'was_successful': True,
                        'now_successful': False
                    })
                
                # Test que se volvió más lento
                time1 = run1.get('elapsed', 0)
                time2 = run2.get('elapsed', 0)
                if time2 > time1 * 1.5:  # 50% más lento
                    regressions.append({
                        'type': 'performance_regression',
                        'timestamp': timestamp,
                        'time1': time1,
                        'time2': time2,
                        'percent_slower': ((time2 - time1) / time1 * 100) if time1 > 0 else 0
                    })
        
        return regressions
    
    def generate_comparison_report(self) -> Dict:
        """Generar reporte de comparación"""
        stats_comparison = self.compare_stats()
        perf_comparison = self.compare_performance()
        regressions = self.find_regressions()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'file1': str(self.file1),
            'file2': str(self.file2),
            'stats_comparison': stats_comparison,
            'performance_comparison': perf_comparison,
            'regressions': regressions,
            'summary': {
                'total_regressions': len(regressions),
                'performance_improved': perf_comparison.get('average_time', {}).get('percent_change', 0) < 0,
                'success_rate_improved': stats_comparison.get('success_rate', {}).get('difference', 0) > 0
            }
        }
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        print("=" * 60)
        print("📊 COMPARACIÓN DE RESULTADOS")
        print("=" * 60)
        print()
        
        print(f"📁 Archivo 1: {report['file1']}")
        print(f"📁 Archivo 2: {report['file2']}")
        print()
        
        # Estadísticas
        stats = report['stats_comparison']
        print("📈 Estadísticas:")
        print(f"   Total runs: {stats['total_runs']['file1']} → {stats['total_runs']['file2']} "
              f"({stats['total_runs']['difference']:+d})")
        print(f"   Exitosos: {stats['successful']['file1']} → {stats['successful']['file2']} "
              f"({stats['successful']['difference']:+d})")
        print(f"   Fallidos: {stats['failed']['file1']} → {stats['failed']['file2']} "
              f"({stats['failed']['difference']:+d})")
        print(f"   Tasa de éxito: {stats['success_rate']['file1']:.1f}% → "
              f"{stats['success_rate']['file2']:.1f}% "
              f"({stats['success_rate']['difference']:+.1f}%)")
        print()
        
        # Rendimiento
        if 'error' not in report['performance_comparison']:
            perf = report['performance_comparison']
            avg = perf['average_time']
            print("⏱️  Rendimiento:")
            print(f"   Tiempo promedio: {avg['file1']:.2f}s → {avg['file2']:.2f}s "
                  f"({avg['difference']:+.2f}s, {avg['percent_change']:+.1f}%)")
            print()
        
        # Regresiones
        if report['regressions']:
            print(f"⚠️  Regresiones encontradas: {len(report['regressions'])}")
            for reg in report['regressions'][:5]:  # Primeras 5
                if reg['type'] == 'new_failure':
                    print(f"   ❌ Nuevo fallo en {reg['timestamp'][:19]}")
                elif reg['type'] == 'performance_regression':
                    print(f"   ⏱️  Degradación de rendimiento: "
                          f"{reg['time1']:.2f}s → {reg['time2']:.2f}s "
                          f"({reg['percent_slower']:.1f}% más lento)")
            print()
        
        # Resumen
        summary = report['summary']
        print("📋 Resumen:")
        print(f"   Regresiones totales: {summary['total_regressions']}")
        print(f"   Rendimiento mejorado: {'✅' if summary['performance_improved'] else '❌'}")
        print(f"   Tasa de éxito mejorada: {'✅' if summary['success_rate_improved'] else '❌'}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Comparar resultados de tests')
    parser.add_argument('file1', type=Path, help='Primer archivo de resultados')
    parser.add_argument('file2', type=Path, help='Segundo archivo de resultados')
    parser.add_argument('--output', type=Path, help='Archivo de salida JSON')
    
    args = parser.parse_args()
    
    comparator = ResultsComparator(args.file1, args.file2)
    report = comparator.generate_comparison_report()
    comparator.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Reporte guardado en: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

