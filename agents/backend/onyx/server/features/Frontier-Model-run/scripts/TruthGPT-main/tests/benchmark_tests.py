#!/usr/bin/env python3
"""
Benchmark de Tests
Ejecuta benchmarks de rendimiento para tests
"""

import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict


class TestBenchmark:
    """Benchmark de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.results = []
    
    def run_benchmark(self, category: str, iterations: int = 3) -> Dict:
        """Ejecutar benchmark de una categoría"""
        print(f"🔬 Ejecutando benchmark: {category} ({iterations} iteraciones)...")
        
        times = []
        for i in range(iterations):
            start = time.time()
            
            try:
                result = subprocess.run(
                    ['python', 'run_tests.py', category, '-v', '--tb=short'],
                    cwd=self.base_path,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutos máximo
                )
                elapsed = time.time() - start
                times.append(elapsed)
                
                if result.returncode != 0:
                    print(f"  ⚠️  Iteración {i+1} falló")
                else:
                    print(f"  ✅ Iteración {i+1}: {elapsed:.2f}s")
            
            except subprocess.TimeoutExpired:
                print(f"  ⏱️  Iteración {i+1} excedió tiempo límite")
                times.append(600.0)
            except Exception as e:
                print(f"  ❌ Iteración {i+1} error: {e}")
                times.append(None)
        
        # Calcular estadísticas
        valid_times = [t for t in times if t is not None]
        
        if not valid_times:
            return {
                'category': category,
                'status': 'failed',
                'iterations': iterations
            }
        
        benchmark_result = {
            'category': category,
            'status': 'success',
            'iterations': iterations,
            'times': valid_times,
            'avg_time': sum(valid_times) / len(valid_times),
            'min_time': min(valid_times),
            'max_time': max(valid_times),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  📊 Promedio: {benchmark_result['avg_time']:.2f}s")
        print(f"  📊 Mínimo: {benchmark_result['min_time']:.2f}s")
        print(f"  📊 Máximo: {benchmark_result['max_time']:.2f}s")
        print()
        
        return benchmark_result
    
    def benchmark_all(self, categories: List[str] = None, iterations: int = 3) -> List[Dict]:
        """Ejecutar benchmarks de todas las categorías"""
        if categories is None:
            categories = [
                'unit',
                'integration',
                'analyzers',
                'performance',
                'quality',
            ]
        
        print("🚀 Iniciando benchmarks de tests...\n")
        print("=" * 60)
        print()
        
        results = []
        for category in categories:
            result = self.run_benchmark(category, iterations)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict]) -> Dict:
        """Generar reporte de benchmarks"""
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') != 'success']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_categories': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'total_time': sum(r.get('avg_time', 0) for r in successful),
            },
            'results': results,
            'ranking': sorted(
                successful,
                key=lambda x: x.get('avg_time', float('inf'))
            )
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        print("=" * 60)
        print("📊 REPORTE DE BENCHMARKS")
        print("=" * 60)
        print()
        
        summary = report['summary']
        print("📈 RESUMEN:")
        print(f"  Categorías totales: {summary['total_categories']}")
        print(f"  Exitosas: {summary['successful']}")
        print(f"  Fallidas: {summary['failed']}")
        print(f"  Tiempo total: {summary['total_time']:.2f}s")
        print()
        
        print("🏆 RANKING (más rápidos primero):")
        for i, result in enumerate(report['ranking'][:10], 1):
            print(f"  {i}. {result['category']}: {result['avg_time']:.2f}s")
        print()
        
        print("📋 DETALLES:")
        for result in report['results']:
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            print(f"  {status_icon} {result['category']}")
            if result.get('status') == 'success':
                print(f"     Promedio: {result['avg_time']:.2f}s")
                print(f"     Rango: {result['min_time']:.2f}s - {result['max_time']:.2f}s")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark de tests')
    parser.add_argument('--categories', nargs='+', help='Categorías a benchmarkear')
    parser.add_argument('--iterations', type=int, default=3, help='Número de iteraciones')
    parser.add_argument('--output', type=Path, help='Archivo de salida JSON')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    benchmark = TestBenchmark(args.base_path)
    results = benchmark.benchmark_all(args.categories, args.iterations)
    report = benchmark.generate_report(results)
    
    benchmark.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Reporte guardado en: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

