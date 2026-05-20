#!/usr/bin/env python3
"""
Profiler de Tests
Analiza el rendimiento de tests y genera perfiles detallados
"""

import sys
import cProfile
import pstats
import io
import subprocess
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


class TestProfiler:
    """Profiler de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    def profile_test(self, test_path: str, output: Optional[Path] = None) -> Dict:
        """Profilear un test específico"""
        print(f"🔬 Profileando test: {test_path}...")
        
        # Crear profiler
        profiler = cProfile.Profile()
        
        # Ejecutar test con profiling
        profiler.enable()
        result = subprocess.run(
            ['pytest', test_path, '-v'],
            cwd=self.base_path,
            capture_output=True,
            text=True
        )
        profiler.disable()
        
        # Generar estadísticas
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 funciones
        
        profile_data = {
            'test_path': test_path,
            'exit_code': result.returncode,
            'success': result.returncode == 0,
            'profile_stats': stats_stream.getvalue(),
            'total_calls': stats.total_calls,
            'primitive_calls': stats.primitive_calls,
            'total_time': stats.total_tt
        }
        
        if output:
            # Guardar perfil completo
            profiler.dump_stats(str(output))
            profile_data['profile_file'] = str(output)
            print(f"✅ Perfil guardado en: {output}")
        
        return profile_data
    
    def analyze_slow_tests(self, category: str = 'all', threshold: float = 1.0) -> Dict:
        """Analizar tests lentos"""
        print(f"🔍 Analizando tests lentos (>{threshold}s)...")
        
        result = subprocess.run(
            ['pytest', 'core/', '--durations=0', '-v'],
            cwd=self.base_path,
            capture_output=True,
            text=True
        )
        
        slow_tests = []
        for line in result.stdout.splitlines():
            # Buscar líneas con tiempos
            if 'passed' in line or 'failed' in line:
                # Extraer tiempo y nombre del test
                import re
                time_match = re.search(r'(\d+\.\d+)s', line)
                test_match = re.search(r'test_\w+', line)
                
                if time_match and test_match:
                    elapsed = float(time_match.group(1))
                    if elapsed > threshold:
                        slow_tests.append({
                            'name': test_match.group(),
                            'time': elapsed,
                            'line': line.strip()
                        })
        
        # Ordenar por tiempo
        slow_tests.sort(key=lambda x: x['time'], reverse=True)
        
        return {
            'threshold': threshold,
            'slow_tests_count': len(slow_tests),
            'slow_tests': slow_tests[:20]  # Top 20
        }
    
    def generate_profile_report(self, test_path: str) -> Dict:
        """Generar reporte de profiling"""
        profile_data = self.profile_test(test_path)
        slow_analysis = self.analyze_slow_tests()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'test_profile': profile_data,
            'slow_tests_analysis': slow_analysis
        }
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        print("=" * 60)
        print("🔬 REPORTE DE PROFILING")
        print("=" * 60)
        print()
        
        profile = report['test_profile']
        print(f"📊 Test: {profile['test_path']}")
        print(f"   Estado: {'✅ Exitoso' if profile['success'] else '❌ Fallido'}")
        print(f"   Total calls: {profile['total_calls']}")
        print(f"   Tiempo total: {profile['total_time']:.4f}s")
        print()
        
        print("📋 Top funciones (por tiempo acumulado):")
        print(profile['profile_stats'])
        print()
        
        slow = report['slow_tests_analysis']
        print(f"⏱️  Tests lentos (>{slow['threshold']}s): {slow['slow_tests_count']}")
        for test in slow['slow_tests'][:10]:
            print(f"   {test['name']}: {test['time']:.2f}s")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profilear tests')
    parser.add_argument('--test', help='Test específico a profilear')
    parser.add_argument('--category', default='all',
                       help='Categoría de tests')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Umbral para tests lentos (segundos)')
    parser.add_argument('--output', type=Path,
                       help='Archivo de salida para perfil')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    profiler = TestProfiler(args.base_path)
    
    if args.test:
        report = profiler.generate_profile_report(args.test)
        profiler.print_report(report)
    else:
        # Analizar tests lentos
        analysis = profiler.analyze_slow_tests(args.category, args.threshold)
        print(f"⏱️  Tests lentos encontrados: {analysis['slow_tests_count']}")
        for test in analysis['slow_tests'][:20]:
            print(f"   {test['name']}: {test['time']:.2f}s")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

