#!/usr/bin/env python3
"""
Test Runner Avanzado
Ejecutor de tests con características avanzadas: paralelización, retry, flaky detection
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import concurrent.futures
import time


class AdvancedTestRunner:
    """Ejecutor avanzado de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'flaky': [],
            'slow': [],
            'results': []
        }
    
    def run_tests_parallel(self, categories: List[str], max_workers: int = 4) -> Dict:
        """Ejecutar tests en paralelo"""
        print(f"🚀 Ejecutando tests en paralelo ({max_workers} workers)...\n")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_category, category): category
                for category in categories
            }
            
            for future in concurrent.futures.as_completed(futures):
                category = futures[future]
                try:
                    result = future.result()
                    self._process_result(category, result)
                except Exception as e:
                    print(f"❌ Error en {category}: {e}")
                    self.results['failed'] += 1
        
        return self.results
    
    def _run_category(self, category: str) -> Dict:
        """Ejecutar una categoría de tests"""
        result = subprocess.run(
            ['python', 'run_tests.py', category, '-v', '--tb=short'],
            cwd=self.base_path,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        return {
            'category': category,
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def _process_result(self, category: str, result: Dict):
        """Procesar resultado de test"""
        self.results['total'] += 1
        
        if result['success']:
            self.results['passed'] += 1
            print(f"✅ {category}: PASSED")
        else:
            self.results['failed'] += 1
            print(f"❌ {category}: FAILED")
        
        self.results['results'].append({
            'category': category,
            'timestamp': datetime.now().isoformat(),
            **result
        })
    
    def run_with_retry(self, category: str, max_retries: int = 3) -> Dict:
        """Ejecutar tests con reintentos"""
        print(f"🔄 Ejecutando {category} con hasta {max_retries} reintentos...\n")
        
        for attempt in range(1, max_retries + 1):
            print(f"Intento {attempt}/{max_retries}...")
            
            result = subprocess.run(
                ['python', 'run_tests.py', category, '-v'],
                cwd=self.base_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {category} pasó en el intento {attempt}")
                return {'success': True, 'attempts': attempt, 'result': result}
            
            if attempt < max_retries:
                wait_time = attempt * 2  # Backoff exponencial
                print(f"⏳ Esperando {wait_time}s antes del siguiente intento...")
                time.sleep(wait_time)
        
        print(f"❌ {category} falló después de {max_retries} intentos")
        return {'success': False, 'attempts': max_retries, 'result': result}
    
    def detect_flaky_tests(self, category: str, runs: int = 5) -> List[Dict]:
        """Detectar tests flaky ejecutando múltiples veces"""
        print(f"🔍 Detectando tests flaky en {category} ({runs} ejecuciones)...\n")
        
        results_by_test = {}
        
        for run_num in range(1, runs + 1):
            print(f"Ejecución {run_num}/{runs}...")
            
            result = subprocess.run(
                ['pytest', f'core/{category}/', '-v', '--tb=no'],
                cwd=self.base_path,
                capture_output=True,
                text=True
            )
            
            # Parsear resultados
            for line in result.stdout.splitlines():
                if 'PASSED' in line or 'FAILED' in line:
                    test_name = line.split()[0] if line.split() else None
                    if test_name:
                        if test_name not in results_by_test:
                            results_by_test[test_name] = []
                        results_by_test[test_name].append('PASSED' in line)
        
        # Identificar flaky tests
        flaky_tests = []
        for test_name, results in results_by_test.items():
            if len(results) == runs:
                passed_count = sum(results)
                if 0 < passed_count < runs:  # Algunos pasan, algunos fallan
                    flaky_tests.append({
                        'test': test_name,
                        'passed': passed_count,
                        'failed': runs - passed_count,
                        'flakiness_rate': (runs - passed_count) / runs * 100
                    })
        
        return flaky_tests
    
    def generate_summary(self) -> str:
        """Generar resumen de resultados"""
        total = self.results['total']
        passed = self.results['passed']
        failed = self.results['failed']
        success_rate = (passed / total * 100) if total > 0 else 0
        
        summary = f"""
{'='*60}
📊 RESUMEN DE EJECUCIÓN
{'='*60}
Total: {total}
✅ Exitosos: {passed}
❌ Fallidos: {failed}
📈 Tasa de éxito: {success_rate:.1f}%

"""
        
        if self.results['flaky']:
            summary += f"⚠️  Tests flaky detectados: {len(self.results['flaky'])}\n"
        
        if self.results['slow']:
            summary += f"⏱️  Tests lentos: {len(self.results['slow'])}\n"
        
        return summary


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ejecutor avanzado de tests')
    parser.add_argument('--parallel', action='store_true',
                       help='Ejecutar tests en paralelo')
    parser.add_argument('--workers', type=int, default=4,
                       help='Número de workers para ejecución paralela')
    parser.add_argument('--retry', type=int, default=0,
                       help='Número de reintentos')
    parser.add_argument('--detect-flaky', action='store_true',
                       help='Detectar tests flaky')
    parser.add_argument('--category', default='all',
                       help='Categoría de tests')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    runner = AdvancedTestRunner(args.base_path)
    
    if args.detect_flaky:
        flaky = runner.detect_flaky_tests(args.category)
        if flaky:
            print("\n⚠️  Tests flaky detectados:")
            for test in flaky:
                print(f"   {test['test']}: {test['flakiness_rate']:.1f}% flaky")
        else:
            print("\n✅ No se detectaron tests flaky")
    
    elif args.retry > 0:
        result = runner.run_with_retry(args.category, args.retry)
        print(runner.generate_summary())
    
    elif args.parallel:
        categories = ['unit', 'integration', 'analyzers']
        results = runner.run_tests_parallel(categories, args.workers)
        print(runner.generate_summary())
    
    else:
        print("Usa --parallel, --retry o --detect-flaky")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

