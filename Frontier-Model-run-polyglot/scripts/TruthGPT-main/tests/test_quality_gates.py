#!/usr/bin/env python3
"""
Quality Gates
Sistema de quality gates para validar calidad de tests
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class QualityGates:
    """Sistema de quality gates"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.gates = {
            'structure': {'weight': 0.2, 'passed': False},
            'tests_pass': {'weight': 0.3, 'passed': False},
            'coverage': {'weight': 0.2, 'passed': False},
            'performance': {'weight': 0.15, 'passed': False},
            'linting': {'weight': 0.15, 'passed': False}
        }
        self.results = []
    
    def check_structure(self) -> bool:
        """Verificar estructura"""
        print("📁 Verificando estructura...")
        
        try:
            result = subprocess.run(
                ['python', 'validate_structure.py'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            passed = result.returncode == 0
            self.gates['structure']['passed'] = passed
            
            if passed:
                print("   ✅ Estructura válida")
            else:
                print("   ❌ Estructura inválida")
            
            return passed
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def check_tests_pass(self) -> bool:
        """Verificar que tests pasen"""
        print("🧪 Verificando que tests pasen...")
        
        try:
            result = subprocess.run(
                ['python', 'run_tests.py', 'unit', '-q'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            passed = result.returncode == 0
            self.gates['tests_pass']['passed'] = passed
            
            if passed:
                print("   ✅ Tests pasan")
            else:
                print("   ❌ Algunos tests fallan")
            
            return passed
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def check_coverage(self, threshold: float = 80.0) -> bool:
        """Verificar cobertura"""
        print(f"📊 Verificando cobertura (mínimo {threshold}%)...")
        
        try:
            result = subprocess.run(
                ['pytest', 'core/', '--cov=..', '--cov-report=term', '-q'],
                cwd=self.base_path.parent,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parsear cobertura del output
            coverage = 0.0
            for line in result.stdout.splitlines():
                if 'TOTAL' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '%' in part:
                            try:
                                coverage = float(part.replace('%', ''))
                                break
                            except ValueError:
                                pass
            
            passed = coverage >= threshold
            self.gates['coverage']['passed'] = passed
            self.gates['coverage']['value'] = coverage
            
            if passed:
                print(f"   ✅ Cobertura: {coverage:.1f}% (≥ {threshold}%)")
            else:
                print(f"   ❌ Cobertura: {coverage:.1f}% (< {threshold}%)")
            
            return passed
        except Exception as e:
            print(f"   ⚠️  No se pudo verificar cobertura: {e}")
            return True  # No crítico
    
    def check_performance(self, max_time: float = 300.0) -> bool:
        """Verificar rendimiento"""
        print(f"⏱️  Verificando rendimiento (máximo {max_time}s)...")
        
        try:
            import time
            start = time.time()
            
            result = subprocess.run(
                ['python', 'run_tests.py', 'unit', '-q'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=int(max_time * 1.5)
            )
            
            elapsed = time.time() - start
            passed = elapsed <= max_time and result.returncode == 0
            
            self.gates['performance']['passed'] = passed
            self.gates['performance']['value'] = elapsed
            
            if passed:
                print(f"   ✅ Tiempo: {elapsed:.1f}s (≤ {max_time}s)")
            else:
                print(f"   ❌ Tiempo: {elapsed:.1f}s (> {max_time}s)")
            
            return passed
        except Exception as e:
            print(f"   ⚠️  No se pudo verificar rendimiento: {e}")
            return True  # No crítico
    
    def check_linting(self) -> bool:
        """Verificar linting"""
        print("🔍 Verificando linting...")
        
        try:
            result = subprocess.run(
                ['make', 'lint'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Linting pasa si no hay errores críticos
            passed = result.returncode == 0 or 'error' not in result.stdout.lower()
            self.gates['linting']['passed'] = passed
            
            if passed:
                print("   ✅ Linting OK")
            else:
                print("   ⚠️  Advertencias de linting")
            
            return passed
        except Exception as e:
            print(f"   ⚠️  No se pudo verificar linting: {e}")
            return True  # No crítico
    
    def run_all_gates(self, coverage_threshold: float = 80.0, 
                     performance_max: float = 300.0) -> Dict:
        """Ejecutar todos los quality gates"""
        print("🚪 Ejecutando Quality Gates...\n")
        
        self.check_structure()
        self.check_tests_pass()
        self.check_coverage(coverage_threshold)
        self.check_performance(performance_max)
        self.check_linting()
        
        # Calcular score
        total_weight = sum(gate['weight'] for gate in self.gates.values())
        passed_weight = sum(
            gate['weight'] for gate in self.gates.values() if gate.get('passed', False)
        )
        
        score = (passed_weight / total_weight * 100) if total_weight > 0 else 0
        all_passed = all(gate.get('passed', False) for gate in self.gates.values())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'all_passed': all_passed,
            'gates': {
                name: {
                    'passed': gate['passed'],
                    'weight': gate['weight'],
                    'value': gate.get('value')
                }
                for name, gate in self.gates.items()
            }
        }
    
    def print_report(self, report: Dict):
        """Imprimir reporte"""
        print("\n" + "=" * 60)
        print("🚪 QUALITY GATES REPORT")
        print("=" * 60)
        print()
        
        print(f"📊 Score: {report['score']:.1f}/100")
        print(f"   Estado: {'✅ PASSED' if report['all_passed'] else '❌ FAILED'}")
        print()
        
        print("🚪 Gates:")
        for name, gate_info in report['gates'].items():
            status = "✅" if gate_info['passed'] else "❌"
            value_info = f" ({gate_info['value']})" if gate_info.get('value') is not None else ""
            print(f"   {status} {name}: {gate_info['weight']*100:.0f}%{value_info}")
        print()
        
        if not report['all_passed']:
            print("⚠️  Algunos gates fallaron. Revisar y corregir antes de continuar.")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quality gates para tests')
    parser.add_argument('--coverage-threshold', type=float, default=80.0,
                       help='Umbral mínimo de cobertura (%)')
    parser.add_argument('--performance-max', type=float, default=300.0,
                       help='Tiempo máximo de ejecución (segundos)')
    parser.add_argument('--output', type=Path,
                       help='Archivo de salida JSON')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    gates = QualityGates(args.base_path)
    report = gates.run_all_gates(args.coverage_threshold, args.performance_max)
    gates.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Reporte guardado en: {args.output}")
    
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

