#!/usr/bin/env python3
"""
Optimizador de Tests
Analiza y sugiere optimizaciones para mejorar rendimiento de tests
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import re


class TestOptimizer:
    """Optimizador de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.suggestions = []
    
    def analyze_test_structure(self) -> Dict:
        """Analizar estructura de tests para optimizaciones"""
        print("🔍 Analizando estructura de tests...")
        
        issues = {
            'duplicate_setup': [],
            'slow_imports': [],
            'missing_fixtures': [],
            'inefficient_assertions': []
        }
        
        # Buscar archivos de test
        for test_file in (self.base_path / 'core').rglob('test_*.py'):
            with open(test_file, 'r') as f:
                content = f.read()
                
                # Detectar setup duplicado
                setup_count = content.count('setUp(')
                if setup_count > 1:
                    issues['duplicate_setup'].append({
                        'file': str(test_file.relative_to(self.base_path)),
                        'setups': setup_count
                    })
                
                # Detectar imports pesados
                if 'import torch' in content and 'import numpy' in content:
                    issues['slow_imports'].append({
                        'file': str(test_file.relative_to(self.base_path)),
                        'heavy_imports': ['torch', 'numpy']
                    })
        
        return issues
    
    def suggest_optimizations(self, issues: Dict) -> List[str]:
        """Generar sugerencias de optimización"""
        suggestions = []
        
        if issues['duplicate_setup']:
            suggestions.append(
                "💡 Optimización: Consolidar setUp() duplicados\n"
                "   - Mover código común a setUpClass()\n"
                "   - Usar fixtures compartidas"
            )
        
        if issues['slow_imports']:
            suggestions.append(
                "💡 Optimización: Lazy imports\n"
                "   - Importar módulos pesados solo cuando se necesiten\n"
                "   - Usar imports dentro de funciones en lugar de nivel de módulo"
            )
        
        if issues['missing_fixtures']:
            suggestions.append(
                "💡 Optimización: Usar fixtures de pytest\n"
                "   - Reemplazar setUp/tearDown con fixtures\n"
                "   - Aprovechar autouse para fixtures comunes"
            )
        
        return suggestions
    
    def find_slow_operations(self) -> List[Dict]:
        """Encontrar operaciones lentas en tests"""
        print("🔍 Buscando operaciones lentas...")
        
        slow_operations = []
        
        for test_file in (self.base_path / 'core').rglob('test_*.py'):
            with open(test_file, 'r') as f:
                lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    # Detectar operaciones potencialmente lentas
                    slow_patterns = [
                        (r'sleep\(', 'time.sleep() - considerar usar mocks'),
                        (r'requests\.', 'HTTP requests - usar mocks'),
                        (r'open\(', 'File I/O - considerar fixtures'),
                        (r'subprocess\.', 'Subprocess - considerar mocks'),
                    ]
                    
                    for pattern, description in slow_patterns:
                        if re.search(pattern, line):
                            slow_operations.append({
                                'file': str(test_file.relative_to(self.base_path)),
                                'line': i,
                                'operation': description,
                                'code': line.strip()
                            })
        
        return slow_operations
    
    def generate_optimization_report(self) -> Dict:
        """Generar reporte de optimización"""
        print("📊 Generando reporte de optimización...\n")
        
        structure_issues = self.analyze_test_structure()
        slow_ops = self.find_slow_operations()
        suggestions = self.suggest_optimizations(structure_issues)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'structure_issues': structure_issues,
            'slow_operations': slow_ops,
            'suggestions': suggestions,
            'summary': {
                'total_issues': (
                    len(structure_issues['duplicate_setup']) +
                    len(structure_issues['slow_imports']) +
                    len(slow_operations)
                ),
                'optimization_potential': 'high' if len(suggestions) > 3 else 'medium'
            }
        }
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        print("=" * 60)
        print("⚡ REPORTE DE OPTIMIZACIÓN")
        print("=" * 60)
        print()
        
        summary = report['summary']
        print(f"📊 Resumen:")
        print(f"   Total de issues: {summary['total_issues']}")
        print(f"   Potencial de optimización: {summary['optimization_potential']}")
        print()
        
        if report['suggestions']:
            print("💡 Sugerencias de Optimización:")
            for i, suggestion in enumerate(report['suggestions'], 1):
                print(f"\n{i}. {suggestion}")
            print()
        
        if report['slow_operations']:
            print(f"⏱️  Operaciones Lentas Detectadas: {len(report['slow_operations'])}")
            for op in report['slow_operations'][:10]:
                print(f"   {op['file']}:{op['line']} - {op['operation']}")
            print()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimizar tests')
    parser.add_argument('--output', type=Path,
                       help='Archivo de salida JSON')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    optimizer = TestOptimizer(args.base_path)
    report = optimizer.generate_optimization_report()
    optimizer.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Reporte guardado en: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

