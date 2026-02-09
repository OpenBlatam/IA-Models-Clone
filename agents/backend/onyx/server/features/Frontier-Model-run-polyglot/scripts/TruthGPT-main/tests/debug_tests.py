#!/usr/bin/env python3
"""
Herramienta de Debugging para Tests
Ayuda a identificar y resolver problemas en tests
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import re


class TestDebugger:
    """Debugger de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    def find_failing_tests(self, category: str = 'all') -> List[Dict]:
        """Encontrar tests que están fallando"""
        print(f"🔍 Buscando tests fallidos en categoría: {category}...")
        
        result = subprocess.run(
            ['python', 'run_tests.py', category, '-v', '--tb=short'],
            cwd=self.base_path,
            capture_output=True,
            text=True
        )
        
        failing_tests = []
        current_test = None
        
        for line in result.stdout.splitlines():
            # Detectar test fallido
            if 'FAILED' in line or 'ERROR' in line:
                # Extraer nombre del test
                match = re.search(r'test_\w+', line)
                if match:
                    current_test = {
                        'name': match.group(),
                        'status': 'FAILED',
                        'error': line
                    }
            
            # Capturar traceback
            if current_test and ('Traceback' in line or 'AssertionError' in line or 'Error:' in line):
                if 'error_details' not in current_test:
                    current_test['error_details'] = []
                current_test['error_details'].append(line)
            
            # Finalizar test
            if current_test and line.strip() == '' and 'error_details' in current_test:
                failing_tests.append(current_test)
                current_test = None
        
        return failing_tests
    
    def analyze_error_patterns(self, failing_tests: List[Dict]) -> Dict:
        """Analizar patrones de errores"""
        patterns = {
            'import_errors': [],
            'assertion_errors': [],
            'timeout_errors': [],
            'other_errors': []
        }
        
        for test in failing_tests:
            error_text = ' '.join(test.get('error_details', []))
            
            if 'ImportError' in error_text or 'ModuleNotFoundError' in error_text:
                patterns['import_errors'].append(test)
            elif 'AssertionError' in error_text:
                patterns['assertion_errors'].append(test)
            elif 'Timeout' in error_text or 'timeout' in error_text.lower():
                patterns['timeout_errors'].append(test)
            else:
                patterns['other_errors'].append(test)
        
        return patterns
    
    def suggest_fixes(self, patterns: Dict) -> List[str]:
        """Sugerir fixes basados en patrones"""
        suggestions = []
        
        if patterns['import_errors']:
            suggestions.append(
                "🔧 Import Errors detectados:\n"
                "   - Verificar que todas las dependencias estén instaladas\n"
                "   - Ejecutar: pip install -r requirements-test.txt\n"
                "   - Verificar paths de imports en los archivos"
            )
        
        if patterns['assertion_errors']:
            suggestions.append(
                "🔧 Assertion Errors detectados:\n"
                "   - Revisar las aserciones en los tests\n"
                "   - Verificar datos de prueba\n"
                "   - Considerar usar debugger para inspeccionar valores"
            )
        
        if patterns['timeout_errors']:
            suggestions.append(
                "🔧 Timeout Errors detectados:\n"
                "   - Los tests pueden estar tardando demasiado\n"
                "   - Considerar aumentar timeout en pytest.ini\n"
                "   - Optimizar código de tests lento"
            )
        
        return suggestions
    
    def debug_test(self, test_name: str, verbose: bool = True) -> Dict:
        """Debuggear un test específico"""
        print(f"🐛 Debuggeando test: {test_name}...")
        
        # Ejecutar con máximo detalle
        result = subprocess.run(
            ['pytest', f'core/**/{test_name}', '-vv', '--tb=long', '--pdb'],
            cwd=self.base_path,
            capture_output=True,
            text=True
        )
        
        debug_info = {
            'test_name': test_name,
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
        if verbose:
            print("\n📋 Output:")
            print(result.stdout)
            if result.stderr:
                print("\n❌ Errors:")
                print(result.stderr)
        
        return debug_info
    
    def check_test_dependencies(self) -> Dict:
        """Verificar dependencias de tests"""
        print("🔍 Verificando dependencias...")
        
        issues = []
        
        # Verificar archivos requeridos
        required_files = ['conftest.py', 'pytest.ini', 'requirements-test.txt']
        for file in required_files:
            if not (self.base_path / file).exists():
                issues.append(f"Archivo faltante: {file}")
        
        # Verificar imports comunes
        try:
            import pytest
            import coverage
        except ImportError as e:
            issues.append(f"Dependencia faltante: {e.name}")
        
        return {
            'status': 'ok' if not issues else 'issues_found',
            'issues': issues
        }
    
    def generate_debug_report(self, category: str = 'all') -> Dict:
        """Generar reporte completo de debugging"""
        print("🔍 Generando reporte de debugging...\n")
        
        # Encontrar tests fallidos
        failing_tests = self.find_failing_tests(category)
        
        # Analizar patrones
        patterns = self.analyze_error_patterns(failing_tests)
        
        # Sugerir fixes
        suggestions = self.suggest_fixes(patterns)
        
        # Verificar dependencias
        dependencies = self.check_test_dependencies()
        
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'category': category,
            'failing_tests_count': len(failing_tests),
            'failing_tests': failing_tests,
            'error_patterns': {
                k: len(v) for k, v in patterns.items()
            },
            'suggestions': suggestions,
            'dependencies': dependencies
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        print("=" * 60)
        print("🐛 REPORTE DE DEBUGGING")
        print("=" * 60)
        print()
        
        print(f"📊 Resumen:")
        print(f"   Categoría: {report['category']}")
        print(f"   Tests fallidos: {report['failing_tests_count']}")
        print()
        
        if report['error_patterns']:
            print("📋 Patrones de Error:")
            for pattern, count in report['error_patterns'].items():
                if count > 0:
                    print(f"   {pattern}: {count}")
            print()
        
        if report['suggestions']:
            print("💡 Sugerencias:")
            for suggestion in report['suggestions']:
                print(suggestion)
                print()
        
        if report['failing_tests']:
            print("❌ Tests Fallidos:")
            for test in report['failing_tests'][:10]:  # Primeros 10
                print(f"   - {test['name']}")
                if 'error_details' in test:
                    print(f"     Error: {test['error_details'][0] if test['error_details'] else 'N/A'}")
            print()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debuggear tests')
    parser.add_argument('--category', default='all',
                       help='Categoría de tests a debuggear')
    parser.add_argument('--test', help='Test específico a debuggear')
    parser.add_argument('--output', type=Path,
                       help='Archivo de salida JSON')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    debugger = TestDebugger(args.base_path)
    
    if args.test:
        # Debuggear test específico
        result = debugger.debug_test(args.test)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    else:
        # Generar reporte completo
        report = debugger.generate_debug_report(args.category)
        debugger.print_report(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Reporte guardado en: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

