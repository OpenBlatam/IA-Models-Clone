#!/usr/bin/env python3
"""
Test Runner Utility
Ejecuta tests organizados por categoría
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional

# Mapeo de categorías a rutas
CATEGORIES = {
    'all': 'core',
    'core': 'core',
    'unit': 'core/unit',
    'integration': 'core/integration',
    'analyzers': 'analyzers',
    'cost': 'analyzers/cost',
    'performance': 'analyzers/performance',
    'quality': 'analyzers/quality',
    'security': 'analyzers/security',
    'compliance': 'analyzers/compliance',
    'coverage': 'analyzers/coverage',
    'dependency': 'analyzers/dependency',
    'trend': 'analyzers/trend',
    'flakiness': 'analyzers/flakiness',
    'regression': 'analyzers/regression',
    'optimization': 'analyzers/optimization',
    'systems': 'systems',
    'reporters': 'reporters',
    'exporters': 'exporters',
    'utilities': 'utilities',
}

def print_usage():
    """Imprime el uso del script"""
    print("""
Test Runner - Ejecuta tests organizados por categoría

Uso:
    python run_tests.py [categoria] [opciones]

Categorías disponibles:
    all              - Todos los tests
    core             - Tests core (unit + integration)
    unit             - Tests unitarios
    integration      - Tests de integración
    analyzers        - Todos los analizadores
    cost             - Análisis de costos
    performance      - Análisis de rendimiento
    quality          - Análisis de calidad
    security         - Análisis de seguridad
    compliance       - Verificación de cumplimiento
    coverage         - Análisis de cobertura
    dependency       - Análisis de dependencias
    trend            - Análisis de tendencias
    flakiness        - Análisis de flakiness
    regression       - Análisis de regresión
    optimization     - Análisis de optimización
    systems          - Sistemas y servicios
    reporters        - Módulos de reportes
    exporters        - Utilidades de exportación
    utilities        - Utilidades

Opciones:
    -v, --verbose    - Modo verbose
    -k PATTERN       - Ejecutar tests que coincidan con el patrón
    -m MARKER        - Ejecutar tests marcados
    --coverage       - Generar reporte de cobertura
    --html           - Generar reporte HTML
    -h, --help       - Mostrar esta ayuda

Ejemplos:
    python run_tests.py all
    python run_tests.py unit -v
    python run_tests.py analyzers --coverage
    python run_tests.py performance -k "test_performance"
    """)

def run_tests(category: str, extra_args: List[str] = None) -> int:
    """Ejecuta tests de una categoría"""
    if category not in CATEGORIES:
        print(f"Error: Categoría '{category}' no encontrada")
        print_usage()
        return 1
    
    test_path = Path(__file__).parent / CATEGORIES[category]
    
    if not test_path.exists():
        print(f"Error: Ruta de tests no encontrada: {test_path}")
        return 1
    
    # Construir comando pytest
    cmd = ['python', '-m', 'pytest', str(test_path)]
    
    if extra_args:
        cmd.extend(extra_args)
    else:
        cmd.append('-v')
    
    print(f"Ejecutando tests en: {test_path}")
    print(f"Comando: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nTests interrumpidos por el usuario")
        return 130
    except Exception as e:
        print(f"Error ejecutando tests: {e}")
        return 1

def main():
    """Función principal"""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print_usage()
        return 0
    
    category = sys.argv[1]
    extra_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    return run_tests(category, extra_args)

if __name__ == '__main__':
    sys.exit(main())

