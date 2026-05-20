#!/usr/bin/env python3
"""
Análisis de Tests
Analiza la suite de tests y genera reportes detallados
"""

import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
import re


class TestAnalyzer:
    """Analizador de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.stats = {
            'total_files': 0,
            'total_tests': 0,
            'total_classes': 0,
            'total_functions': 0,
            'test_files': [],
            'analyzer_files': [],
            'system_files': [],
            'categories': defaultdict(int),
            'imports': defaultdict(int),
            'decorators': defaultdict(int),
        }
    
    def analyze_file(self, file_path: Path) -> Dict:
        """Analizar un archivo Python"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            file_stats = {
                'path': str(file_path.relative_to(self.base_path)),
                'lines': len(content.splitlines()),
                'classes': [],
                'test_functions': [],
                'imports': [],
                'decorators': [],
            }
            
            # Analizar AST
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith('Test'):
                        file_stats['classes'].append(node.name)
                        self.stats['total_classes'] += 1
                
                elif isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        file_stats['test_functions'].append(node.name)
                        self.stats['total_tests'] += 1
                        
                        # Analizar decoradores
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Name):
                                self.stats['decorators'][decorator.id] += 1
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        file_stats['imports'].append(alias.name)
                        self.stats['imports'][alias.name.split('.')[0]] += 1
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        file_stats['imports'].append(node.module)
                        self.stats['imports'][node.module.split('.')[0]] += 1
            
            file_stats['total_functions'] = len(file_stats['test_functions'])
            self.stats['total_functions'] += file_stats['total_functions']
            
            return file_stats
            
        except Exception as e:
            print(f"Error analizando {file_path}: {e}")
            return {}
    
    def analyze_directory(self, directory: Path, category: str = None):
        """Analizar un directorio completo"""
        for py_file in directory.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
            
            self.stats['total_files'] += 1
            
            file_stats = self.analyze_file(py_file)
            
            if not file_stats:
                continue
            
            # Categorizar archivo
            rel_path = str(py_file.relative_to(self.base_path))
            if 'test_' in py_file.name:
                self.stats['test_files'].append(file_stats)
            elif 'analyzer' in rel_path.lower():
                self.stats['analyzer_files'].append(file_stats)
            elif 'system' in rel_path.lower():
                self.stats['system_files'].append(file_stats)
            
            # Categoría
            if category:
                self.stats['categories'][category] += 1
            else:
                # Detectar categoría desde path
                parts = rel_path.split('/')
                if len(parts) > 1:
                    self.stats['categories'][parts[0]] += 1
    
    def generate_report(self) -> Dict:
        """Generar reporte completo"""
        # Analizar todas las carpetas
        self.analyze_directory(self.base_path / 'core', 'core')
        self.analyze_directory(self.base_path / 'analyzers', 'analyzers')
        self.analyze_directory(self.base_path / 'systems', 'systems')
        self.analyze_directory(self.base_path / 'reporters', 'reporters')
        self.analyze_directory(self.base_path / 'exporters', 'exporters')
        self.analyze_directory(self.base_path / 'utilities', 'utilities')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': self.stats['total_files'],
                'total_tests': self.stats['total_tests'],
                'total_classes': self.stats['total_classes'],
                'total_functions': self.stats['total_functions'],
                'test_files_count': len(self.stats['test_files']),
                'analyzer_files_count': len(self.stats['analyzer_files']),
                'system_files_count': len(self.stats['system_files']),
            },
            'categories': dict(self.stats['categories']),
            'top_imports': dict(sorted(
                self.stats['imports'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'decorators_usage': dict(self.stats['decorators']),
            'test_files': [
                {
                    'path': f['path'],
                    'tests': len(f['test_functions']),
                    'classes': len(f['classes']),
                    'lines': f['lines']
                }
                for f in self.stats['test_files'][:20]  # Top 20
            ],
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        print("=" * 60)
        print("📊 REPORTE DE ANÁLISIS DE TESTS")
        print("=" * 60)
        print()
        
        summary = report['summary']
        print("📈 RESUMEN:")
        print(f"  Total archivos: {summary['total_files']}")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Total clases: {summary['total_classes']}")
        print(f"  Total funciones: {summary['total_functions']}")
        print(f"  Archivos de test: {summary['test_files_count']}")
        print(f"  Archivos de analizadores: {summary['analyzer_files_count']}")
        print(f"  Archivos de sistemas: {summary['system_files_count']}")
        print()
        
        print("📁 CATEGORÍAS:")
        for category, count in sorted(report['categories'].items()):
            print(f"  {category}: {count}")
        print()
        
        print("📦 TOP IMPORTS:")
        for imp, count in list(report['top_imports'].items())[:10]:
            print(f"  {imp}: {count}")
        print()
        
        print("🎨 DECORADORES USADOS:")
        for decorator, count in report['decorators_usage'].items():
            print(f"  @{decorator}: {count}")
        print()
        
        print("📝 TOP ARCHIVOS DE TEST:")
        for test_file in report['test_files'][:10]:
            print(f"  {test_file['path']}")
            print(f"    Tests: {test_file['tests']}, Clases: {test_file['classes']}, Líneas: {test_file['lines']}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar suite de tests')
    parser.add_argument('--output', type=Path, help='Archivo de salida JSON')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    analyzer = TestAnalyzer(args.base_path)
    report = analyzer.generate_report()
    
    analyzer.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Reporte guardado en: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

