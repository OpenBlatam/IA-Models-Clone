#!/usr/bin/env python3
"""
Test Discovery
Descubre y cataloga todos los tests disponibles
"""

import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class TestDiscovery:
    """Descubridor de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.discovered = {
            'tests': [],
            'test_classes': [],
            'test_functions': [],
            'fixtures': [],
            'markers': defaultdict(list)
        }
    
    def discover_tests(self) -> Dict:
        """Descubrir todos los tests"""
        print("🔍 Descubriendo tests...\n")
        
        for test_file in self.base_path.rglob('test_*.py'):
            self._analyze_test_file(test_file)
        
        return {
            'total_test_files': len(self.discovered['tests']),
            'total_test_classes': len(self.discovered['test_classes']),
            'total_test_functions': len(self.discovered['test_functions']),
            'total_fixtures': len(self.discovered['fixtures']),
            'tests': self.discovered['tests'],
            'markers': dict(self.discovered['markers'])
        }
    
    def _analyze_test_file(self, file_path: Path):
        """Analizar archivo de test"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            file_info = {
                'path': str(file_path.relative_to(self.base_path)),
                'classes': [],
                'functions': [],
                'fixtures': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith('Test'):
                        class_info = {
                            'name': node.name,
                            'methods': []
                        }
                        
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                if item.name.startswith('test_'):
                                    class_info['methods'].append(item.name)
                                    self.discovered['test_functions'].append({
                                        'name': item.name,
                                        'class': node.name,
                                        'file': file_info['path']
                                    })
                        
                        if class_info['methods']:
                            file_info['classes'].append(class_info)
                            self.discovered['test_classes'].append({
                                'name': node.name,
                                'file': file_info['path'],
                                'methods_count': len(class_info['methods'])
                            })
                
                elif isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        file_info['functions'].append(node.name)
                        self.discovered['test_functions'].append({
                            'name': node.name,
                            'class': None,
                            'file': file_info['path']
                        })
                    
                    # Detectar fixtures
                    if 'fixture' in node.name.lower() or any(
                        isinstance(d, ast.Name) and d.id == 'pytest.fixture'
                        for d in node.decorator_list
                    ):
                        file_info['fixtures'].append(node.name)
                        self.discovered['fixtures'].append({
                            'name': node.name,
                            'file': file_info['path']
                        })
            
            if file_info['classes'] or file_info['functions']:
                self.discovered['tests'].append(file_info)
        
        except Exception as e:
            print(f"   ⚠️  Error analizando {file_path}: {e}")
    
    def generate_catalog(self) -> Dict:
        """Generar catálogo completo"""
        discovery = self.discover_tests()
        
        catalog = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'summary': {
                'test_files': discovery['total_test_files'],
                'test_classes': discovery['total_test_classes'],
                'test_functions': discovery['total_test_functions'],
                'fixtures': discovery['total_fixtures']
            },
            'by_category': self._categorize_tests(),
            'tests': discovery['tests']
        }
        
        return catalog
    
    def _categorize_tests(self) -> Dict:
        """Categorizar tests por ubicación"""
        categories = defaultdict(int)
        
        for test in self.discovered['tests']:
            path = test['path']
            if 'unit' in path:
                categories['unit'] += 1
            elif 'integration' in path:
                categories['integration'] += 1
            else:
                categories['other'] += 1
        
        return dict(categories)
    
    def print_catalog(self, catalog: Dict):
        """Imprimir catálogo"""
        print("=" * 60)
        print("📚 CATÁLOGO DE TESTS")
        print("=" * 60)
        print()
        
        summary = catalog['summary']
        print("📊 Resumen:")
        print(f"   Archivos de test: {summary['test_files']}")
        print(f"   Clases de test: {summary['test_classes']}")
        print(f"   Funciones de test: {summary['test_functions']}")
        print(f"   Fixtures: {summary['fixtures']}")
        print()
        
        print("📁 Por categoría:")
        for category, count in catalog['by_category'].items():
            print(f"   {category}: {count}")
        print()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Descubrir tests')
    parser.add_argument('--output', type=Path,
                       help='Archivo JSON de salida')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    discovery = TestDiscovery(args.base_path)
    catalog = discovery.generate_catalog()
    discovery.print_catalog(catalog)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(catalog, f, indent=2)
        print(f"✅ Catálogo guardado en: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

