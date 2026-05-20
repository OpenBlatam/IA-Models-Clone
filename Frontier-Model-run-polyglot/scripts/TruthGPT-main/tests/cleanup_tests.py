#!/usr/bin/env python3
"""
Limpieza de Tests
Limpia archivos temporales, cachés y resultados antiguos
"""

import sys
import shutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
import argparse


class TestCleanup:
    """Sistema de limpieza para tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.cleaned = {
            'files': 0,
            'directories': 0,
            'size_freed': 0
        }
    
    def clean_cache(self) -> int:
        """Limpiar cachés"""
        print("🧹 Limpiando cachés...")
        
        cache_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '.pytest_cache',
            '.coverage',
            'htmlcov',
            '.mypy_cache',
            '.ruff_cache'
        ]
        
        cleaned = 0
        
        # Limpiar __pycache__
        for pycache in self.base_path.rglob('__pycache__'):
            size = self._get_dir_size(pycache)
            shutil.rmtree(pycache)
            self.cleaned['directories'] += 1
            self.cleaned['size_freed'] += size
            cleaned += 1
        
        # Limpiar archivos .pyc
        for pyc in self.base_path.rglob('*.pyc'):
            size = pyc.stat().st_size
            pyc.unlink()
            self.cleaned['files'] += 1
            self.cleaned['size_freed'] += size
            cleaned += 1
        
        # Limpiar .pytest_cache
        pytest_cache = self.base_path / '.pytest_cache'
        if pytest_cache.exists():
            size = self._get_dir_size(pytest_cache)
            shutil.rmtree(pytest_cache)
            self.cleaned['directories'] += 1
            self.cleaned['size_freed'] += size
            cleaned += 1
        
        print(f"   ✅ Limpiados {cleaned} elementos de caché")
        return cleaned
    
    def clean_old_results(self, days: int = 7) -> int:
        """Limpiar resultados antiguos"""
        print(f"🧹 Limpiando resultados antiguos (>{days} días)...")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned = 0
        
        # Limpiar archivos de resultados antiguos
        result_patterns = [
            '*_stats.json',
            'results_*.json',
            'report_*.html',
            'dashboard.html',
            'test-report.html'
        ]
        
        for pattern in result_patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff_date:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        self.cleaned['files'] += 1
                        self.cleaned['size_freed'] += size
                        cleaned += 1
        
        # Limpiar directorio test-results antiguo
        test_results_dir = self.base_path / 'test-results'
        if test_results_dir.exists():
            for item in test_results_dir.iterdir():
                if item.is_file():
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff_date:
                        size = item.stat().st_size
                        item.unlink()
                        self.cleaned['files'] += 1
                        self.cleaned['size_freed'] += size
                        cleaned += 1
        
        print(f"   ✅ Limpiados {cleaned} archivos antiguos")
        return cleaned
    
    def clean_logs(self) -> int:
        """Limpiar archivos de log"""
        print("🧹 Limpiando logs...")
        
        cleaned = 0
        
        for log_file in self.base_path.rglob('*.log'):
            size = log_file.stat().st_size
            log_file.unlink()
            self.cleaned['files'] += 1
            self.cleaned['size_freed'] += size
            cleaned += 1
        
        print(f"   ✅ Limpiados {cleaned} archivos de log")
        return cleaned
    
    def clean_temporary(self) -> int:
        """Limpiar archivos temporales"""
        print("🧹 Limpiando archivos temporales...")
        
        temp_patterns = ['*.tmp', '*.temp', '*.swp', '*.swo', '*~']
        cleaned = 0
        
        for pattern in temp_patterns:
            for temp_file in self.base_path.rglob(pattern):
                if temp_file.is_file():
                    size = temp_file.stat().st_size
                    temp_file.unlink()
                    self.cleaned['files'] += 1
                    self.cleaned['size_freed'] += size
                    cleaned += 1
        
        print(f"   ✅ Limpiados {cleaned} archivos temporales")
        return cleaned
    
    def _get_dir_size(self, directory: Path) -> int:
        """Calcular tamaño de directorio"""
        total = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except Exception:
            pass
        return total
    
    def clean_all(self, days: int = 7) -> Dict:
        """Limpieza completa"""
        print("🧹 Iniciando limpieza completa...\n")
        
        self.clean_cache()
        self.clean_old_results(days)
        self.clean_logs()
        self.clean_temporary()
        
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE LIMPIEZA")
        print("=" * 60)
        print(f"Archivos eliminados: {self.cleaned['files']}")
        print(f"Directorios eliminados: {self.cleaned['directories']}")
        print(f"Espacio liberado: {self.cleaned['size_freed'] / 1024 / 1024:.2f} MB")
        print("=" * 60)
        
        return self.cleaned


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Limpiar archivos de tests')
    parser.add_argument('--cache', action='store_true',
                       help='Limpiar solo cachés')
    parser.add_argument('--results', type=int, metavar='DAYS',
                       help='Limpiar resultados más antiguos que N días')
    parser.add_argument('--logs', action='store_true',
                       help='Limpiar solo logs')
    parser.add_argument('--temp', action='store_true',
                       help='Limpiar solo archivos temporales')
    parser.add_argument('--all', action='store_true',
                       help='Limpieza completa')
    parser.add_argument('--days', type=int, default=7,
                       help='Días para considerar resultados antiguos')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    cleanup = TestCleanup(args.base_path)
    
    if args.cache:
        cleanup.clean_cache()
    elif args.results:
        cleanup.clean_old_results(args.results)
    elif args.logs:
        cleanup.clean_logs()
    elif args.temp:
        cleanup.clean_temporary()
    elif args.all:
        cleanup.clean_all(args.days)
    else:
        print("Especifica --cache, --results, --logs, --temp o --all")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

