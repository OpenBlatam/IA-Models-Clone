#!/usr/bin/env python3
"""
Migrador de Tests
Ayuda a migrar tests de estructura antigua a nueva
"""

import sys
import shutil
from pathlib import Path
from typing import Dict, List
import re


class TestMigrator:
    """Migrador de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.migrations = []
    
    def analyze_imports(self, file_path: Path) -> List[str]:
        """Analizar imports en un archivo"""
        imports = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Buscar imports relativos antiguos
                old_patterns = [
                    r'from\s+tests\.',
                    r'import\s+tests\.',
                    r'from\s+\.\.tests\.',
                ]
                
                for pattern in old_patterns:
                    matches = re.findall(pattern, content)
                    imports.extend(matches)
        except Exception:
            pass
        
        return imports
    
    def suggest_migration(self, file_path: Path) -> Dict:
        """Sugerir migración para un archivo"""
        imports = self.analyze_imports(file_path)
        
        suggestions = []
        
        if imports:
            suggestions.append({
                'file': str(file_path.relative_to(self.base_path)),
                'issue': 'Imports antiguos detectados',
                'fix': 'Actualizar imports a nueva estructura'
            })
        
        return {
            'file': str(file_path.relative_to(self.base_path)),
            'suggestions': suggestions
        }
    
    def migrate_imports(self, file_path: Path, dry_run: bool = True) -> bool:
        """Migrar imports en un archivo"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Reemplazos de imports
            replacements = [
                (r'from\s+tests\.core\.', 'from core.'),
                (r'from\s+tests\.analyzers\.', 'from analyzers.'),
                (r'from\s+tests\.systems\.', 'from systems.'),
                (r'import\s+tests\.', 'import '),
            ]
            
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                if not dry_run:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    print(f"   ✅ Migrado: {file_path.name}")
                else:
                    print(f"   📝 Necesita migración: {file_path.name}")
                return True
            
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def scan_for_migrations(self) -> List[Dict]:
        """Escanear archivos que necesitan migración"""
        print("🔍 Escaneando archivos para migración...\n")
        
        files_to_migrate = []
        
        for py_file in self.base_path.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
            
            migration = self.suggest_migration(py_file)
            if migration['suggestions']:
                files_to_migrate.append(migration)
        
        return files_to_migrate
    
    def migrate_all(self, dry_run: bool = True) -> Dict:
        """Migrar todos los archivos"""
        print(f"{'🔍 [DRY RUN]' if dry_run else '🚀'} Migrando archivos...\n")
        
        migrated_count = 0
        
        for py_file in self.base_path.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
            
            if self.migrate_imports(py_file, dry_run):
                migrated_count += 1
        
        return {
            'migrated_files': migrated_count,
            'dry_run': dry_run
        }


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrar tests a nueva estructura')
    parser.add_argument('--scan', action='store_true',
                       help='Solo escanear archivos')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Simular migración sin cambios')
    parser.add_argument('--apply', action='store_true',
                       help='Aplicar migración (sobrescribe --dry-run)')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    migrator = TestMigrator(args.base_path)
    
    if args.scan:
        migrations = migrator.scan_for_migrations()
        print(f"\n📋 Archivos que necesitan migración: {len(migrations)}")
        for mig in migrations[:10]:
            print(f"   - {mig['file']}")
    else:
        dry_run = not args.apply
        result = migrator.migrate_all(dry_run)
        
        if dry_run:
            print(f"\n💡 Para aplicar migraciones, usa: --apply")
        else:
            print(f"\n✅ Migración completada: {result['migrated_files']} archivos")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

