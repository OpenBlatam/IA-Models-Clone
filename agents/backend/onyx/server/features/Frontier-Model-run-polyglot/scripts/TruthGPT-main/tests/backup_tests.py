#!/usr/bin/env python3
"""
Backup de Tests
Crea backups de resultados y configuración de tests
"""

import sys
import json
import shutil
import tarfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import zipfile


class TestBackup:
    """Sistema de backup para tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.backup_dir = base_path / 'backups'
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_results(self, output_name: Optional[str] = None) -> Path:
        """Crear backup de resultados"""
        print("💾 Creando backup de resultados...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = output_name or f"results_backup_{timestamp}"
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        # Archivos a respaldar
        files_to_backup = []
        
        # Resultados JSON
        for json_file in self.base_path.glob('*_stats.json'):
            files_to_backup.append(json_file)
        
        for json_file in self.base_path.glob('test-results/*.json'):
            files_to_backup.append(json_file)
        
        # Reportes HTML
        for html_file in self.base_path.glob('*.html'):
            if 'report' in html_file.name or 'dashboard' in html_file.name:
                files_to_backup.append(html_file)
        
        if not files_to_backup:
            print("   ⚠️  No se encontraron archivos para respaldar")
            return None
        
        # Crear archivo tar.gz
        with tarfile.open(backup_path, 'w:gz') as tar:
            for file_path in files_to_backup:
                if file_path.exists():
                    tar.add(file_path, arcname=file_path.name)
        
        print(f"   ✅ Backup creado: {backup_path}")
        print(f"   📦 Archivos respaldados: {len(files_to_backup)}")
        
        return backup_path
    
    def backup_config(self, output_name: Optional[str] = None) -> Path:
        """Crear backup de configuración"""
        print("💾 Creando backup de configuración...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = output_name or f"config_backup_{timestamp}"
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        # Archivos de configuración
        config_files = [
            'pytest.ini',
            'conftest.py',
            'requirements-test.txt',
            'Makefile',
            '.pre-commit-config.yaml'
        ]
        
        # Configuraciones
        config_dir = self.base_path / 'config'
        if config_dir.exists():
            for config_file in config_dir.glob('*.json'):
                if 'example' not in config_file.name:
                    config_files.append(f'config/{config_file.name}')
        
        files_to_backup = []
        for file_name in config_files:
            file_path = self.base_path / file_name
            if file_path.exists():
                files_to_backup.append(file_path)
        
        if not files_to_backup:
            print("   ⚠️  No se encontraron archivos de configuración")
            return None
        
        # Crear archivo zip
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_backup:
                zipf.write(file_path, file_path.name)
        
        print(f"   ✅ Backup creado: {backup_path}")
        print(f"   📦 Archivos respaldados: {len(files_to_backup)}")
        
        return backup_path
    
    def backup_all(self, output_name: Optional[str] = None) -> List[Path]:
        """Crear backup completo"""
        print("💾 Creando backup completo...\n")
        
        backups = []
        
        # Backup de resultados
        results_backup = self.backup_results()
        if results_backup:
            backups.append(results_backup)
        
        # Backup de configuración
        config_backup = self.backup_config()
        if config_backup:
            backups.append(config_backup)
        
        print(f"\n✅ Backups completados: {len(backups)}")
        
        return backups
    
    def list_backups(self) -> List[Path]:
        """Listar backups disponibles"""
        backups = sorted(self.backup_dir.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
        return backups
    
    def restore_backup(self, backup_path: Path, target_dir: Optional[Path] = None):
        """Restaurar desde backup"""
        print(f"📥 Restaurando desde: {backup_path}")
        
        target = Path(target_dir) if target_dir else self.base_path
        
        if backup_path.suffix == '.tar.gz':
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(target)
        elif backup_path.suffix == '.zip':
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(target)
        
        print(f"   ✅ Restaurado en: {target}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup de tests')
    parser.add_argument('--results', action='store_true',
                       help='Backup solo de resultados')
    parser.add_argument('--config', action='store_true',
                       help='Backup solo de configuración')
    parser.add_argument('--all', action='store_true',
                       help='Backup completo')
    parser.add_argument('--list', action='store_true',
                       help='Listar backups')
    parser.add_argument('--restore', type=Path,
                       help='Restaurar desde backup')
    parser.add_argument('--name', type=str,
                       help='Nombre personalizado para backup')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    backup = TestBackup(args.base_path)
    
    if args.list:
        backups = backup.list_backups()
        print(f"\n📦 Backups disponibles ({len(backups)}):")
        for b in backups[:10]:  # Mostrar últimos 10
            size = b.stat().st_size / 1024  # KB
            mtime = datetime.fromtimestamp(b.stat().st_mtime)
            print(f"   {b.name} ({size:.1f} KB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    elif args.restore:
        backup.restore_backup(args.restore)
    
    elif args.results:
        backup.backup_results(args.name)
    
    elif args.config:
        backup.backup_config(args.name)
    
    elif args.all:
        backup.backup_all(args.name)
    
    else:
        print("Especifica --results, --config, --all, --list o --restore")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

