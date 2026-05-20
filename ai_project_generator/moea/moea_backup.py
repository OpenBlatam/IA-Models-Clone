"""
MOEA Backup System - Sistema de backup automático
==================================================
Sistema completo de backup para proyectos y configuraciones MOEA
"""
import shutil
import json
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import hashlib


class MOEABackupSystem:
    """Sistema de backup MOEA"""
    
    def __init__(self, backup_dir: str = "moea_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.backup_dir / "backup_manifest.json"
        self.manifests = self._load_manifests()
    
    def _load_manifests(self) -> List[Dict]:
        """Cargar manifiestos de backups"""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_manifests(self):
        """Guardar manifiestos"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifests, f, indent=2)
    
    def create_backup(
        self,
        project_dir: str,
        backup_name: Optional[str] = None,
        format: str = "zip",
        include_results: bool = True,
        include_config: bool = True
    ) -> Optional[str]:
        """Crear backup de un proyecto"""
        project_path = Path(project_dir)
        
        if not project_path.exists():
            print(f"❌ Directorio no encontrado: {project_dir}")
            return None
        
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{project_path.name}_{timestamp}"
        
        backup_file = self.backup_dir / f"{backup_name}.{format}"
        
        print(f"📦 Creando backup: {backup_name}")
        
        try:
            if format == "zip":
                self._create_zip_backup(project_path, backup_file, include_results, include_config)
            elif format in ["tar", "tar.gz"]:
                compress = format == "tar.gz"
                self._create_tar_backup(project_path, backup_file, include_results, include_config, compress)
            else:
                print(f"❌ Formato no soportado: {format}")
                return None
            
            # Calcular hash
            file_hash = self._calculate_hash(backup_file)
            
            # Crear manifiesto
            manifest = {
                "name": backup_name,
                "file": str(backup_file),
                "format": format,
                "project_dir": str(project_path),
                "created_at": datetime.now().isoformat(),
                "size": backup_file.stat().st_size,
                "hash": file_hash,
                "includes_results": include_results,
                "includes_config": include_config
            }
            
            self.manifests.append(manifest)
            self._save_manifests()
            
            print(f"✅ Backup creado: {backup_file}")
            print(f"   Tamaño: {self._format_size(backup_file.stat().st_size)}")
            print(f"   Hash: {file_hash[:16]}...")
            
            return str(backup_file)
            
        except Exception as e:
            print(f"❌ Error creando backup: {e}")
            return None
    
    def _create_zip_backup(
        self,
        project_path: Path,
        backup_file: Path,
        include_results: bool,
        include_config: bool
    ):
        """Crear backup ZIP"""
        with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in project_path.rglob('*'):
                if file_path.is_file():
                    # Filtrar
                    if not include_results and 'results' in str(file_path):
                        continue
                    if not include_config and file_path.name in ['.env', 'config.json']:
                        continue
                    if file_path.name.startswith('.') and file_path.name != '.env':
                        continue
                    
                    arcname = file_path.relative_to(project_path.parent)
                    zipf.write(file_path, arcname)
    
    def _create_tar_backup(
        self,
        project_path: Path,
        backup_file: Path,
        include_results: bool,
        include_config: bool,
        compress: bool
    ):
        """Crear backup TAR"""
        mode = 'w:gz' if compress else 'w'
        with tarfile.open(backup_file, mode) as tarf:
            for file_path in project_path.rglob('*'):
                if file_path.is_file():
                    # Filtrar
                    if not include_results and 'results' in str(file_path):
                        continue
                    if not include_config and file_path.name in ['.env', 'config.json']:
                        continue
                    if file_path.name.startswith('.') and file_path.name != '.env':
                        continue
                    
                    arcname = file_path.relative_to(project_path.parent)
                    tarf.add(file_path, arcname=arcname)
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Calcular hash SHA256"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _format_size(self, bytes_size: int) -> str:
        """Formatear tamaño"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"
    
    def list_backups(self) -> List[Dict]:
        """Listar todos los backups"""
        return self.manifests
    
    def restore_backup(self, backup_name: str, target_dir: Optional[str] = None) -> bool:
        """Restaurar backup"""
        manifest = next((m for m in self.manifests if m['name'] == backup_name), None)
        
        if not manifest:
            print(f"❌ Backup no encontrado: {backup_name}")
            return False
        
        backup_file = Path(manifest['file'])
        if not backup_file.exists():
            print(f"❌ Archivo de backup no encontrado: {backup_file}")
            return False
        
        if not target_dir:
            target_dir = manifest['project_dir']
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Restaurando backup: {backup_name}")
        print(f"   Destino: {target_dir}")
        
        try:
            format = manifest['format']
            if format == "zip":
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    zipf.extractall(target_path.parent)
            elif format in ["tar", "tar.gz"]:
                with tarfile.open(backup_file, 'r:*') as tarf:
                    tarf.extractall(target_path.parent)
            
            print(f"✅ Backup restaurado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error restaurando backup: {e}")
            return False
    
    def delete_backup(self, backup_name: str) -> bool:
        """Eliminar backup"""
        manifest = next((m for m in self.manifests if m['name'] == backup_name), None)
        
        if not manifest:
            print(f"❌ Backup no encontrado: {backup_name}")
            return False
        
        backup_file = Path(manifest['file'])
        if backup_file.exists():
            backup_file.unlink()
        
        self.manifests.remove(manifest)
        self._save_manifests()
        
        print(f"✅ Backup eliminado: {backup_name}")
        return True
    
    def verify_backup(self, backup_name: str) -> bool:
        """Verificar integridad del backup"""
        manifest = next((m for m in self.manifests if m['name'] == backup_name), None)
        
        if not manifest:
            print(f"❌ Backup no encontrado: {backup_name}")
            return False
        
        backup_file = Path(manifest['file'])
        if not backup_file.exists():
            print(f"❌ Archivo no encontrado: {backup_file}")
            return False
        
        print(f"🔍 Verificando backup: {backup_name}")
        
        # Verificar tamaño
        actual_size = backup_file.stat().st_size
        expected_size = manifest['size']
        
        if actual_size != expected_size:
            print(f"❌ Tamaño no coincide: esperado {expected_size}, actual {actual_size}")
            return False
        
        # Verificar hash
        actual_hash = self._calculate_hash(backup_file)
        expected_hash = manifest['hash']
        
        if actual_hash != expected_hash:
            print(f"❌ Hash no coincide: el backup puede estar corrupto")
            return False
        
        print(f"✅ Backup verificado: integridad OK")
        return True


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Backup System")
    subparsers = parser.add_subparsers(dest='command', help='Comandos')
    
    # Comando create
    create_parser = subparsers.add_parser('create', help='Crear backup')
    create_parser.add_argument('project_dir', help='Directorio del proyecto')
    create_parser.add_argument('--name', help='Nombre del backup')
    create_parser.add_argument('--format', choices=['zip', 'tar', 'tar.gz'], default='zip')
    create_parser.add_argument('--no-results', action='store_true')
    create_parser.add_argument('--no-config', action='store_true')
    
    # Comando list
    list_parser = subparsers.add_parser('list', help='Listar backups')
    
    # Comando restore
    restore_parser = subparsers.add_parser('restore', help='Restaurar backup')
    restore_parser.add_argument('backup_name', help='Nombre del backup')
    restore_parser.add_argument('--target', help='Directorio destino')
    
    # Comando delete
    delete_parser = subparsers.add_parser('delete', help='Eliminar backup')
    delete_parser.add_argument('backup_name', help='Nombre del backup')
    
    # Comando verify
    verify_parser = subparsers.add_parser('verify', help='Verificar backup')
    verify_parser.add_argument('backup_name', help='Nombre del backup')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    backup_system = MOEABackupSystem()
    
    if args.command == 'create':
        backup_system.create_backup(
            args.project_dir,
            backup_name=args.name,
            format=args.format,
            include_results=not args.no_results,
            include_config=not args.no_config
        )
    elif args.command == 'list':
        backups = backup_system.list_backups()
        print(f"\n📦 Backups disponibles: {len(backups)}\n")
        for backup in backups:
            print(f"  {backup['name']}")
            print(f"    Archivo: {backup['file']}")
            print(f"    Creado: {backup['created_at']}")
            print(f"    Tamaño: {backup['size']} bytes")
            print()
    elif args.command == 'restore':
        backup_system.restore_backup(args.backup_name, args.target)
    elif args.command == 'delete':
        backup_system.delete_backup(args.backup_name)
    elif args.command == 'verify':
        backup_system.verify_backup(args.backup_name)


if __name__ == "__main__":
    main()

