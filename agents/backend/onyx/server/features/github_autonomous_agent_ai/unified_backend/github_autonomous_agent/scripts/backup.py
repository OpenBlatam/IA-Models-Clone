#!/usr/bin/env python3
"""
Script para crear backups de la base de datos y archivos importantes.
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import json

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_header(msg: str):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{msg}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def backup_database(backup_dir: Path) -> bool:
    """Crea backup de la base de datos."""
    try:
        from config.settings import settings
        
        db_url = settings.DATABASE_URL
        
        # SQLite
        if 'sqlite' in db_url.lower():
            db_path = db_url.split('///')[-1] if '///' in db_url else db_url.split(':///')[-1]
            db_file = Path(db_path)
            
            if db_file.exists():
                backup_file = backup_dir / f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                shutil.copy2(db_file, backup_file)
                print_success(f"Backup de SQLite creado: {backup_file.name}")
                return True
            else:
                print(f"⚠️  Archivo de base de datos no encontrado: {db_file}")
                return False
        
        # PostgreSQL
        elif 'postgresql' in db_url.lower():
            backup_file = backup_dir / f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            
            # Extraer información de conexión
            # Formato: postgresql+asyncpg://user:password@host:port/dbname
            import re
            match = re.search(r'postgresql[^:]*://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', db_url)
            
            if match:
                user, password, host, port, dbname = match.groups()
                
                # Usar pg_dump
                env = {'PGPASSWORD': password}
                result = subprocess.run(
                    ['pg_dump', '-h', host, '-p', port, '-U', user, '-d', dbname, '-f', str(backup_file)],
                    env=env,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print_success(f"Backup de PostgreSQL creado: {backup_file.name}")
                    return True
                else:
                    print(f"❌ Error en backup de PostgreSQL: {result.stderr}")
                    return False
            else:
                print("⚠️  No se pudo parsear URL de PostgreSQL")
                return False
        
        # MySQL
        elif 'mysql' in db_url.lower():
            print("⚠️  Backup de MySQL no implementado aún")
            return False
        
        else:
            print(f"⚠️  Tipo de base de datos no soportado: {db_url}")
            return False
            
    except Exception as e:
        print(f"❌ Error creando backup de base de datos: {e}")
        return False

def backup_files(backup_dir: Path) -> bool:
    """Crea backup de archivos importantes."""
    try:
        from config.settings import settings
        
        files_to_backup = []
        
        # .env (si existe)
        env_file = Path('.env')
        if env_file.exists():
            files_to_backup.append(env_file)
        
        # Storage (si se especifica)
        storage_path = Path(settings.STORAGE_PATH)
        if storage_path.exists():
            storage_backup = backup_dir / "storage"
            shutil.copytree(storage_path, storage_backup, dirs_exist_ok=True)
            print_success(f"Backup de storage creado: {storage_backup.name}")
        
        # Backup de archivos individuales
        for file_path in files_to_backup:
            backup_file = backup_dir / file_path.name
            shutil.copy2(file_path, backup_file)
            print_success(f"Backup de {file_path.name} creado")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creando backup de archivos: {e}")
        return False

def create_backup_manifest(backup_dir: Path) -> bool:
    """Crea manifest del backup."""
    try:
        import hashlib
        
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'backup_dir': str(backup_dir),
            'version': '1.0',
            'files': [],
            'total_size': 0,
            'checksums': {}
        }
        
        for file_path in backup_dir.rglob('*'):
            if file_path.is_file() and file_path.name != 'manifest.json':
                file_size = file_path.stat().st_size
                manifest['total_size'] += file_size
                
                # Calcular checksum
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                        manifest['checksums'][str(file_path.relative_to(backup_dir))] = file_hash
                except Exception:
                    pass
                
                manifest['files'].append({
                    'name': file_path.name,
                    'path': str(file_path.relative_to(backup_dir)),
                    'size': file_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        manifest_file = backup_dir / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Calcular checksum del manifest
        with open(manifest_file, 'rb') as f:
            manifest_hash = hashlib.sha256(f.read()).hexdigest()
            manifest['manifest_checksum'] = manifest_hash
        
        # Reescribir con checksum
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print_success(f"Manifest creado: {manifest_file.name} ({len(manifest['files'])} archivos, {manifest['total_size']} bytes)")
        return True
        
    except Exception as e:
        print(f"⚠️  Error creando manifest: {e}")
        return False

def main():
    """Función principal."""
    import os
    
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print_header("Backup - GitHub Autonomous Agent")
    
    # Crear directorio de backup
    backup_base = Path('backups')
    backup_base.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = backup_base / timestamp
    backup_dir.mkdir(exist_ok=True)
    
    print_info(f"Directorio de backup: {backup_dir}")
    
    # Backup de base de datos
    print_info("\nCreando backup de base de datos...")
    db_ok = backup_database(backup_dir)
    
    # Backup de archivos
    print_info("\nCreando backup de archivos...")
    files_ok = backup_files(backup_dir)
    
    # Crear manifest
    print_info("\nCreando manifest...")
    manifest_ok = create_backup_manifest(backup_dir)
    
    # Resumen
    print_header("Resumen de Backup")
    
    if db_ok and files_ok:
        print_success("✅ Backup completado exitosamente")
        print_info(f"Ubicación: {backup_dir}")
        print_info(f"Para restaurar, revisa el manifest.json en el directorio de backup")
        return 0
    else:
        print("⚠️  Backup completado con advertencias")
        return 1

if __name__ == "__main__":
    sys.exit(main())


