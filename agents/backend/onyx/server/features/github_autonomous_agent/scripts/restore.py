#!/usr/bin/env python3
"""
Script para restaurar backups de la base de datos y archivos importantes.
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import json
import hashlib

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    RESET = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_header(msg: str):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{msg}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def verify_backup_integrity(backup_dir: Path) -> bool:
    """Verificar integridad del backup."""
    manifest_file = backup_dir / 'manifest.json'
    
    if not manifest_file.exists():
        print_error("Manifest no encontrado en el backup")
        return False
    
    try:
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        print_info("Verificando integridad de archivos...")
        all_valid = True
        
        for file_info in manifest.get('files', []):
            file_path = backup_dir / file_info['path']
            
            if not file_path.exists():
                print_error(f"Archivo faltante: {file_info['path']}")
                all_valid = False
                continue
            
            # Verificar tamaño
            actual_size = file_path.stat().st_size
            if actual_size != file_info['size']:
                print_error(f"Tamaño incorrecto: {file_info['path']}")
                all_valid = False
                continue
            
            # Verificar checksum si está disponible
            if 'checksums' in manifest:
                expected_hash = manifest['checksums'].get(file_info['path'])
                if expected_hash:
                    with open(file_path, 'rb') as f:
                        actual_hash = hashlib.sha256(f.read()).hexdigest()
                    if actual_hash != expected_hash:
                        print_error(f"Checksum incorrecto: {file_info['path']}")
                        all_valid = False
                        continue
        
        if all_valid:
            print_success("Integridad del backup verificada")
        else:
            print_warning("Algunos archivos tienen problemas de integridad")
        
        return all_valid
        
    except Exception as e:
        print_error(f"Error verificando integridad: {e}")
        return False

def restore_database(backup_dir: Path, dry_run: bool = False) -> bool:
    """Restaurar base de datos desde backup."""
    try:
        from config.settings import settings
        
        db_url = settings.DATABASE_URL
        
        # Buscar archivo de backup de base de datos
        db_backup_files = list(backup_dir.glob("database_*.db")) + list(backup_dir.glob("database_*.sql"))
        
        if not db_backup_files:
            print_warning("No se encontró backup de base de datos")
            return False
        
        db_backup_file = db_backup_files[0]
        print_info(f"Restaurando desde: {db_backup_file.name}")
        
        if dry_run:
            print_info("[DRY RUN] Se restauraría la base de datos")
            return True
        
        # SQLite
        if 'sqlite' in db_url.lower():
            db_path = db_url.split('///')[-1] if '///' in db_url else db_url.split(':///')[-1]
            db_file = Path(db_path)
            
            # Backup del archivo actual antes de restaurar
            if db_file.exists():
                backup_current = db_file.parent / f"{db_file.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(db_file, backup_current)
                print_info(f"Backup del archivo actual creado: {backup_current.name}")
            
            # Restaurar
            shutil.copy2(db_backup_file, db_file)
            print_success(f"Base de datos SQLite restaurada")
            return True
        
        # PostgreSQL
        elif 'postgresql' in db_url.lower():
            import re
            match = re.search(r'postgresql[^:]*://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', db_url)
            
            if match:
                user, password, host, port, dbname = match.groups()
                
                env = {'PGPASSWORD': password}
                result = subprocess.run(
                    ['psql', '-h', host, '-p', port, '-U', user, '-d', dbname, '-f', str(db_backup_file)],
                    env=env,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print_success("Base de datos PostgreSQL restaurada")
                    return True
                else:
                    print_error(f"Error restaurando PostgreSQL: {result.stderr}")
                    return False
        
        else:
            print_warning(f"Tipo de base de datos no soportado para restore: {db_url}")
            return False
            
    except Exception as e:
        print_error(f"Error restaurando base de datos: {e}")
        return False

def restore_files(backup_dir: Path, dry_run: bool = False) -> bool:
    """Restaurar archivos desde backup."""
    try:
        from config.settings import settings
        
        # Restaurar storage
        storage_backup = backup_dir / "storage"
        if storage_backup.exists():
            storage_path = Path(settings.STORAGE_PATH)
            
            if dry_run:
                print_info(f"[DRY RUN] Se restauraría storage a: {storage_path}")
            else:
                if storage_path.exists():
                    # Backup del storage actual
                    backup_current = storage_path.parent / f"{storage_path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copytree(storage_path, backup_current, dirs_exist_ok=True)
                    print_info(f"Backup del storage actual creado: {backup_current.name}")
                
                shutil.copytree(storage_backup, storage_path, dirs_exist_ok=True)
                print_success("Storage restaurado")
        
        # Restaurar .env si existe
        env_backup = backup_dir / ".env"
        if env_backup.exists():
            env_file = Path('.env')
            if dry_run:
                print_info("[DRY RUN] Se restauraría .env")
            else:
                if env_file.exists():
                    backup_current = env_file.parent / f"{env_file.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copy2(env_file, backup_current)
                    print_info(f"Backup del .env actual creado: {backup_current.name}")
                
                shutil.copy2(env_backup, env_file)
                print_success(".env restaurado")
        
        return True
        
    except Exception as e:
        print_error(f"Error restaurando archivos: {e}")
        return False

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Restaurar backup del GitHub Autonomous Agent")
    parser.add_argument("backup_path", help="Ruta al directorio de backup a restaurar")
    parser.add_argument("--dry-run", action="store_true", help="Simular restauración sin hacer cambios")
    parser.add_argument("--skip-verify", action="store_true", help="Saltar verificación de integridad")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    import os
    os.chdir(script_dir)
    
    backup_dir = Path(args.backup_path)
    
    if not backup_dir.exists():
        print_error(f"Directorio de backup no encontrado: {backup_dir}")
        return 1
    
    print_header("Restore - GitHub Autonomous Agent")
    
    if args.dry_run:
        print_warning("MODO DRY RUN - No se realizarán cambios")
    
    # Verificar integridad
    if not args.skip_verify:
        if not verify_backup_integrity(backup_dir):
            response = input("¿Continuar de todas formas? (s/N): ")
            if response.lower() != 's':
                print_info("Restauración cancelada")
                return 1
    else:
        print_warning("Verificación de integridad omitida")
    
    # Confirmar
    if not args.dry_run:
        print_warning("⚠️  Esta operación sobrescribirá datos actuales")
        response = input("¿Continuar? (s/N): ")
        if response.lower() != 's':
            print_info("Restauración cancelada")
            return 0
    
    # Restaurar base de datos
    print_info("\nRestaurando base de datos...")
    db_ok = restore_database(backup_dir, args.dry_run)
    
    # Restaurar archivos
    print_info("\nRestaurando archivos...")
    files_ok = restore_files(backup_dir, args.dry_run)
    
    # Resumen
    print_header("Resumen de Restauración")
    
    if db_ok and files_ok:
        if args.dry_run:
            print_success("✅ Simulación de restauración completada")
        else:
            print_success("✅ Restauración completada exitosamente")
        return 0
    else:
        print_error("⚠️  Restauración completada con errores")
        return 1

if __name__ == "__main__":
    sys.exit(main())



