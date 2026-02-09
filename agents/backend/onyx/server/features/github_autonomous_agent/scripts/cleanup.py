#!/usr/bin/env python3
"""
Script para limpiar archivos temporales, cache y logs antiguos.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta

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

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_header(msg: str):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{msg}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def cleanup_python_cache(base_dir: Path, dry_run: bool = True) -> int:
    """Limpia archivos __pycache__ y .pyc."""
    count = 0
    
    for pycache in base_dir.rglob('__pycache__'):
        if dry_run:
            print_info(f"[DRY RUN] Eliminaría: {pycache}")
        else:
            shutil.rmtree(pycache)
            print_success(f"Eliminado: {pycache}")
        count += 1
    
    for pyc in base_dir.rglob('*.pyc'):
        if dry_run:
            print_info(f"[DRY RUN] Eliminaría: {pyc}")
        else:
            pyc.unlink()
            print_success(f"Eliminado: {pyc}")
        count += 1
    
    for pyo in base_dir.rglob('*.pyo'):
        if dry_run:
            print_info(f"[DRY RUN] Eliminaría: {pyo}")
        else:
            pyo.unlink()
            print_success(f"Eliminado: {pyo}")
        count += 1
    
    return count

def cleanup_test_artifacts(base_dir: Path, dry_run: bool = True) -> int:
    """Limpia artefactos de testing."""
    count = 0
    patterns = ['.pytest_cache', '.coverage', 'htmlcov', '.tox', '.hypothesis']
    
    for pattern in patterns:
        for item in base_dir.rglob(pattern):
            if item.is_dir():
                if dry_run:
                    print_info(f"[DRY RUN] Eliminaría directorio: {item}")
                else:
                    shutil.rmtree(item)
                    print_success(f"Eliminado directorio: {item}")
                count += 1
            elif item.is_file():
                if dry_run:
                    print_info(f"[DRY RUN] Eliminaría archivo: {item}")
                else:
                    item.unlink()
                    print_success(f"Eliminado archivo: {item}")
                count += 1
    
    return count

def cleanup_logs(base_dir: Path, days: int = 7, dry_run: bool = True) -> int:
    """Limpia logs más antiguos que X días."""
    count = 0
    cutoff_date = datetime.now() - timedelta(days=days)
    log_dirs = ['logs', 'storage/logs']
    
    for log_dir_name in log_dirs:
        log_dir = base_dir / log_dir_name
        if not log_dir.exists():
            continue
        
        for log_file in log_dir.rglob('*.log'):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime < cutoff_date:
                    if dry_run:
                        print_info(f"[DRY RUN] Eliminaría log antiguo: {log_file} ({mtime.date()})")
                    else:
                        log_file.unlink()
                        print_success(f"Eliminado log: {log_file}")
                    count += 1
            except Exception as e:
                print_warning(f"Error procesando {log_file}: {e}")
    
    return count

def cleanup_backups(base_dir: Path, keep_days: int = 30, dry_run: bool = True) -> int:
    """Limpia backups antiguos, manteniendo los últimos X días."""
    count = 0
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    backup_dir = base_dir / 'backups'
    
    if not backup_dir.exists():
        return 0
    
    for backup_item in backup_dir.iterdir():
        try:
            mtime = datetime.fromtimestamp(backup_item.stat().st_mtime)
            if mtime < cutoff_date:
                if dry_run:
                    print_info(f"[DRY RUN] Eliminaría backup antiguo: {backup_item} ({mtime.date()})")
                else:
                    if backup_item.is_dir():
                        shutil.rmtree(backup_item)
                    else:
                        backup_item.unlink()
                    print_success(f"Eliminado backup: {backup_item}")
                count += 1
        except Exception as e:
            print_warning(f"Error procesando {backup_item}: {e}")
    
    return count

def cleanup_build_artifacts(base_dir: Path, dry_run: bool = True) -> int:
    """Limpia artefactos de build."""
    count = 0
    patterns = ['build', 'dist', '*.egg-info']
    
    for pattern in patterns:
        if '*' in pattern:
            for item in base_dir.rglob(pattern):
                if item.is_dir():
                    if dry_run:
                        print_info(f"[DRY RUN] Eliminaría: {item}")
                    else:
                        shutil.rmtree(item)
                        print_success(f"Eliminado: {item}")
                    count += 1
        else:
            item = base_dir / pattern
            if item.exists():
                if dry_run:
                    print_info(f"[DRY RUN] Eliminaría: {item}")
                else:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    print_success(f"Eliminado: {item}")
                count += 1
    
    return count

def get_directory_size(path: Path) -> int:
    """Obtiene tamaño total de un directorio en bytes."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except:
        pass
    return total

def format_size(size_bytes: int) -> str:
    """Formatea tamaño en formato legible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    """Función principal."""
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Limpiar archivos temporales y cache')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Modo dry-run (no elimina archivos)')
    parser.add_argument('--force', action='store_true',
                       help='Eliminar realmente (desactiva dry-run)')
    parser.add_argument('--logs-days', type=int, default=7,
                       help='Días de logs a mantener (default: 7)')
    parser.add_argument('--backups-days', type=int, default=30,
                       help='Días de backups a mantener (default: 30)')
    
    args = parser.parse_args()
    dry_run = not args.force
    
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print_header("Cleanup - GitHub Autonomous Agent")
    
    if dry_run:
        print_info("Modo DRY RUN - No se eliminarán archivos realmente")
        print_info("Usa --force para eliminar realmente\n")
    
    total_cleaned = 0
    space_freed = 0
    
    # Python cache
    print_info("Limpiando Python cache...")
    count = cleanup_python_cache(script_dir, dry_run)
    total_cleaned += count
    print_info(f"Archivos de cache: {count}\n")
    
    # Test artifacts
    print_info("Limpiando artefactos de testing...")
    count = cleanup_test_artifacts(script_dir, dry_run)
    total_cleaned += count
    print_info(f"Artefactos de test: {count}\n")
    
    # Logs
    print_info(f"Limpiando logs más antiguos de {args.logs_days} días...")
    count = cleanup_logs(script_dir, args.logs_days, dry_run)
    total_cleaned += count
    print_info(f"Logs eliminados: {count}\n")
    
    # Backups
    print_info(f"Limpiando backups más antiguos de {args.backups_days} días...")
    count = cleanup_backups(script_dir, args.backups_days, dry_run)
    total_cleaned += count
    print_info(f"Backups eliminados: {count}\n")
    
    # Build artifacts
    print_info("Limpiando artefactos de build...")
    count = cleanup_build_artifacts(script_dir, dry_run)
    total_cleaned += count
    print_info(f"Artefactos de build: {count}\n")
    
    # Resumen
    print_header("Resumen de Limpieza")
    print_info(f"Total de items procesados: {total_cleaned}")
    
    if not dry_run:
        print_success("✅ Limpieza completada")
    else:
        print_info("✅ Análisis completado (modo dry-run)")
        print_info("Ejecuta con --force para eliminar realmente")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())




