"""
MOEA Cleanup Tool - Herramienta de limpieza
===========================================
Limpia archivos temporales, logs antiguos y proyectos obsoletos
"""
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import json


class MOEACleanup:
    """Herramienta de limpieza MOEA"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.cleaned_items: List[Dict] = []
    
    def cleanup_temp_files(self, directories: List[str]) -> int:
        """Limpiar archivos temporales"""
        count = 0
        temp_patterns = ['*.tmp', '*.temp', '*.log', '__pycache__', '*.pyc', '.pytest_cache']
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue
            
            for pattern in temp_patterns:
                for item in dir_path.rglob(pattern):
                    if item.is_file():
                        if not self.dry_run:
                            item.unlink()
                        self.cleaned_items.append({
                            "type": "temp_file",
                            "path": str(item),
                            "action": "deleted"
                        })
                        count += 1
                    elif item.is_dir() and item.name in ['__pycache__', '.pytest_cache']:
                        if not self.dry_run:
                            shutil.rmtree(item)
                        self.cleaned_items.append({
                            "type": "temp_dir",
                            "path": str(item),
                            "action": "deleted"
                        })
                        count += 1
        
        return count
    
    def cleanup_old_logs(self, log_dir: str, days: int = 30) -> int:
        """Limpiar logs antiguos"""
        log_path = Path(log_dir)
        if not log_path.exists():
            return 0
        
        cutoff = datetime.now() - timedelta(days=days)
        count = 0
        
        for log_file in log_path.glob('*.log'):
            if log_file.stat().st_mtime < cutoff.timestamp():
                if not self.dry_run:
                    log_file.unlink()
                self.cleaned_items.append({
                    "type": "old_log",
                    "path": str(log_file),
                    "action": "deleted",
                    "age_days": days
                })
                count += 1
        
        return count
    
    def cleanup_old_backups(self, backup_dir: str, days: int = 90, keep_last: int = 10) -> int:
        """Limpiar backups antiguos manteniendo los últimos N"""
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return 0
        
        manifest_file = backup_path / "backup_manifest.json"
        if not manifest_file.exists():
            return 0
        
        try:
            with open(manifest_file, 'r') as f:
                manifests = json.load(f)
        except:
            return 0
        
        # Ordenar por fecha
        manifests.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        cutoff = datetime.now() - timedelta(days=days)
        count = 0
        
        # Mantener los últimos N
        to_keep = manifests[:keep_last]
        to_check = manifests[keep_last:]
        
        for manifest in to_check:
            created_at = datetime.fromisoformat(manifest.get('created_at', ''))
            if created_at < cutoff:
                backup_file = Path(manifest.get('file', ''))
                if backup_file.exists():
                    if not self.dry_run:
                        backup_file.unlink()
                    self.cleaned_items.append({
                        "type": "old_backup",
                        "path": str(backup_file),
                        "action": "deleted",
                        "age_days": (datetime.now() - created_at).days
                    })
                    count += 1
                    manifests.remove(manifest)
        
        if not self.dry_run and count > 0:
            with open(manifest_file, 'w') as f:
                json.dump(manifests, f, indent=2)
        
        return count
    
    def cleanup_empty_dirs(self, directories: List[str]) -> int:
        """Limpiar directorios vacíos"""
        count = 0
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue
            
            for item in dir_path.rglob('*'):
                if item.is_dir():
                    try:
                        if not any(item.iterdir()):
                            if not self.dry_run:
                                item.rmdir()
                            self.cleaned_items.append({
                                "type": "empty_dir",
                                "path": str(item),
                                "action": "deleted"
                            })
                            count += 1
                    except:
                        pass
        
        return count
    
    def cleanup_old_metrics(self, metrics_file: str, keep_last: int = 1000) -> int:
        """Limpiar métricas antiguas manteniendo las últimas N"""
        metrics_path = Path(metrics_file)
        if not metrics_path.exists():
            return 0
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except:
            return 0
        
        original_count = len(metrics)
        
        if len(metrics) > keep_last:
            if not self.dry_run:
                metrics = metrics[-keep_last:]
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            removed = original_count - keep_last
            self.cleaned_items.append({
                "type": "old_metrics",
                "path": str(metrics_path),
                "action": "trimmed",
                "removed": removed,
                "kept": keep_last
            })
            return removed
        
        return 0
    
    def get_summary(self) -> Dict:
        """Obtener resumen de limpieza"""
        summary = {
            "total_items": len(self.cleaned_items),
            "by_type": {},
            "total_size_freed": 0
        }
        
        for item in self.cleaned_items:
            item_type = item['type']
            summary['by_type'][item_type] = summary['by_type'].get(item_type, 0) + 1
        
        return summary


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Cleanup Tool")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simular sin eliminar archivos'
    )
    parser.add_argument(
        '--temp-files',
        action='store_true',
        help='Limpiar archivos temporales'
    )
    parser.add_argument(
        '--old-logs',
        type=int,
        metavar='DAYS',
        help='Limpiar logs más antiguos que N días'
    )
    parser.add_argument(
        '--old-backups',
        type=int,
        metavar='DAYS',
        help='Limpiar backups más antiguos que N días'
    )
    parser.add_argument(
        '--empty-dirs',
        action='store_true',
        help='Limpiar directorios vacíos'
    )
    parser.add_argument(
        '--old-metrics',
        type=int,
        metavar='KEEP',
        help='Limpiar métricas manteniendo las últimas N'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Ejecutar todas las limpiezas'
    )
    parser.add_argument(
        '--directories',
        nargs='+',
        default=['generated_projects', 'moea_backups', 'moea_visualizations'],
        help='Directorios a limpiar'
    )
    
    args = parser.parse_args()
    
    cleanup = MOEACleanup(dry_run=args.dry_run)
    
    if args.dry_run:
        print("🔍 Modo DRY-RUN: No se eliminarán archivos\n")
    
    total_cleaned = 0
    
    if args.all or args.temp_files:
        count = cleanup.cleanup_temp_files(args.directories)
        total_cleaned += count
        print(f"🧹 Archivos temporales: {count}")
    
    if args.all or args.old_logs:
        count = cleanup.cleanup_old_logs('logs', args.old_logs or 30)
        total_cleaned += count
        print(f"📋 Logs antiguos: {count}")
    
    if args.all or args.old_backups:
        count = cleanup.cleanup_old_backups('moea_backups', args.old_backups or 90)
        total_cleaned += count
        print(f"📦 Backups antiguos: {count}")
    
    if args.all or args.empty_dirs:
        count = cleanup.cleanup_empty_dirs(args.directories)
        total_cleaned += count
        print(f"📁 Directorios vacíos: {count}")
    
    if args.all or args.old_metrics:
        count = cleanup.cleanup_old_metrics('moea_metrics.json', args.old_metrics or 1000)
        total_cleaned += count
        print(f"📊 Métricas antiguas: {count}")
    
    print(f"\n✅ Total limpiado: {total_cleaned} items")
    
    if args.dry_run:
        print("\n💡 Ejecuta sin --dry-run para aplicar cambios")


if __name__ == "__main__":
    main()

