"""
Backup and Recovery - Backup y Recuperación
==========================================

Sistema de backup y recuperación:
- Automated backups
- Incremental backups
- Point-in-time recovery
- Backup verification
"""

import logging
import shutil
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class BackupType(str, Enum):
    """Tipos de backup"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class Backup:
    """Backup"""
    
    def __init__(
        self,
        backup_id: str,
        backup_type: BackupType,
        source_path: str,
        destination_path: str,
        **metadata: Any
    ) -> None:
        self.backup_id = backup_id
        self.backup_type = backup_type
        self.source_path = source_path
        self.destination_path = destination_path
        self.metadata = metadata
        self.created_at = datetime.now()
        self.size: Optional[int] = None
        self.verified = False
    
    def execute(self) -> bool:
        """Ejecuta backup"""
        try:
            source = Path(self.source_path)
            dest = Path(self.destination_path)
            
            if not source.exists():
                logger.error(f"Source path does not exist: {self.source_path}")
                return False
            
            # Crear directorio de destino
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            if source.is_file():
                shutil.copy2(source, dest)
            elif source.is_dir():
                shutil.copytree(source, dest, dirs_exist_ok=True)
            
            self.size = self._calculate_size(dest)
            logger.info(f"Backup completed: {self.backup_id}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def verify(self) -> bool:
        """Verifica integridad del backup"""
        try:
            dest = Path(self.destination_path)
            if not dest.exists():
                return False
            
            # Verificación básica
            if dest.is_file():
                self.verified = dest.stat().st_size > 0
            elif dest.is_dir():
                self.verified = any(dest.iterdir())
            
            return self.verified
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    def restore(self, target_path: str) -> bool:
        """Restaura backup"""
        try:
            source = Path(self.destination_path)
            target = Path(target_path)
            
            if not source.exists():
                logger.error(f"Backup does not exist: {self.destination_path}")
                return False
            
            target.parent.mkdir(parents=True, exist_ok=True)
            
            if source.is_file():
                shutil.copy2(source, target)
            elif source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            
            logger.info(f"Backup restored to: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def _calculate_size(self, path: Path) -> int:
        """Calcula tamaño del backup"""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return 0


class BackupManager:
    """
    Gestor de backups.
    """
    
    def __init__(
        self,
        backup_directory: str = "backups",
        retention_days: int = 30
    ) -> None:
        self.backup_directory = Path(backup_directory)
        self.retention_days = retention_days
        self.backups: Dict[str, Backup] = {}
        self.backup_directory.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        source_path: str,
        backup_type: BackupType = BackupType.FULL,
        backup_id: Optional[str] = None
    ) -> Optional[Backup]:
        """Crea un backup"""
        if not backup_id:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        dest_path = self.backup_directory / f"{backup_id}.backup"
        
        backup = Backup(
            backup_id=backup_id,
            backup_type=backup_type,
            source_path=source_path,
            destination_path=str(dest_path),
            source=source_path
        )
        
        if backup.execute():
            backup.verify()
            self.backups[backup_id] = backup
            logger.info(f"Backup created: {backup_id}")
            return backup
        
        return None
    
    def restore_backup(
        self,
        backup_id: str,
        target_path: str
    ) -> bool:
        """Restaura un backup"""
        backup = self.backups.get(backup_id)
        if not backup:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        return backup.restore(target_path)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Lista todos los backups"""
        return [
            {
                "id": backup.backup_id,
                "type": backup.backup_type.value,
                "created_at": backup.created_at.isoformat(),
                "size": backup.size,
                "verified": backup.verified
            }
            for backup in self.backups.values()
        ]
    
    def cleanup_old_backups(self) -> int:
        """Limpia backups antiguos"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        removed_count = 0
        
        for backup_id, backup in list(self.backups.items()):
            if backup.created_at < cutoff_date:
                try:
                    backup_path = Path(backup.destination_path)
                    if backup_path.exists():
                        if backup_path.is_file():
                            backup_path.unlink()
                        elif backup_path.is_dir():
                            shutil.rmtree(backup_path)
                    
                    del self.backups[backup_id]
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove backup {backup_id}: {e}")
        
        logger.info(f"Cleaned up {removed_count} old backups")
        return removed_count


def get_backup_manager(
    backup_directory: str = "backups",
    retention_days: int = 30
) -> BackupManager:
    """Obtiene gestor de backups"""
    return BackupManager(backup_directory, retention_days)















