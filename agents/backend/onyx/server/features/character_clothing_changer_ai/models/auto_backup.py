"""
Auto Backup for Flux2 Clothing Changer
======================================

Automatic backup and recovery system.
"""

import time
import shutil
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Backup:
    """Backup information."""
    backup_id: str
    source_path: Path
    destination_path: Path
    backup_type: str
    created_at: float
    size: int = 0
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AutoBackup:
    """Automatic backup system."""
    
    def __init__(
        self,
        backup_directory: Path = Path("backups"),
        max_backups: int = 10,
    ):
        """
        Initialize auto backup.
        
        Args:
            backup_directory: Backup directory
            max_backups: Maximum number of backups to keep
        """
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        
        self.backups: Dict[str, List[Backup]] = {}
        self.backup_schedules: Dict[str, Dict[str, Any]] = {}
    
    def create_backup(
        self,
        source_path: Path,
        backup_id: Optional[str] = None,
        backup_type: str = "full",
    ) -> Backup:
        """
        Create backup.
        
        Args:
            source_path: Source path to backup
            backup_id: Optional backup identifier
            backup_type: Backup type
            
        Returns:
            Created backup
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        backup_id = backup_id or f"backup_{int(time.time())}"
        timestamp = int(time.time())
        
        # Create destination path
        dest_name = f"{backup_id}_{timestamp}"
        dest_path = self.backup_directory / dest_name
        
        # Copy files
        if source_path.is_file():
            shutil.copy2(source_path, dest_path)
            size = dest_path.stat().st_size
        else:
            shutil.copytree(source_path, dest_path)
            size = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
        
        # Calculate checksum
        checksum = self._calculate_checksum(dest_path)
        
        backup = Backup(
            backup_id=backup_id,
            source_path=source_path,
            destination_path=dest_path,
            backup_type=backup_type,
            created_at=time.time(),
            size=size,
            checksum=checksum,
        )
        
        # Store backup
        source_key = str(source_path)
        if source_key not in self.backups:
            self.backups[source_key] = []
        
        self.backups[source_key].append(backup)
        
        # Cleanup old backups
        self._cleanup_old_backups(source_key)
        
        logger.info(f"Created backup: {backup_id}")
        return backup
    
    def restore_backup(
        self,
        backup_id: str,
        destination_path: Optional[Path] = None,
    ) -> bool:
        """
        Restore from backup.
        
        Args:
            backup_id: Backup identifier
            destination_path: Optional destination path
            
        Returns:
            True if restored
        """
        # Find backup
        backup = None
        for backups_list in self.backups.values():
            for b in backups_list:
                if b.backup_id == backup_id:
                    backup = b
                    break
            if backup:
                break
        
        if not backup:
            return False
        
        dest = destination_path or backup.source_path
        
        try:
            if backup.destination_path.is_file():
                shutil.copy2(backup.destination_path, dest)
            else:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(backup.destination_path, dest)
            
            logger.info(f"Restored backup: {backup_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate path checksum."""
        hash_md5 = hashlib.md5()
        
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            for file_path in sorted(path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _cleanup_old_backups(self, source_key: str) -> None:
        """Cleanup old backups."""
        if source_key not in self.backups:
            return
        
        backups = self.backups[source_key]
        if len(backups) > self.max_backups:
            # Sort by creation time
            backups.sort(key=lambda b: b.created_at)
            
            # Remove oldest
            while len(backups) > self.max_backups:
                old_backup = backups.pop(0)
                try:
                    if old_backup.destination_path.exists():
                        if old_backup.destination_path.is_file():
                            old_backup.destination_path.unlink()
                        else:
                            shutil.rmtree(old_backup.destination_path)
                    logger.info(f"Removed old backup: {old_backup.backup_id}")
                except Exception as e:
                    logger.error(f"Failed to remove old backup: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_size = sum(
            sum(b.size for b in backups)
            for backups in self.backups.values()
        )
        
        return {
            "total_backup_sources": len(self.backups),
            "total_backups": sum(len(backups) for backups in self.backups.values()),
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }


