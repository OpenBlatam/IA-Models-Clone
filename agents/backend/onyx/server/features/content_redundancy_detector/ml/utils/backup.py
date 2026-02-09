"""
Backup Utilities
Backup and restore utilities
"""

import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manage backups of models and checkpoints
    """
    
    def __init__(self, backup_dir: Path = Path("backups")):
        """
        Initialize backup manager
        
        Args:
            backup_dir: Backup directory
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        source: Path,
        name: Optional[str] = None,
    ) -> Path:
        """
        Create backup of file or directory
        
        Args:
            source: Source path
            name: Backup name (optional)
            
        Returns:
            Backup path
        """
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{source.stem}_{timestamp}"
        
        backup_path = self.backup_dir / name
        
        if source.is_file():
            shutil.copy2(source, backup_path)
        else:
            shutil.copytree(source, backup_path, dirs_exist_ok=True)
        
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    
    def list_backups(self) -> List[Path]:
        """
        List all backups
        
        Returns:
            List of backup paths
        """
        return sorted(self.backup_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    
    def restore_backup(
        self,
        backup_name: str,
        destination: Path,
    ) -> None:
        """
        Restore from backup
        
        Args:
            backup_name: Backup name
            destination: Destination path
        """
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if backup_path.is_file():
            shutil.copy2(backup_path, destination)
        else:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(backup_path, destination)
        
        logger.info(f"Restored backup: {backup_path} -> {destination}")
    
    def cleanup_old_backups(self, keep_last_n: int = 10) -> None:
        """
        Cleanup old backups, keeping only last N
        
        Args:
            keep_last_n: Number of backups to keep
        """
        backups = self.list_backups()
        if len(backups) > keep_last_n:
            for backup in backups[keep_last_n:]:
                if backup.is_file():
                    backup.unlink()
                else:
                    shutil.rmtree(backup)
                logger.info(f"Removed old backup: {backup}")



