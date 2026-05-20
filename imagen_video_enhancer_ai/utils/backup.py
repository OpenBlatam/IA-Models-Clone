"""
Backup and Recovery Utilities for Imagen Video Enhancer AI
==========================================================

Backup and recovery system for tasks and results.
"""

import logging
import json
import shutil
import zipfile
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manages backups of tasks and results.
    
    Features:
    - Full backup
    - Incremental backup
    - Backup compression
    - Backup restoration
    - Backup listing
    """
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory for backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        source_dir: str,
        backup_name: Optional[str] = None,
        compress: bool = True
    ) -> str:
        """
        Create a backup.
        
        Args:
            source_dir: Directory to backup
            backup_name: Optional backup name (defaults to timestamp)
            compress: Whether to compress backup
            
        Returns:
            Path to backup file
        """
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        source_path = Path(source_dir)
        if not source_path.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        
        backup_path = self.backup_dir / backup_name
        
        if compress:
            backup_file = backup_path.with_suffix(".zip")
            self._create_zip_backup(source_path, backup_file)
            logger.info(f"Created compressed backup: {backup_file}")
            return str(backup_file)
        else:
            backup_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_path, backup_path / source_path.name, dirs_exist_ok=True)
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
    
    def _create_zip_backup(self, source_path: Path, backup_file: Path):
        """Create ZIP backup."""
        with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_path)
                    zipf.write(file_path, arcname)
    
    def restore_backup(
        self,
        backup_path: str,
        target_dir: str,
        overwrite: bool = False
    ) -> str:
        """
        Restore a backup.
        
        Args:
            backup_path: Path to backup file
            target_dir: Directory to restore to
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path to restored directory
        """
        backup_file = Path(backup_path)
        target_path = Path(target_dir)
        
        if not backup_file.exists():
            raise ValueError(f"Backup file does not exist: {backup_path}")
        
        target_path.mkdir(parents=True, exist_ok=True)
        
        if backup_file.suffix == ".zip":
            self._restore_zip_backup(backup_file, target_path)
        else:
            if target_path.exists() and not overwrite:
                raise ValueError(f"Target directory exists: {target_dir}")
            shutil.copytree(backup_file, target_path, dirs_exist_ok=overwrite)
        
        logger.info(f"Restored backup to: {target_path}")
        return str(target_path)
    
    def _restore_zip_backup(self, backup_file: Path, target_path: Path):
        """Restore ZIP backup."""
        with zipfile.ZipFile(backup_file, 'r') as zipf:
            zipf.extractall(target_path)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("backup_*"):
            if backup_file.is_file() and backup_file.suffix == ".zip":
                stat = backup_file.stat()
                backups.append({
                    "name": backup_file.stem,
                    "path": str(backup_file),
                    "size_mb": stat.st_size / (1024 * 1024),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            elif backup_file.is_dir():
                stat = backup_file.stat()
                backups.append({
                    "name": backup_file.name,
                    "path": str(backup_file),
                    "size_mb": self._get_dir_size(backup_file) / (1024 * 1024),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)
    
    def _get_dir_size(self, path: Path) -> int:
        """Get directory size in bytes."""
        total = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total
    
    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_name: Backup name or path
            
        Returns:
            True if deleted
        """
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            backup_path = Path(backup_name)
        
        if not backup_path.exists():
            logger.warning(f"Backup not found: {backup_name}")
            return False
        
        if backup_path.is_file():
            backup_path.unlink()
        else:
            shutil.rmtree(backup_path)
        
        logger.info(f"Deleted backup: {backup_name}")
        return True




