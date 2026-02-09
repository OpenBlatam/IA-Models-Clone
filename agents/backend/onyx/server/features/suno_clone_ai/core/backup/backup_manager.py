"""
Backup Manager

Utilities for backing up and restoring models and checkpoints.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage backups of models and checkpoints."""
    
    def __init__(self, backup_dir: str = "./backups"):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Backup directory
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.backup_dir / "backups.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load backup index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.backups = json.load(f)
        else:
            self.backups = {}
    
    def _save_index(self) -> None:
        """Save backup index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.backups, f, indent=2)
    
    def create_backup(
        self,
        source_path: str,
        backup_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create backup.
        
        Args:
            source_path: Path to backup
            backup_name: Backup name (auto-generated if None)
            metadata: Optional metadata
            
        Returns:
            Backup path
        """
        source = Path(source_path)
        
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")
        
        if backup_name is None:
            backup_name = f"{source.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        
        # Copy file or directory
        if source.is_file():
            backup_path = backup_path.with_suffix(source.suffix)
            shutil.copy2(source, backup_path)
        else:
            shutil.copytree(source, backup_path, dirs_exist_ok=True)
        
        # Update index
        self.backups[backup_name] = {
            'source_path': str(source),
            'backup_path': str(backup_path),
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save_index()
        
        logger.info(f"Created backup: {backup_name} -> {backup_path}")
        
        return str(backup_path)
    
    def restore_backup(
        self,
        backup_name: str,
        target_path: Optional[str] = None
    ) -> str:
        """
        Restore backup.
        
        Args:
            backup_name: Backup name
            target_path: Target path (uses original if None)
            
        Returns:
            Restored path
        """
        if backup_name not in self.backups:
            raise ValueError(f"Backup not found: {backup_name}")
        
        backup_info = self.backups[backup_name]
        backup_path = Path(backup_info['backup_path'])
        target = Path(target_path or backup_info['source_path'])
        
        # Ensure target directory exists
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy backup to target
        if backup_path.is_file():
            shutil.copy2(backup_path, target)
        else:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(backup_path, target)
        
        logger.info(f"Restored backup: {backup_name} -> {target}")
        
        return str(target)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all backups.
        
        Returns:
            List of backup information
        """
        return list(self.backups.values())
    
    def delete_backup(self, backup_name: str) -> None:
        """
        Delete backup.
        
        Args:
            backup_name: Backup name
        """
        if backup_name not in self.backups:
            raise ValueError(f"Backup not found: {backup_name}")
        
        backup_info = self.backups[backup_name]
        backup_path = Path(backup_info['backup_path'])
        
        if backup_path.exists():
            if backup_path.is_file():
                backup_path.unlink()
            else:
                shutil.rmtree(backup_path)
        
        del self.backups[backup_name]
        self._save_index()
        
        logger.info(f"Deleted backup: {backup_name}")


def create_backup(
    source_path: str,
    backup_dir: str = "./backups",
    **kwargs
) -> str:
    """Create backup."""
    manager = BackupManager(backup_dir)
    return manager.create_backup(source_path, **kwargs)


def restore_backup(
    backup_name: str,
    backup_dir: str = "./backups",
    **kwargs
) -> str:
    """Restore backup."""
    manager = BackupManager(backup_dir)
    return manager.restore_backup(backup_name, **kwargs)


def list_backups(backup_dir: str = "./backups") -> List[Dict[str, Any]]:
    """List backups."""
    manager = BackupManager(backup_dir)
    return manager.list_backups()



