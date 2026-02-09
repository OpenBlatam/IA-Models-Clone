"""
Backup Module - Backup and restore functionality.

Provides:
- Database backup
- Configuration backup
- Incremental backups
- Restore functionality
"""

import logging
import shutil
import sqlite3
import json
import gzip
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BackupInfo:
    """Backup information."""
    backup_id: str
    backup_type: str
    path: Path
    size_bytes: int
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class BackupManager:
    """Backup and restore manager."""
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Backup directory
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backups: List[BackupInfo] = []
        self._load_backup_index()
    
    def _load_backup_index(self) -> None:
        """Load backup index."""
        index_file = self.backup_dir / "backup_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    self.backups = [
                        BackupInfo(
                            backup_id=b["backup_id"],
                            backup_type=b["backup_type"],
                            path=Path(b["path"]),
                            size_bytes=b["size_bytes"],
                            created_at=b["created_at"],
                            metadata=b.get("metadata", {}),
                        )
                        for b in data
                    ]
            except Exception as e:
                logger.error(f"Error loading backup index: {e}")
    
    def _save_backup_index(self) -> None:
        """Save backup index."""
        index_file = self.backup_dir / "backup_index.json"
        data = [b.to_dict() for b in self.backups]
        with open(index_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def backup_database(self, db_path: str, compress: bool = True) -> BackupInfo:
        """
        Backup database.
        
        Args:
            db_path: Database path
            compress: Compress backup
            
        Returns:
            Backup info
        """
        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"db_{timestamp}"
        backup_filename = f"{backup_id}.db"
        
        if compress:
            backup_filename += ".gz"
        
        backup_path = self.backup_dir / backup_filename
        
        # Copy database
        if compress:
            with open(db_path, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(db_path, backup_path)
        
        size = backup_path.stat().st_size
        
        backup_info = BackupInfo(
            backup_id=backup_id,
            backup_type="database",
            path=backup_path,
            size_bytes=size,
            created_at=datetime.now().isoformat(),
            metadata={
                "source": str(db_path),
                "compressed": compress,
            },
        )
        
        self.backups.append(backup_info)
        self._save_backup_index()
        
        logger.info(f"Backed up database: {backup_id} ({size} bytes)")
        return backup_info
    
    def backup_config(self, config_path: str, compress: bool = True) -> BackupInfo:
        """
        Backup configuration.
        
        Args:
            config_path: Configuration file path
            compress: Compress backup
            
        Returns:
            Backup info
        """
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"config_{timestamp}"
        backup_filename = f"{backup_id}.json"
        
        if compress:
            backup_filename += ".gz"
        
        backup_path = self.backup_dir / backup_filename
        
        # Copy config
        if compress:
            with open(config_path, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(config_path, backup_path)
        
        size = backup_path.stat().st_size
        
        backup_info = BackupInfo(
            backup_id=backup_id,
            backup_type="config",
            path=backup_path,
            size_bytes=size,
            created_at=datetime.now().isoformat(),
            metadata={
                "source": str(config_path),
                "compressed": compress,
            },
        )
        
        self.backups.append(backup_info)
        self._save_backup_index()
        
        logger.info(f"Backed up config: {backup_id} ({size} bytes)")
        return backup_info
    
    def restore_database(self, backup_id: str, target_path: str) -> None:
        """
        Restore database from backup.
        
        Args:
            backup_id: Backup ID
            target_path: Target database path
        """
        backup_info = next((b for b in self.backups if b.backup_id == backup_id), None)
        if not backup_info:
            raise ValueError(f"Backup not found: {backup_id}")
        
        if backup_info.backup_type != "database":
            raise ValueError(f"Backup is not a database backup: {backup_id}")
        
        if not backup_info.path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_info.path}")
        
        target_path_obj = Path(target_path)
        target_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Restore
        if backup_info.path.suffix == ".gz":
            with gzip.open(backup_info.path, 'rb') as f_in:
                with open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(backup_info.path, target_path)
        
        logger.info(f"Restored database from {backup_id} to {target_path}")
    
    def list_backups(self, backup_type: Optional[str] = None) -> List[BackupInfo]:
        """
        List backups.
        
        Args:
            backup_type: Filter by type
            
        Returns:
            List of backups
        """
        backups = self.backups
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        return sorted(backups, key=lambda b: b.created_at, reverse=True)
    
    def delete_backup(self, backup_id: str) -> None:
        """
        Delete backup.
        
        Args:
            backup_id: Backup ID
        """
        backup_info = next((b for b in self.backups if b.backup_id == backup_id), None)
        if not backup_info:
            raise ValueError(f"Backup not found: {backup_id}")
        
        if backup_info.path.exists():
            backup_info.path.unlink()
        
        self.backups.remove(backup_info)
        self._save_backup_index()
        
        logger.info(f"Deleted backup: {backup_id}")












