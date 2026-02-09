"""
Backup Manager for Piel Mejorador AI SAM3
==========================================

Manages backups and recovery of tasks and data.
"""

import asyncio
import logging
import shutil
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BackupInfo:
    """Backup information."""
    backup_id: str
    timestamp: datetime
    size_bytes: int
    task_count: int
    success: bool
    error: Optional[str] = None


class BackupManager:
    """
    Manages backups and recovery.
    
    Features:
    - Automatic backups
    - Incremental backups
    - Backup rotation
    - Recovery operations
    """
    
    def __init__(self, backup_dir: Path, retention_days: int = 7):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory for backups
            retention_days: Days to retain backups
        """
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        
        self._backup_history: List[BackupInfo] = []
    
    async def create_backup(
        self,
        source_dir: Path,
        backup_name: Optional[str] = None
    ) -> BackupInfo:
        """
        Create a backup of source directory.
        
        Args:
            source_dir: Directory to backup
            backup_name: Optional backup name
            
        Returns:
            BackupInfo
        """
        if not source_dir.exists():
            return BackupInfo(
                backup_id="",
                timestamp=datetime.now(),
                size_bytes=0,
                task_count=0,
                success=False,
                error=f"Source directory does not exist: {source_dir}"
            )
        
        backup_id = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_id
        
        try:
            # Count tasks
            task_count = len(list(source_dir.glob("*.json")))
            
            # Copy directory
            shutil.copytree(source_dir, backup_path, dirs_exist_ok=True)
            
            # Calculate size
            size_bytes = sum(
                f.stat().st_size
                for f in backup_path.rglob("*")
                if f.is_file()
            )
            
            backup_info = BackupInfo(
                backup_id=backup_id,
                timestamp=datetime.now(),
                size_bytes=size_bytes,
                task_count=task_count,
                success=True
            )
            
            self._backup_history.append(backup_info)
            logger.info(f"Backup created: {backup_id} ({size_bytes / 1024 / 1024:.2f}MB, {task_count} tasks)")
            
            return backup_info
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return BackupInfo(
                backup_id=backup_id,
                timestamp=datetime.now(),
                size_bytes=0,
                task_count=0,
                success=False,
                error=str(e)
            )
    
    async def restore_backup(
        self,
        backup_id: str,
        target_dir: Path
    ) -> bool:
        """
        Restore from backup.
        
        Args:
            backup_id: Backup identifier
            target_dir: Target directory for restoration
            
        Returns:
            True if successful
        """
        backup_path = self.backup_dir / backup_id
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        try:
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy backup to target
            shutil.copytree(backup_path, target_dir, dirs_exist_ok=True)
            
            logger.info(f"Backup restored: {backup_id} -> {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    async def cleanup_old_backups(self) -> int:
        """
        Clean up old backups based on retention policy.
        
        Returns:
            Number of backups cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        cleaned = 0
        
        for backup_path in self.backup_dir.iterdir():
            if not backup_path.is_dir():
                continue
            
            # Get backup timestamp from name or mtime
            try:
                # Try to parse from name
                timestamp_str = backup_path.name.replace("backup_", "")
                backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except:
                # Fallback to modification time
                backup_date = datetime.fromtimestamp(backup_path.stat().st_mtime)
            
            if backup_date < cutoff_date:
                try:
                    shutil.rmtree(backup_path)
                    cleaned += 1
                    logger.info(f"Cleaned up old backup: {backup_path.name}")
                except Exception as e:
                    logger.warning(f"Error cleaning backup {backup_path.name}: {e}")
        
        return cleaned
    
    def list_backups(self) -> List[BackupInfo]:
        """List all available backups."""
        backups = []
        
        for backup_path in self.backup_dir.iterdir():
            if not backup_path.is_dir():
                continue
            
            try:
                size_bytes = sum(
                    f.stat().st_size
                    for f in backup_path.rglob("*")
                    if f.is_file()
                )
                
                task_count = len(list(backup_path.glob("*.json")))
                
                # Get timestamp
                try:
                    timestamp_str = backup_path.name.replace("backup_", "")
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except:
                    timestamp = datetime.fromtimestamp(backup_path.stat().st_mtime)
                
                backups.append(BackupInfo(
                    backup_id=backup_path.name,
                    timestamp=timestamp,
                    size_bytes=size_bytes,
                    task_count=task_count,
                    success=True
                ))
            except Exception as e:
                logger.warning(f"Error reading backup {backup_path.name}: {e}")
        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        backups = self.list_backups()
        total_size = sum(b.size_bytes for b in backups)
        total_tasks = sum(b.task_count for b in backups)
        
        return {
            "total_backups": len(backups),
            "total_size_mb": total_size / (1024 * 1024),
            "total_tasks": total_tasks,
            "oldest_backup": backups[-1].timestamp.isoformat() if backups else None,
            "newest_backup": backups[0].timestamp.isoformat() if backups else None,
        }




