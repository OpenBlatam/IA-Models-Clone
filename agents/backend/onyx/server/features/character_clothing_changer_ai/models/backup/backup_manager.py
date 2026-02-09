"""
Backup Manager
==============

Automated backup system for models and data.
"""

import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Backup type."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class BackupConfig:
    """Backup configuration."""
    source_path: str
    destination_path: str
    backup_type: BackupType = BackupType.FULL
    schedule: Optional[str] = None  # Cron expression
    retention_days: int = 30
    compression: bool = True
    encryption: bool = False


@dataclass
class BackupResult:
    """Backup result."""
    backup_id: str
    config: BackupConfig
    status: BackupStatus
    duration: float
    size_bytes: int = 0
    error: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class BackupManager:
    """Backup manager."""
    
    def __init__(self):
        """Initialize backup manager."""
        self.configs: List[BackupConfig] = []
        self.history: List[BackupResult] = []
        self.on_backup_complete: Optional[Callable] = None
    
    def register_backup(
        self,
        config: BackupConfig,
    ) -> None:
        """
        Register a backup configuration.
        
        Args:
            config: Backup configuration
        """
        self.configs.append(config)
        logger.info(f"Registered backup: {config.source_path} -> {config.destination_path}")
    
    def create_backup(
        self,
        config: BackupConfig,
        backup_id: Optional[str] = None,
    ) -> BackupResult:
        """
        Create a backup.
        
        Args:
            config: Backup configuration
            backup_id: Optional backup ID
            
        Returns:
            Backup result
        """
        if backup_id is None:
            backup_id = f"backup_{int(time.time())}"
        
        start_time = time.time()
        
        result = BackupResult(
            backup_id=backup_id,
            config=config,
            status=BackupStatus.RUNNING,
            duration=0.0,
            timestamp=time.time(),
        )
        
        logger.info(f"Creating backup: {backup_id} ({config.backup_type.value})")
        
        try:
            source = Path(config.source_path)
            destination = Path(config.destination_path) / backup_id
            
            if not source.exists():
                raise FileNotFoundError(f"Source path does not exist: {config.source_path}")
            
            # Create destination directory
            destination.mkdir(parents=True, exist_ok=True)
            
            # Perform backup based on type
            if config.backup_type == BackupType.FULL:
                self._full_backup(source, destination)
            elif config.backup_type == BackupType.INCREMENTAL:
                self._incremental_backup(source, destination)
            elif config.backup_type == BackupType.DIFFERENTIAL:
                self._differential_backup(source, destination)
            
            # Calculate size
            result.size_bytes = self._calculate_size(destination)
            result.duration = time.time() - start_time
            result.status = BackupStatus.SUCCESS
            
            logger.info(f"Backup completed: {backup_id} ({result.size_bytes / 1024 / 1024:.2f} MB)")
            
        except Exception as e:
            result.duration = time.time() - start_time
            result.status = BackupStatus.FAILED
            result.error = str(e)
            logger.error(f"Backup failed: {backup_id} - {e}")
        
        self.history.append(result)
        
        # Keep only last 1000 backups
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        # Callback
        if self.on_backup_complete:
            try:
                self.on_backup_complete(result)
            except Exception as e:
                logger.error(f"Backup callback error: {e}")
        
        return result
    
    def _full_backup(self, source: Path, destination: Path) -> None:
        """Perform full backup."""
        if source.is_file():
            shutil.copy2(source, destination / source.name)
        else:
            shutil.copytree(source, destination / source.name, dirs_exist_ok=True)
    
    def _incremental_backup(self, source: Path, destination: Path) -> None:
        """Perform incremental backup."""
        # Simplified incremental backup
        # In real implementation, would track changes
        self._full_backup(source, destination)
    
    def _differential_backup(self, source: Path, destination: Path) -> None:
        """Perform differential backup."""
        # Simplified differential backup
        # In real implementation, would track differences
        self._full_backup(source, destination)
    
    def _calculate_size(self, path: Path) -> int:
        """Calculate total size of path."""
        total = 0
        if path.is_file():
            total = path.stat().st_size
        else:
            for file in path.rglob('*'):
                if file.is_file():
                    total += file.stat().st_size
        return total
    
    def restore_backup(
        self,
        backup_id: str,
        restore_path: str,
    ) -> bool:
        """
        Restore a backup.
        
        Args:
            backup_id: Backup ID
            restore_path: Path to restore to
            
        Returns:
            True if successful
        """
        # Find backup
        backup_result = None
        for result in self.history:
            if result.backup_id == backup_id:
                backup_result = result
                break
        
        if not backup_result or backup_result.status != BackupStatus.SUCCESS:
            logger.error(f"Backup not found or failed: {backup_id}")
            return False
        
        try:
            backup_path = Path(backup_result.config.destination_path) / backup_id
            restore_path_obj = Path(restore_path)
            
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup path does not exist: {backup_path}")
            
            # Restore
            if backup_path.is_file():
                shutil.copy2(backup_path, restore_path_obj)
            else:
                shutil.copytree(backup_path, restore_path_obj, dirs_exist_ok=True)
            
            logger.info(f"Backup restored: {backup_id} -> {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {backup_id} - {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """
        Cleanup old backups based on retention policy.
        
        Returns:
            Number of backups removed
        """
        removed = 0
        current_time = time.time()
        
        for config in self.configs:
            retention_seconds = config.retention_days * 24 * 60 * 60
            cutoff_time = current_time - retention_seconds
            
            for result in self.history:
                if (result.config == config and 
                    result.timestamp < cutoff_time and 
                    result.status == BackupStatus.SUCCESS):
                    
                    backup_path = Path(config.destination_path) / result.backup_id
                    if backup_path.exists():
                        try:
                            if backup_path.is_file():
                                backup_path.unlink()
                            else:
                                shutil.rmtree(backup_path)
                            removed += 1
                            logger.info(f"Removed old backup: {result.backup_id}")
                        except Exception as e:
                            logger.error(f"Failed to remove backup: {result.backup_id} - {e}")
        
        return removed
    
    def get_backup_history(
        self,
        config: Optional[BackupConfig] = None,
        limit: int = 10,
    ) -> List[BackupResult]:
        """
        Get backup history.
        
        Args:
            config: Optional backup config filter
            limit: Maximum number of results
            
        Returns:
            Backup history
        """
        history = self.history
        
        if config:
            history = [r for r in history if r.config == config]
        
        return history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get backup statistics."""
        if not self.history:
            return {
                "total_backups": 0,
                "registered_configs": len(self.configs),
            }
        
        successful = sum(1 for r in self.history if r.status == BackupStatus.SUCCESS)
        failed = sum(1 for r in self.history if r.status == BackupStatus.FAILED)
        total_size = sum(r.size_bytes for r in self.history if r.status == BackupStatus.SUCCESS)
        
        return {
            "total_backups": len(self.history),
            "registered_configs": len(self.configs),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / len(self.history) * 100) if self.history else 0.0,
            "total_size_mb": total_size / 1024 / 1024,
        }

