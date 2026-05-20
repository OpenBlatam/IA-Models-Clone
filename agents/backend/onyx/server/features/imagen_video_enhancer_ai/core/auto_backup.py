"""
Automatic Backup System
=======================

Advanced system for automatic backups with scheduling and monitoring.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Backup types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackupConfig:
    """Backup configuration."""
    name: str
    source_paths: List[str]
    backup_type: BackupType = BackupType.FULL
    schedule: Optional[str] = None  # Cron expression
    retention_days: int = 30
    compress: bool = True
    enabled: bool = True
    max_backups: Optional[int] = None
    exclude_patterns: List[str] = field(default_factory=list)


@dataclass
class BackupResult:
    """Backup execution result."""
    backup_id: str
    config_name: str
    status: BackupStatus
    backup_path: Optional[str] = None
    size_mb: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class AutoBackupManager:
    """Automatic backup manager with scheduling."""
    
    def __init__(
        self,
        backup_dir: Path,
        backup_manager: Optional[Any] = None
    ):
        """
        Initialize auto backup manager.
        
        Args:
            backup_dir: Directory for backups
            backup_manager: Optional BackupManager instance
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_manager = backup_manager
        self.configs: Dict[str, BackupConfig] = {}
        self.results: List[BackupResult] = []
        self.history_file = self.backup_dir / "backup_history.json"
        self.config_file = self.backup_dir / "backup_configs.json"
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._load_configs()
        self._load_history()
    
    def _load_configs(self):
        """Load backup configurations."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs_data = json.load(f)
                    for name, config_data in configs_data.items():
                        config = BackupConfig(
                            name=name,
                            source_paths=config_data.get("source_paths", []),
                            backup_type=BackupType(config_data.get("backup_type", "full")),
                            schedule=config_data.get("schedule"),
                            retention_days=config_data.get("retention_days", 30),
                            compress=config_data.get("compress", True),
                            enabled=config_data.get("enabled", True),
                            max_backups=config_data.get("max_backups"),
                            exclude_patterns=config_data.get("exclude_patterns", [])
                        )
                        self.configs[name] = config
            except Exception as e:
                logger.warning(f"Error loading backup configs: {e}")
    
    def _save_configs(self):
        """Save backup configurations."""
        try:
            configs_data = {
                name: {
                    "source_paths": config.source_paths,
                    "backup_type": config.backup_type.value,
                    "schedule": config.schedule,
                    "retention_days": config.retention_days,
                    "compress": config.compress,
                    "enabled": config.enabled,
                    "max_backups": config.max_backups,
                    "exclude_patterns": config.exclude_patterns
                }
                for name, config in self.configs.items()
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(configs_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving backup configs: {e}")
    
    def _load_history(self):
        """Load backup history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    for result_data in history_data:
                        result = BackupResult(
                            backup_id=result_data.get("backup_id"),
                            config_name=result_data.get("config_name"),
                            status=BackupStatus(result_data.get("status")),
                            backup_path=result_data.get("backup_path"),
                            size_mb=result_data.get("size_mb", 0.0),
                            started_at=datetime.fromisoformat(result_data["started_at"]) if result_data.get("started_at") else None,
                            completed_at=datetime.fromisoformat(result_data["completed_at"]) if result_data.get("completed_at") else None,
                            error=result_data.get("error"),
                            duration_seconds=result_data.get("duration_seconds", 0.0)
                        )
                        self.results.append(result)
            except Exception as e:
                logger.warning(f"Error loading backup history: {e}")
    
    def _save_history(self):
        """Save backup history."""
        try:
            history_data = [
                {
                    "backup_id": result.backup_id,
                    "config_name": result.config_name,
                    "status": result.status.value,
                    "backup_path": result.backup_path,
                    "size_mb": result.size_mb,
                    "started_at": result.started_at.isoformat() if result.started_at else None,
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                    "error": result.error,
                    "duration_seconds": result.duration_seconds
                }
                for result in self.results
            ]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving backup history: {e}")
    
    def register_config(self, config: BackupConfig):
        """
        Register a backup configuration.
        
        Args:
            config: Backup configuration
        """
        self.configs[config.name] = config
        self._save_configs()
        logger.info(f"Registered backup config: {config.name}")
    
    async def run_backup(self, config_name: str) -> BackupResult:
        """
        Run a backup.
        
        Args:
            config_name: Configuration name
            
        Returns:
            Backup result
        """
        if config_name not in self.configs:
            raise ValueError(f"Backup config not found: {config_name}")
        
        config = self.configs[config_name]
        
        if not config.enabled:
            logger.info(f"Backup config disabled: {config_name}")
            return BackupResult(
                backup_id=f"{config_name}_{datetime.now().isoformat()}",
                config_name=config_name,
                status=BackupStatus.CANCELLED
            )
        
        backup_id = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()
        
        result = BackupResult(
            backup_id=backup_id,
            config_name=config_name,
            status=BackupStatus.RUNNING,
            started_at=started_at
        )
        self.results.append(result)
        self._save_history()
        
        try:
            # Use backup manager if available
            if self.backup_manager:
                backup_paths = []
                for source_path in config.source_paths:
                    backup_path = self.backup_manager.create_backup(
                        source_dir=source_path,
                        backup_name=f"{backup_id}_{Path(source_path).name}",
                        compress=config.compress
                    )
                    backup_paths.append(backup_path)
                
                # Get backup size
                total_size = 0
                for backup_path in backup_paths:
                    backup_file = Path(backup_path)
                    if backup_file.exists():
                        if backup_file.is_file():
                            total_size += backup_file.stat().st_size
                        else:
                            for file_path in backup_file.rglob('*'):
                                if file_path.is_file():
                                    total_size += file_path.stat().st_size
                
                result.backup_path = backup_paths[0] if backup_paths else None
                result.size_mb = total_size / (1024 * 1024)
            else:
                logger.warning("No backup manager available")
            
            result.status = BackupStatus.COMPLETED
            result.completed_at = datetime.now()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            
            # Cleanup old backups
            await self._cleanup_old_backups(config)
            
            logger.info(f"Backup completed: {backup_id}")
            
        except Exception as e:
            result.status = BackupStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            logger.error(f"Backup failed: {backup_id} - {e}")
        
        self._save_history()
        return result
    
    async def _cleanup_old_backups(self, config: BackupConfig):
        """Cleanup old backups based on retention policy."""
        if not config.retention_days and not config.max_backups:
            return
        
        cutoff_date = datetime.now() - timedelta(days=config.retention_days) if config.retention_days else None
        
        # Get backups for this config
        config_backups = [
            result for result in self.results
            if result.config_name == config.name and result.status == BackupStatus.COMPLETED
        ]
        
        # Sort by date (newest first)
        config_backups.sort(key=lambda r: r.completed_at or datetime.min, reverse=True)
        
        # Remove old backups
        if config.max_backups:
            for backup in config_backups[config.max_backups:]:
                if backup.backup_path and Path(backup.backup_path).exists():
                    try:
                        backup_file = Path(backup.backup_path)
                        if backup_file.is_file():
                            backup_file.unlink()
                        else:
                            import shutil
                            shutil.rmtree(backup_file)
                        logger.info(f"Deleted old backup: {backup.backup_id}")
                    except Exception as e:
                        logger.warning(f"Error deleting backup: {backup.backup_id} - {e}")
        
        if cutoff_date:
            for backup in config_backups:
                if backup.completed_at and backup.completed_at < cutoff_date:
                    if backup.backup_path and Path(backup.backup_path).exists():
                        try:
                            backup_file = Path(backup.backup_path)
                            if backup_file.is_file():
                                backup_file.unlink()
                            else:
                                import shutil
                                shutil.rmtree(backup_file)
                            logger.info(f"Deleted expired backup: {backup.backup_id}")
                        except Exception as e:
                            logger.warning(f"Error deleting backup: {backup.backup_id} - {e}")
    
    async def start_scheduler(self):
        """Start backup scheduler."""
        logger.info("Starting backup scheduler")
        # Note: Full cron implementation would require additional dependencies
        # For now, this is a placeholder for scheduled backups
        while True:
            await asyncio.sleep(3600)  # Check every hour
            for config in self.configs.values():
                if config.enabled and config.schedule:
                    # Simple schedule check (full cron parsing would be needed)
                    await self.run_backup(config.name)
    
    def get_backup_stats(self, config_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get backup statistics.
        
        Args:
            config_name: Optional config name to filter
            
        Returns:
            Statistics dictionary
        """
        results = self.results
        if config_name:
            results = [r for r in results if r.config_name == config_name]
        
        total = len(results)
        completed = len([r for r in results if r.status == BackupStatus.COMPLETED])
        failed = len([r for r in results if r.status == BackupStatus.FAILED])
        total_size = sum(r.size_mb for r in results if r.status == BackupStatus.COMPLETED)
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total * 100) if total > 0 else 0,
            "total_size_mb": total_size,
            "average_size_mb": (total_size / completed) if completed > 0 else 0
        }


