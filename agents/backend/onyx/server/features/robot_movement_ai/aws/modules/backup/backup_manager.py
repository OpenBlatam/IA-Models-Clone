"""
Backup Manager
==============

Advanced backup management.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Backup types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


@dataclass
class Backup:
    """Backup definition."""
    id: str
    type: BackupType
    resource: str
    location: str
    size_bytes: int
    created_at: datetime
    status: str = "completed"  # pending, in_progress, completed, failed


class BackupManager:
    """Advanced backup manager."""
    
    def __init__(self):
        self._backups: Dict[str, Backup] = {}
        self._schedules: Dict[str, Dict[str, Any]] = {}
        self._backup_tasks: Dict[str, asyncio.Task] = {}
    
    def register_backup_resource(
        self,
        resource_id: str,
        backup_handler: Callable,
        schedule: Optional[str] = None
    ):
        """Register resource for backup."""
        self._schedules[resource_id] = {
            "handler": backup_handler,
            "schedule": schedule,
            "last_backup": None
        }
        logger.info(f"Registered backup resource: {resource_id}")
    
    async def create_backup(
        self,
        resource_id: str,
        backup_type: BackupType = BackupType.FULL,
        location: Optional[str] = None
    ) -> Optional[Backup]:
        """Create backup."""
        if resource_id not in self._schedules:
            logger.warning(f"Resource {resource_id} not registered for backup")
            return None
        
        schedule = self._schedules[resource_id]
        handler = schedule["handler"]
        
        backup_id = f"{resource_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Execute backup
            if asyncio.iscoroutinefunction(handler):
                result = await handler()
            else:
                result = handler()
            
            backup = Backup(
                id=backup_id,
                type=backup_type,
                resource=resource_id,
                location=location or f"backups/{backup_id}",
                size_bytes=result.get("size_bytes", 0) if isinstance(result, dict) else 0,
                created_at=datetime.now(),
                status="completed"
            )
            
            self._backups[backup_id] = backup
            schedule["last_backup"] = datetime.now()
            
            logger.info(f"Backup created: {backup_id}")
            return backup
        
        except Exception as e:
            logger.error(f"Backup failed for {resource_id}: {e}")
            backup = Backup(
                id=backup_id,
                type=backup_type,
                resource=resource_id,
                location=location or "",
                size_bytes=0,
                created_at=datetime.now(),
                status="failed"
            )
            self._backups[backup_id] = backup
            return backup
    
    def schedule_backup(
        self,
        resource_id: str,
        schedule: str,  # cron-like or interval
        backup_type: BackupType = BackupType.FULL
    ):
        """Schedule automatic backup."""
        if resource_id not in self._schedules:
            logger.warning(f"Resource {resource_id} not registered")
            return
        
        self._schedules[resource_id]["schedule"] = schedule
        self._schedules[resource_id]["backup_type"] = backup_type
        
        # In production, implement cron-like scheduling
        logger.info(f"Scheduled backup for {resource_id}: {schedule}")
    
    def get_backups(self, resource_id: Optional[str] = None, limit: int = 100) -> List[Backup]:
        """Get backups."""
        backups = list(self._backups.values())
        
        if resource_id:
            backups = [b for b in backups if b.resource == resource_id]
        
        # Sort by date
        backups.sort(key=lambda x: x.created_at, reverse=True)
        return backups[:limit]
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_size = sum(b.size_bytes for b in self._backups.values())
        successful = sum(1 for b in self._backups.values() if b.status == "completed")
        failed = sum(1 for b in self._backups.values() if b.status == "failed")
        
        return {
            "total_backups": len(self._backups),
            "successful": successful,
            "failed": failed,
            "total_size_bytes": total_size,
            "total_size_gb": total_size / 1024 / 1024 / 1024,
            "registered_resources": len(self._schedules)
        }















