"""
Backup and Recovery Utilities
For database and state backup
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manages backups for database and critical state.
    Supports scheduled backups and recovery.
    """
    
    def __init__(self, storage_adapter):
        self.storage = storage_adapter
        self.backup_retention_days = 30
    
    async def create_backup(
        self,
        backup_type: str = "full",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create backup
        
        Args:
            backup_type: Type of backup (full, incremental, differential)
            metadata: Additional metadata
            
        Returns:
            Backup ID
        """
        backup_id = f"backup-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            # Create backup based on type
            if backup_type == "full":
                backup_data = await self._create_full_backup()
            elif backup_type == "incremental":
                backup_data = await self._create_incremental_backup()
            else:
                backup_data = await self._create_differential_backup()
            
            # Store backup
            await self.storage.save_backup(backup_id, backup_data, metadata)
            
            logger.info(f"Backup created: {backup_id} (type: {backup_type})")
            return backup_id
            
        except Exception as e:
            logger.error(f"Backup failed: {e}", exc_info=True)
            raise
    
    async def restore_backup(self, backup_id: str) -> bool:
        """
        Restore from backup
        
        Args:
            backup_id: Backup ID to restore
            
        Returns:
            True if successful
        """
        try:
            backup_data = await self.storage.get_backup(backup_id)
            
            if not backup_data:
                raise ValueError(f"Backup {backup_id} not found")
            
            # Restore data
            await self._restore_data(backup_data)
            
            logger.info(f"Backup restored: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}", exc_info=True)
            return False
    
    async def list_backups(self) -> list[Dict[str, Any]]:
        """List all available backups"""
        return await self.storage.list_backups()
    
    async def delete_old_backups(self):
        """Delete backups older than retention period"""
        backups = await self.list_backups()
        cutoff_date = datetime.utcnow() - timedelta(days=self.backup_retention_days)
        
        deleted = 0
        for backup in backups:
            if backup["created_at"] < cutoff_date:
                await self.storage.delete_backup(backup["id"])
                deleted += 1
        
        logger.info(f"Deleted {deleted} old backups")
        return deleted
    
    async def _create_full_backup(self) -> Dict[str, Any]:
        """Create full backup"""
        # Implementation depends on storage
        return {"type": "full", "data": {}}
    
    async def _create_incremental_backup(self) -> Dict[str, Any]:
        """Create incremental backup"""
        return {"type": "incremental", "data": {}}
    
    async def _create_differential_backup(self) -> Dict[str, Any]:
        """Create differential backup"""
        return {"type": "differential", "data": {}}
    
    async def _restore_data(self, backup_data: Dict[str, Any]):
        """Restore data from backup"""
        # Implementation depends on storage
        pass


# Global backup manager
_backup_manager: Optional[BackupManager] = None


def get_backup_manager(storage_adapter=None) -> Optional[BackupManager]:
    """Get or create backup manager"""
    global _backup_manager
    if _backup_manager is None and storage_adapter:
        _backup_manager = BackupManager(storage_adapter)
    return _backup_manager

