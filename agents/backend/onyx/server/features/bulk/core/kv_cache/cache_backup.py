"""
Cache backup and restore utilities.

Provides backup and restore capabilities for cache state.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheBackupManager:
    """
    Cache backup manager.
    
    Manages backup and restore operations for cache.
    """
    
    def __init__(
        self,
        cache: Any,
        backup_dir: str = "cache_backups",
        max_backups: int = 10
    ):
        """
        Initialize backup manager.
        
        Args:
            cache: Cache instance
            backup_dir: Directory for backups
            max_backups: Maximum number of backups to keep
        """
        self.cache = cache
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, name: Optional[str] = None) -> str:
        """
        Create backup of cache state.
        
        Args:
            name: Optional backup name (None = auto-generated)
            
        Returns:
            Path to backup file
        """
        from kv_cache import CacheSerializer
        
        if name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            name = f"cache_backup_{timestamp}"
        
        backup_path = self.backup_dir / f"{name}.pkl"
        
        # Serialize cache
        serializer = CacheSerializer(compress=True)
        data = serializer.serialize_cache(self.cache, include_stats=True)
        
        # Write to file
        with open(backup_path, "wb") as f:
            f.write(data)
        
        logger.info(f"Backup created: {backup_path}")
        
        # Cleanup old backups
        self._cleanup_old_backups()
        
        return str(backup_path)
    
    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """
        Restore cache from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Dictionary with restore information
        """
        from kv_cache import CacheSerializer
        
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Read backup
        with open(backup_path, "rb") as f:
            data = f.read()
        
        # Deserialize
        serializer = CacheSerializer(compress=True)
        restore_info = serializer.deserialize_cache(data, self.cache)
        
        logger.info(f"Backup restored from: {backup_path}")
        
        return restore_info
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List available backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*.pkl"):
            stat = backup_file.stat()
            backups.append({
                "name": backup_file.stem,
                "path": str(backup_file),
                "size_mb": stat.st_size / (1024 * 1024),
                "created": time.ctime(stat.st_ctime),
                "modified": time.ctime(stat.st_mtime)
            })
        
        # Sort by modified time (newest first)
        backups.sort(key=lambda x: x["modified"], reverse=True)
        
        return backups
    
    def _cleanup_old_backups(self) -> None:
        """Cleanup old backups if exceeding max_backups."""
        backups = self.list_backups()
        
        if len(backups) > self.max_backups:
            # Remove oldest backups
            to_remove = backups[self.max_backups:]
            for backup in to_remove:
                try:
                    os.remove(backup["path"])
                    logger.info(f"Removed old backup: {backup['name']}")
                except Exception as e:
                    logger.warning(f"Failed to remove backup {backup['name']}: {e}")
    
    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_name: Name of backup to delete
            
        Returns:
            True if deleted successfully
        """
        backup_path = self.backup_dir / f"{backup_name}.pkl"
        
        if not backup_path.exists():
            return False
        
        try:
            backup_path.unlink()
            logger.info(f"Deleted backup: {backup_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete backup {backup_name}: {e}")
            return False


class CacheSnapshot:
    """
    Cache snapshot utilities.
    
    Provides fast snapshot creation and restoration.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize snapshot manager.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.snapshots: Dict[str, Dict[str, Any]] = {}
    
    def create_snapshot(self, name: str) -> Dict[str, Any]:
        """
        Create snapshot of current cache state.
        
        Args:
            name: Snapshot name
            
        Returns:
            Snapshot metadata
        """
        stats = self.cache.get_stats()
        
        snapshot = {
            "name": name,
            "timestamp": time.time(),
            "stats": stats,
            "num_entries": stats.get("num_entries", 0)
        }
        
        self.snapshots[name] = snapshot
        
        logger.info(f"Snapshot created: {name}")
        
        return snapshot
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all snapshots.
        
        Returns:
            List of snapshot metadata
        """
        return list(self.snapshots.values())
    
    def get_snapshot(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get snapshot by name.
        
        Args:
            name: Snapshot name
            
        Returns:
            Snapshot metadata or None
        """
        return self.snapshots.get(name)
    
    def delete_snapshot(self, name: str) -> bool:
        """
        Delete snapshot.
        
        Args:
            name: Snapshot name
            
        Returns:
            True if deleted
        """
        if name in self.snapshots:
            del self.snapshots[name]
            logger.info(f"Snapshot deleted: {name}")
            return True
        return False

