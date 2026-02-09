"""
Backup and Recovery System for Flux2 Clothing Changer
======================================================

Automated backup and recovery for models and configurations.
"""

import json
import shutil
import tarfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging
import hashlib

logger = logging.getLogger(__name__)


class BackupRecovery:
    """Backup and recovery system."""
    
    def __init__(
        self,
        backup_dir: Path = Path("backups"),
        retention_days: int = 30,
        auto_backup: bool = True,
        backup_interval_hours: int = 24,
    ):
        """
        Initialize backup and recovery system.
        
        Args:
            backup_dir: Directory for backups
            retention_days: Days to retain backups
            auto_backup: Enable automatic backups
            backup_interval_hours: Hours between automatic backups
        """
        self.backup_dir = backup_dir
        self.retention_days = retention_days
        self.auto_backup = auto_backup
        self.backup_interval_hours = backup_interval_hours
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_backup_time: Optional[float] = None
        self.backup_history: List[Dict[str, Any]] = []
        
        # Load backup history
        self._load_backup_history()
    
    def create_backup(
        self,
        source_paths: List[Path],
        backup_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Create a backup of specified paths.
        
        Args:
            source_paths: List of paths to backup
            backup_name: Optional backup name (auto-generated if None)
            metadata: Optional metadata to include
            
        Returns:
            Path to backup file
        """
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_file = self.backup_dir / f"{backup_name}.tar.gz"
        
        try:
            with tarfile.open(backup_file, "w:gz") as tar:
                # Add files
                for source_path in source_paths:
                    if source_path.exists():
                        if source_path.is_file():
                            tar.add(source_path, arcname=source_path.name)
                        elif source_path.is_dir():
                            tar.add(source_path, arcname=source_path.name)
                
                # Add metadata
                if metadata:
                    metadata_str = json.dumps(metadata, indent=2)
                    metadata_bytes = metadata_str.encode("utf-8")
                    metadata_info = tarfile.TarInfo(name="backup_metadata.json")
                    metadata_info.size = len(metadata_bytes)
                    tar.addfile(metadata_info, fileobj=open(metadata_bytes, "rb"))
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_file)
            
            # Record backup
            backup_record = {
                "name": backup_name,
                "file": str(backup_file),
                "timestamp": datetime.now().timestamp(),
                "checksum": checksum,
                "source_paths": [str(p) for p in source_paths],
                "metadata": metadata or {},
                "size": backup_file.stat().st_size,
            }
            
            self.backup_history.append(backup_record)
            self.last_backup_time = backup_record["timestamp"]
            
            # Save backup history
            self._save_backup_history()
            
            logger.info(f"Created backup: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(
        self,
        backup_file: Path,
        target_dir: Path,
        verify: bool = True,
    ) -> bool:
        """
        Restore a backup.
        
        Args:
            backup_file: Path to backup file
            target_dir: Target directory for restoration
            verify: Verify checksum before restoring
            
        Returns:
            True if successful
        """
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        # Verify checksum
        if verify:
            expected_checksum = None
            for record in self.backup_history:
                if record["file"] == str(backup_file):
                    expected_checksum = record["checksum"]
                    break
            
            if expected_checksum:
                actual_checksum = self._calculate_checksum(backup_file)
                if actual_checksum != expected_checksum:
                    logger.error("Backup checksum verification failed")
                    return False
        
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(target_dir)
            
            logger.info(f"Restored backup to {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def list_backups(
        self,
        sort_by: str = "timestamp",
        reverse: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List all backups.
        
        Args:
            sort_by: Sort by field (timestamp, size, name)
            reverse: Reverse sort order
            
        Returns:
            List of backup records
        """
        backups = list(self.backup_history)
        
        if sort_by == "timestamp":
            backups.sort(key=lambda b: b["timestamp"], reverse=reverse)
        elif sort_by == "size":
            backups.sort(key=lambda b: b.get("size", 0), reverse=reverse)
        elif sort_by == "name":
            backups.sort(key=lambda b: b["name"], reverse=reverse)
        
        return backups
    
    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_name: Name of backup to delete
            
        Returns:
            True if successful
        """
        backup_record = None
        for record in self.backup_history:
            if record["name"] == backup_name:
                backup_record = record
                break
        
        if not backup_record:
            return False
        
        try:
            backup_file = Path(backup_record["file"])
            if backup_file.exists():
                backup_file.unlink()
            
            self.backup_history.remove(backup_record)
            self._save_backup_history()
            
            logger.info(f"Deleted backup: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """
        Clean up old backups based on retention policy.
        
        Returns:
            Number of backups deleted
        """
        cutoff_time = (datetime.now() - timedelta(days=self.retention_days)).timestamp()
        
        deleted = 0
        to_delete = [
            record for record in self.backup_history
            if record["timestamp"] < cutoff_time
        ]
        
        for record in to_delete:
            if self.delete_backup(record["name"]):
                deleted += 1
        
        logger.info(f"Cleaned up {deleted} old backups")
        return deleted
    
    def should_backup(self) -> bool:
        """Check if automatic backup should be performed."""
        if not self.auto_backup:
            return False
        
        if self.last_backup_time is None:
            return True
        
        time_since_backup = datetime.now().timestamp() - self.last_backup_time
        return time_since_backup >= (self.backup_interval_hours * 3600)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _load_backup_history(self) -> None:
        """Load backup history from disk."""
        history_file = self.backup_dir / "backup_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    self.backup_history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load backup history: {e}")
                self.backup_history = []
    
    def _save_backup_history(self) -> None:
        """Save backup history to disk."""
        history_file = self.backup_dir / "backup_history.json"
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(self.backup_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save backup history: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_size = sum(b.get("size", 0) for b in self.backup_history)
        
        return {
            "total_backups": len(self.backup_history),
            "total_size_mb": total_size / (1024 * 1024),
            "last_backup": self.last_backup_time,
            "auto_backup": self.auto_backup,
            "retention_days": self.retention_days,
        }


