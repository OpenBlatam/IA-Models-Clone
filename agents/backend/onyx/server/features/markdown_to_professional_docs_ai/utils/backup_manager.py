"""Backup and recovery utilities"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage backups of documents and configurations"""
    
    def __init__(self, backup_dir: Optional[str] = None):
        """
        Initialize backup manager
        
        Args:
            backup_dir: Backup directory
        """
        if backup_dir is None:
            from config import settings
            backup_dir = settings.temp_dir + "/backups"
        
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        source_path: str,
        backup_name: Optional[str] = None
    ) -> str:
        """
        Create backup of a file or directory
        
        Args:
            source_path: Path to backup
            backup_name: Optional backup name
            
        Returns:
            Backup path
        """
        try:
            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"Source not found: {source_path}")
            
            # Generate backup name
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{source.name}_{timestamp}"
            
            backup_path = self.backup_dir / backup_name
            
            # Copy source
            if source.is_file():
                shutil.copy2(source, backup_path)
            else:
                shutil.copytree(source, backup_path)
            
            # Save metadata
            metadata = {
                "backup_name": backup_name,
                "source_path": str(source_path),
                "backup_path": str(backup_path),
                "created_at": datetime.now().isoformat(),
                "type": "file" if source.is_file() else "directory"
            }
            
            metadata_file = backup_path.with_suffix('.meta.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return str(backup_path)
        except Exception as e:
            logger.error(f"Backup error: {e}")
            raise
    
    def list_backups(
        self,
        source_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all backups
        
        Args:
            source_path: Optional source path to filter
            
        Returns:
            List of backup information
        """
        backups = []
        
        for item in self.backup_dir.iterdir():
            if item.is_file() and item.suffix == '.meta.json':
                with open(item, 'r') as f:
                    metadata = json.load(f)
                    
                    if source_path is None or metadata.get("source_path") == source_path:
                        backups.append(metadata)
        
        # Sort by creation date
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return backups
    
    def restore_backup(
        self,
        backup_name: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Restore from backup
        
        Args:
            backup_name: Backup name
            output_path: Output path (default: original path)
            
        Returns:
            Path to restored file or None
        """
        try:
            backup_path = self.backup_dir / backup_name
            
            if not backup_path.exists():
                # Try to find metadata
                metadata_file = self.backup_dir / f"{backup_name}.meta.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    backup_path = Path(metadata["backup_path"])
                else:
                    return None
            
            if output_path is None:
                # Get original path from metadata
                metadata_file = backup_path.with_suffix('.meta.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    output_path = metadata.get("source_path")
                else:
                    output_path = str(backup_path)
            
            # Restore
            if backup_path.is_file():
                shutil.copy2(backup_path, output_path)
            else:
                if Path(output_path).exists():
                    shutil.rmtree(output_path)
                shutil.copytree(backup_path, output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Restore error: {e}")
            return None
    
    def delete_backup(
        self,
        backup_name: str
    ) -> bool:
        """
        Delete a backup
        
        Args:
            backup_name: Backup name
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            backup_path = self.backup_dir / backup_name
            metadata_file = backup_path.with_suffix('.meta.json')
            
            if backup_path.exists():
                if backup_path.is_file():
                    backup_path.unlink()
                else:
                    shutil.rmtree(backup_path)
            
            if metadata_file.exists():
                metadata_file.unlink()
            
            return True
        except Exception as e:
            logger.error(f"Delete backup error: {e}")
            return False
    
    def cleanup_old_backups(
        self,
        days: int = 30
    ) -> int:
        """
        Clean up backups older than specified days
        
        Args:
            days: Number of days
            
        Returns:
            Number of backups deleted
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted = 0
        
        for item in self.backup_dir.iterdir():
            if item.is_file() and item.suffix == '.meta.json':
                with open(item, 'r') as f:
                    metadata = json.load(f)
                
                created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                
                if created_at < cutoff_date:
                    backup_name = metadata.get("backup_name")
                    if self.delete_backup(backup_name):
                        deleted += 1
        
        return deleted


# Global backup manager
_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """Get global backup manager"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager

