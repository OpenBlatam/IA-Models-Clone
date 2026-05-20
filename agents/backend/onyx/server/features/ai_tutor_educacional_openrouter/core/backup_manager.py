"""
Backup and restore management system.
"""

import json
import logging
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import zipfile

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manages backups and restorations of system data.
    """
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory to store backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        data: Dict[str, Any],
        backup_name: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Create a backup of data.
        
        Args:
            data: Data to backup
            backup_name: Optional custom backup name
            include_metadata: Include metadata in backup
        
        Returns:
            Path to backup file
        """
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        # Create temporary directory for backup contents
        temp_dir = self.backup_dir / backup_name
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Save data as JSON
            data_file = temp_dir / "data.json"
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            # Add metadata if requested
            if include_metadata:
                metadata = {
                    "backup_name": backup_name,
                    "created_at": datetime.now().isoformat(),
                    "data_size": len(json.dumps(data)),
                    "items_count": len(data) if isinstance(data, dict) else 0
                }
                
                metadata_file = temp_dir / "metadata.json"
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            
            # Create zip archive
            with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file in temp_dir.rglob("*"):
                    if file.is_file():
                        zipf.write(file, file.relative_to(temp_dir))
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            # Clean up on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
    
    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """
        Restore data from backup.
        
        Args:
            backup_path: Path to backup file
        
        Returns:
            Restored data
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Extract to temporary directory
        temp_dir = self.backup_dir / f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract zip archive
            with zipfile.ZipFile(backup_file, "r") as zipf:
                zipf.extractall(temp_dir)
            
            # Load data
            data_file = temp_dir / "data.json"
            if not data_file.exists():
                raise ValueError("Backup file does not contain data.json")
            
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            logger.info(f"Backup restored from: {backup_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            # Clean up on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("backup_*.zip"):
            try:
                stat = backup_file.stat()
                backups.append({
                    "name": backup_file.stem,
                    "path": str(backup_file),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                logger.warning(f"Error reading backup {backup_file}: {e}")
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        
        return backups
    
    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_name: Name of backup to delete
        
        Returns:
            True if deleted successfully
        """
        backup_file = self.backup_dir / f"{backup_name}.zip"
        
        if not backup_file.exists():
            logger.warning(f"Backup not found: {backup_name}")
            return False
        
        try:
            backup_file.unlink()
            logger.info(f"Backup deleted: {backup_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False




