"""
Backup Manager for Color Grading AI
====================================

Manages backups of configurations, presets, and history.
"""

import logging
import json
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import zipfile
import tarfile

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manages backups.
    
    Features:
    - Create backups
    - Restore from backups
    - List backups
    - Automatic backups
    """
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory for backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        source_dirs: List[str],
        backup_name: Optional[str] = None,
        include_history: bool = True,
        include_presets: bool = True,
        include_templates: bool = True
    ) -> str:
        """
        Create a backup.
        
        Args:
            source_dirs: Directories to backup
            backup_name: Optional backup name
            include_history: Include history
            include_presets: Include presets
            include_templates: Include templates
            
        Returns:
            Path to backup file
        """
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Backup directories
            for source_dir in source_dirs:
                source_path = Path(source_dir)
                if source_path.exists():
                    if source_path.is_dir():
                        for file_path in source_path.rglob("*"):
                            if file_path.is_file():
                                arcname = file_path.relative_to(source_path.parent)
                                zipf.write(file_path, arcname)
                    else:
                        zipf.write(source_path, source_path.name)
            
            # Backup metadata
            metadata = {
                "backup_name": backup_name,
                "created_at": datetime.now().isoformat(),
                "includes": {
                    "history": include_history,
                    "presets": include_presets,
                    "templates": include_templates,
                }
            }
            zipf.writestr("backup_metadata.json", json.dumps(metadata, indent=2))
        
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)
    
    def restore_backup(
        self,
        backup_path: str,
        target_dir: str,
        overwrite: bool = False
    ) -> bool:
        """
        Restore from backup.
        
        Args:
            backup_path: Path to backup file
            target_dir: Target directory for restoration
            overwrite: Overwrite existing files
            
        Returns:
            True if successful
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(backup_file, "r") as zipf:
                # Extract all files
                zipf.extractall(target_path)
            
            logger.info(f"Restored backup to: {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List available backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(backup_file, "r") as zipf:
                    # Try to read metadata
                    try:
                        metadata_str = zipf.read("backup_metadata.json").decode()
                        metadata = json.loads(metadata_str)
                    except:
                        metadata = {
                            "backup_name": backup_file.stem,
                            "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                        }
                
                backups.append({
                    "name": backup_file.name,
                    "path": str(backup_file),
                    "size_mb": backup_file.stat().st_size / (1024 * 1024),
                    "created_at": metadata.get("created_at"),
                    "metadata": metadata,
                })
            except Exception as e:
                logger.error(f"Error reading backup {backup_file}: {e}")
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return backups
    
    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_name: Backup name or path
            
        Returns:
            True if deleted
        """
        backup_path = Path(backup_name)
        if not backup_path.is_absolute():
            backup_path = self.backup_dir / backup_name
        
        if backup_path.exists():
            backup_path.unlink()
            logger.info(f"Deleted backup: {backup_path}")
            return True
        
        return False




