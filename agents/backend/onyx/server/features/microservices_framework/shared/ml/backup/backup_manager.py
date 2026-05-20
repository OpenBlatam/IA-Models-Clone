"""
Backup Manager
Backup and restore utilities for models and checkpoints.
"""

import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage backups of models and checkpoints."""
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        source_path: str,
        backup_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create backup of a directory or file."""
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
            shutil.copytree(source, backup_path, dirs_exist_ok=True)
        
        # Save metadata
        if metadata:
            metadata_path = backup_path / "backup_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({
                    "backup_name": backup_name,
                    "source_path": str(source),
                    "created_at": datetime.now().isoformat(),
                    **metadata,
                }, f, indent=2)
        
        logger.info(f"Backup created: {backup_path}")
        return str(backup_path)
    
    def restore_backup(
        self,
        backup_name: str,
        target_path: str,
        overwrite: bool = False,
    ) -> bool:
        """Restore backup to target path."""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_name}")
        
        target = Path(target_path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Target exists: {target_path}")
        
        # Remove target if exists
        if target.exists():
            if target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        
        # Restore
        if backup_path.is_file():
            shutil.copy2(backup_path, target)
        else:
            shutil.copytree(backup_path, target)
        
        logger.info(f"Backup restored: {backup_name} -> {target_path}")
        return True
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all backups."""
        backups = []
        
        for item in self.backup_dir.iterdir():
            if item.is_dir() or item.suffix == ".json":
                metadata_path = item / "backup_metadata.json" if item.is_dir() else item
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        backups.append(metadata)
                    except Exception:
                        pass
        
        return sorted(backups, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def delete_backup(self, backup_name: str) -> bool:
        """Delete a backup."""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            return False
        
        if backup_path.is_file():
            backup_path.unlink()
        else:
            shutil.rmtree(backup_path)
        
        logger.info(f"Backup deleted: {backup_name}")
        return True
    
    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Clean up old backups, keeping only the most recent ones."""
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return 0
        
        deleted_count = 0
        for backup in backups[keep_count:]:
            if self.delete_backup(backup["backup_name"]):
                deleted_count += 1
        
        return deleted_count



