"""
Model Backup and Recovery
Backup and recovery system for models and data
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from datetime import datetime
import shutil
import json
import hashlib

logger = logging.getLogger(__name__)


class ModelBackup:
    """
    Backup and recovery system for models
    """
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_metadata: Dict[str, Dict[str, Any]] = {}
    
    def create_backup(
        self,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        backup_name: Optional[str] = None
    ) -> str:
        """Create backup of model"""
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Generate backup name
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{model_file.stem}_{timestamp}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        backup_model_path = backup_path / model_file.name
        shutil.copy2(model_file, backup_model_path)
        
        # Save metadata
        backup_metadata = {
            "original_path": str(model_path),
            "backup_path": str(backup_model_path),
            "backup_name": backup_name,
            "created_at": datetime.now().isoformat(),
            "file_size": model_file.stat().st_size,
            "checksum": self._calculate_checksum(model_path),
            "metadata": metadata or {}
        }
        
        metadata_file = backup_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(backup_metadata, f, indent=2)
        
        self.backup_metadata[backup_name] = backup_metadata
        
        logger.info(f"Created backup: {backup_name}")
        return backup_name
    
    def restore_backup(
        self,
        backup_name: str,
        restore_path: Optional[str] = None
    ) -> str:
        """Restore model from backup"""
        if backup_name not in self.backup_metadata:
            # Try to load from disk
            backup_path = self.backup_dir / backup_name
            if not backup_path.exists():
                raise ValueError(f"Backup not found: {backup_name}")
            
            metadata_file = backup_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    backup_metadata = json.load(f)
            else:
                raise ValueError(f"Backup metadata not found: {backup_name}")
        else:
            backup_metadata = self.backup_metadata[backup_name]
        
        backup_model_path = Path(backup_metadata["backup_path"])
        if not backup_model_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_model_path}")
        
        # Determine restore path
        if restore_path is None:
            restore_path = backup_metadata["original_path"]
        
        restore_path_obj = Path(restore_path)
        restore_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy backup to restore path
        shutil.copy2(backup_model_path, restore_path_obj)
        
        # Verify checksum
        restored_checksum = self._calculate_checksum(restore_path)
        if restored_checksum != backup_metadata["checksum"]:
            logger.warning("Checksum mismatch after restore")
        
        logger.info(f"Restored backup {backup_name} to {restore_path}")
        return restore_path
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all backups"""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            backups.append(metadata)
                    except Exception as e:
                        logger.error(f"Error reading backup metadata: {str(e)}")
        
        # Sort by creation date
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return backups
    
    def delete_backup(self, backup_name: str):
        """Delete backup"""
        backup_path = self.backup_dir / backup_name
        if backup_path.exists():
            shutil.rmtree(backup_path)
            if backup_name in self.backup_metadata:
                del self.backup_metadata[backup_name]
            logger.info(f"Deleted backup: {backup_name}")
        else:
            raise ValueError(f"Backup not found: {backup_name}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

