"""
Backup Manager for Document Analyzer
======================================

Advanced backup and restore system for configurations and data.
"""

import asyncio
import logging
import shutil
import json
import zipfile
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class BackupInfo:
    """Backup information"""
    backup_id: str
    timestamp: datetime
    backup_type: str
    size_bytes: int = 0
    location: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class BackupManager:
    """Advanced backup manager"""
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backups: Dict[str, BackupInfo] = {}
        logger.info(f"BackupManager initialized. Backup dir: {backup_dir}")
    
    def create_backup(
        self,
        backup_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> BackupInfo:
        """Create a backup"""
        backup_id = f"{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / f"{backup_id}.json"
        
        # Save backup data
        backup_data = {
            "backup_id": backup_id,
            "timestamp": datetime.now().isoformat(),
            "backup_type": backup_type,
            "data": data,
            "metadata": metadata or {}
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        size_bytes = backup_path.stat().st_size
        
        backup_info = BackupInfo(
            backup_id=backup_id,
            timestamp=datetime.now(),
            backup_type=backup_type,
            size_bytes=size_bytes,
            location=str(backup_path),
            metadata=metadata or {}
        )
        
        self.backups[backup_id] = backup_info
        logger.info(f"Created backup: {backup_id} ({size_bytes} bytes)")
        
        return backup_info
    
    def create_full_backup(
        self,
        config: Dict[str, Any],
        models_dir: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> BackupInfo:
        """Create a full backup including files"""
        backup_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / f"{backup_id}.zip"
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add config
            config_data = json.dumps(config, indent=2)
            zipf.writestr("config.json", config_data)
            
            # Add models if provided
            if models_dir and Path(models_dir).exists():
                for model_file in Path(models_dir).rglob("*"):
                    if model_file.is_file():
                        zipf.write(model_file, f"models/{model_file.name}")
            
            # Add cache if provided
            if cache_dir and Path(cache_dir).exists():
                for cache_file in Path(cache_dir).rglob("*"):
                    if cache_file.is_file():
                        zipf.write(cache_file, f"cache/{cache_file.name}")
        
        size_bytes = backup_path.stat().st_size
        
        backup_info = BackupInfo(
            backup_id=backup_id,
            timestamp=datetime.now(),
            backup_type="full",
            size_bytes=size_bytes,
            location=str(backup_path),
            metadata={
                "has_models": models_dir is not None,
                "has_cache": cache_dir is not None
            }
        )
        
        self.backups[backup_id] = backup_info
        logger.info(f"Created full backup: {backup_id} ({size_bytes} bytes)")
        
        return backup_info
    
    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore from backup"""
        if backup_id not in self.backups:
            raise ValueError(f"Backup {backup_id} not found")
        
        backup_info = self.backups[backup_id]
        backup_path = Path(backup_info.location)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        if backup_path.suffix == '.zip':
            # Full backup restore
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                config_data = zipf.read("config.json")
                return json.loads(config_data)
        else:
            # JSON backup restore
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
                return backup_data.get("data", {})
    
    def list_backups(self, backup_type: Optional[str] = None) -> List[BackupInfo]:
        """List available backups"""
        backups = list(self.backups.values())
        
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        if backup_id not in self.backups:
            return False
        
        backup_info = self.backups[backup_id]
        backup_path = Path(backup_info.location)
        
        if backup_path.exists():
            backup_path.unlink()
        
        del self.backups[backup_id]
        logger.info(f"Deleted backup: {backup_id}")
        
        return True
    
    def cleanup_old_backups(self, keep_last: int = 10):
        """Cleanup old backups, keeping only the last N"""
        backups = self.list_backups()
        
        if len(backups) > keep_last:
            to_delete = backups[keep_last:]
            for backup in to_delete:
                self.delete_backup(backup.backup_id)
            
            logger.info(f"Cleaned up {len(to_delete)} old backups")

# Global instance
backup_manager = BackupManager()
















