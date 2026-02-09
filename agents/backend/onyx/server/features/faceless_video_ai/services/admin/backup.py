"""
Backup Service
Manages backups and recovery
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
import shutil
import json

logger = logging.getLogger(__name__)


class BackupService:
    """Manages backups and recovery"""
    
    def __init__(self, backup_dir: Optional[str] = None):
        self.backup_dir = Path(backup_dir) if backup_dir else Path("/tmp/faceless_video/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(
        self,
        include_videos: bool = True,
        include_metadata: bool = True,
        include_config: bool = True
    ) -> Dict[str, Any]:
        """
        Create system backup
        
        Args:
            include_videos: Include video files
            include_metadata: Include metadata
            include_config: Include configuration
            
        Returns:
            Backup information
        """
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        backup_info = {
            "backup_id": backup_id,
            "created_at": datetime.utcnow().isoformat(),
            "includes": {
                "videos": include_videos,
                "metadata": include_metadata,
                "config": include_config,
            },
            "files": [],
        }
        
        # Backup videos
        if include_videos:
            videos_dir = Path("/tmp/faceless_video/output")
            if videos_dir.exists():
                backup_videos_dir = backup_path / "videos"
                backup_videos_dir.mkdir()
                # Copy video files (in production, use more efficient method)
                for video_file in videos_dir.glob("*.mp4"):
                    shutil.copy2(video_file, backup_videos_dir)
                    backup_info["files"].append(f"videos/{video_file.name}")
        
        # Backup metadata
        if include_metadata:
            metadata_file = backup_path / "metadata.json"
            # In production, export from database
            metadata = {
                "backup_date": datetime.utcnow().isoformat(),
                "total_videos": 0,
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            backup_info["files"].append("metadata.json")
        
        # Backup config
        if include_config:
            config_file = backup_path / "config.json"
            # In production, export configuration
            config = {
                "backup_date": datetime.utcnow().isoformat(),
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            backup_info["files"].append("config.json")
        
        # Save backup info
        info_file = backup_path / "backup_info.json"
        with open(info_file, 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        logger.info(f"Backup created: {backup_id}")
        return backup_info
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all backups"""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                info_file = backup_dir / "backup_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        backup_info = json.load(f)
                        backups.append(backup_info)
        
        # Sort by date (newest first)
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return backups
    
    def get_backup(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get backup information"""
        backup_path = self.backup_dir / backup_id
        info_file = backup_path / "backup_info.json"
        
        if info_file.exists():
            with open(info_file, 'r') as f:
                return json.load(f)
        return None
    
    async def restore_backup(
        self,
        backup_id: str,
        restore_videos: bool = True,
        restore_metadata: bool = True
    ) -> bool:
        """
        Restore from backup
        
        Args:
            backup_id: Backup ID to restore
            restore_videos: Restore video files
            restore_metadata: Restore metadata
            
        Returns:
            True if successful
        """
        backup_path = self.backup_dir / backup_id
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        try:
            # Restore videos
            if restore_videos:
                backup_videos_dir = backup_path / "videos"
                if backup_videos_dir.exists():
                    videos_dir = Path("/tmp/faceless_video/output")
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    
                    for video_file in backup_videos_dir.glob("*.mp4"):
                        shutil.copy2(video_file, videos_dir)
            
            # Restore metadata (would restore to database in production)
            if restore_metadata:
                metadata_file = backup_path / "metadata.json"
                if metadata_file.exists():
                    # In production, import to database
                    pass
            
            logger.info(f"Backup restored: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Backup restore failed: {str(e)}")
            return False
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete backup"""
        backup_path = self.backup_dir / backup_id
        
        if not backup_path.exists():
            return False
        
        try:
            shutil.rmtree(backup_path)
            logger.info(f"Backup deleted: {backup_id}")
            return True
        except Exception as e:
            logger.error(f"Backup deletion failed: {str(e)}")
            return False


_backup_service: Optional[BackupService] = None


def get_backup_service(backup_dir: Optional[str] = None) -> BackupService:
    """Get backup service instance (singleton)"""
    global _backup_service
    if _backup_service is None:
        _backup_service = BackupService(backup_dir=backup_dir)
    return _backup_service

