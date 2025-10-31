"""
Gamma App - Backup Service
Advanced backup and restore service
"""

import os
import shutil
import tarfile
import zipfile
import json
import hashlib
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
import psutil
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Backup types"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    DATABASE = "database"
    FILES = "files"
    CONFIG = "config"

class BackupStatus(Enum):
    """Backup status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BackupConfig:
    """Backup configuration"""
    backup_type: BackupType
    source_paths: List[str]
    destination_path: str
    compression: bool = True
    encryption: bool = False
    encryption_key: Optional[str] = None
    exclude_patterns: List[str] = None
    include_patterns: List[str] = None
    max_backup_size: int = 1024 * 1024 * 1024  # 1GB
    retention_days: int = 30
    schedule: Optional[str] = None

@dataclass
class BackupInfo:
    """Backup information"""
    id: str
    name: str
    backup_type: BackupType
    status: BackupStatus
    source_paths: List[str]
    destination_path: str
    size: int
    checksum: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class BackupService:
    """Advanced backup service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_dir = Path(config.get('backup_dir', './backups'))
        self.temp_dir = Path(config.get('temp_dir', './temp'))
        self.s3_client = None
        self.backup_history: List[BackupInfo] = []
        
        # Create directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client if configured
        if config.get('s3', {}).get('enabled'):
            self._init_s3_client()
        
        # Load backup history
        self._load_backup_history()
    
    def _init_s3_client(self):
        """Initialize S3 client"""
        try:
            s3_config = self.config['s3']
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=s3_config['access_key'],
                aws_secret_access_key=s3_config['secret_key'],
                region_name=s3_config['region']
            )
        except Exception as e:
            logger.error(f"Error initializing S3 client: {e}")
    
    def _load_backup_history(self):
        """Load backup history from file"""
        try:
            history_file = self.backup_dir / 'backup_history.json'
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.backup_history = [
                        BackupInfo(**item) for item in data
                    ]
        except Exception as e:
            logger.error(f"Error loading backup history: {e}")
    
    def _save_backup_history(self):
        """Save backup history to file"""
        try:
            history_file = self.backup_dir / 'backup_history.json'
            data = [asdict(backup) for backup in self.backup_history]
            with open(history_file, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Error saving backup history: {e}")
    
    async def create_backup(
        self,
        name: str,
        backup_config: BackupConfig
    ) -> str:
        """Create a new backup"""
        try:
            backup_id = self._generate_backup_id()
            
            # Create backup info
            backup_info = BackupInfo(
                id=backup_id,
                name=name,
                backup_type=backup_config.backup_type,
                status=BackupStatus.PENDING,
                source_paths=backup_config.source_paths,
                destination_path=backup_config.destination_path,
                size=0,
                checksum="",
                created_at=datetime.now(),
                metadata={}
            )
            
            # Add to history
            self.backup_history.append(backup_info)
            self._save_backup_history()
            
            # Start backup process
            asyncio.create_task(self._process_backup(backup_info, backup_config))
            
            logger.info(f"Started backup: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    async def _process_backup(self, backup_info: BackupInfo, backup_config: BackupConfig):
        """Process backup"""
        try:
            backup_info.status = BackupStatus.RUNNING
            self._save_backup_history()
            
            # Create backup
            backup_path = await self._create_backup_file(backup_info, backup_config)
            
            # Calculate size and checksum
            backup_info.size = os.path.getsize(backup_path)
            backup_info.checksum = self._calculate_checksum(backup_path)
            
            # Upload to S3 if configured
            if self.s3_client and backup_config.destination_path.startswith('s3://'):
                await self._upload_to_s3(backup_path, backup_config.destination_path)
            
            # Update status
            backup_info.status = BackupStatus.COMPLETED
            backup_info.completed_at = datetime.now()
            self._save_backup_history()
            
            logger.info(f"Backup completed: {backup_info.id}")
            
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            self._save_backup_history()
            logger.error(f"Backup failed: {backup_info.id} - {e}")
    
    async def _create_backup_file(
        self,
        backup_info: BackupInfo,
        backup_config: BackupConfig
    ) -> str:
        """Create backup file"""
        try:
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{backup_info.name}_{timestamp}"
            
            if backup_config.compression:
                filename += ".tar.gz"
                backup_path = self.backup_dir / filename
                await self._create_tar_backup(backup_info, backup_config, backup_path)
            else:
                filename += ".zip"
                backup_path = self.backup_dir / filename
                await self._create_zip_backup(backup_info, backup_config, backup_path)
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup file: {e}")
            raise
    
    async def _create_tar_backup(
        self,
        backup_info: BackupInfo,
        backup_config: BackupConfig,
        backup_path: Path
    ):
        """Create tar.gz backup"""
        try:
            with tarfile.open(backup_path, "w:gz") as tar:
                for source_path in backup_info.source_paths:
                    if os.path.exists(source_path):
                        if os.path.isfile(source_path):
                            tar.add(source_path, arcname=os.path.basename(source_path))
                        elif os.path.isdir(source_path):
                            tar.add(source_path, arcname=os.path.basename(source_path))
        except Exception as e:
            logger.error(f"Error creating tar backup: {e}")
            raise
    
    async def _create_zip_backup(
        self,
        backup_info: BackupInfo,
        backup_config: BackupConfig,
        backup_path: Path
    ):
        """Create zip backup"""
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for source_path in backup_info.source_paths:
                    if os.path.exists(source_path):
                        if os.path.isfile(source_path):
                            zipf.write(source_path, os.path.basename(source_path))
                        elif os.path.isdir(source_path):
                            for root, dirs, files in os.walk(source_path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, source_path)
                                    zipf.write(file_path, arcname)
        except Exception as e:
            logger.error(f"Error creating zip backup: {e}")
            raise
    
    async def _upload_to_s3(self, local_path: str, s3_path: str):
        """Upload backup to S3"""
        try:
            if not self.s3_client:
                raise ValueError("S3 client not initialized")
            
            # Parse S3 path
            s3_path = s3_path.replace('s3://', '')
            bucket, key = s3_path.split('/', 1)
            
            # Upload file
            self.s3_client.upload_file(local_path, bucket, key)
            
            logger.info(f"Uploaded backup to S3: {s3_path}")
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID"""
        return f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    async def restore_backup(
        self,
        backup_id: str,
        destination_path: str,
        overwrite: bool = False
    ) -> bool:
        """Restore backup"""
        try:
            # Find backup info
            backup_info = self._get_backup_info(backup_id)
            if not backup_info:
                raise ValueError(f"Backup not found: {backup_id}")
            
            if backup_info.status != BackupStatus.COMPLETED:
                raise ValueError(f"Backup not completed: {backup_id}")
            
            # Find backup file
            backup_file = self._find_backup_file(backup_info)
            if not backup_file:
                raise ValueError(f"Backup file not found: {backup_id}")
            
            # Verify checksum
            if not self._verify_checksum(backup_file, backup_info.checksum):
                raise ValueError(f"Backup checksum verification failed: {backup_id}")
            
            # Restore backup
            await self._restore_backup_file(backup_file, destination_path, overwrite)
            
            logger.info(f"Backup restored: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def _get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Get backup info by ID"""
        for backup in self.backup_history:
            if backup.id == backup_id:
                return backup
        return None
    
    def _find_backup_file(self, backup_info: BackupInfo) -> Optional[str]:
        """Find backup file"""
        try:
            # Look for backup file in backup directory
            for file in self.backup_dir.glob(f"*{backup_info.name}*"):
                if file.is_file():
                    return str(file)
            return None
        except Exception as e:
            logger.error(f"Error finding backup file: {e}")
            return None
    
    def _verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """Verify file checksum"""
        try:
            actual_checksum = self._calculate_checksum(file_path)
            return actual_checksum == expected_checksum
        except Exception as e:
            logger.error(f"Error verifying checksum: {e}")
            return False
    
    async def _restore_backup_file(
        self,
        backup_file: str,
        destination_path: str,
        overwrite: bool
    ):
        """Restore backup file"""
        try:
            # Create destination directory
            os.makedirs(destination_path, exist_ok=True)
            
            # Extract backup
            if backup_file.endswith('.tar.gz'):
                with tarfile.open(backup_file, 'r:gz') as tar:
                    tar.extractall(destination_path)
            elif backup_file.endswith('.zip'):
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    zipf.extractall(destination_path)
            else:
                raise ValueError(f"Unsupported backup format: {backup_file}")
                
        except Exception as e:
            logger.error(f"Error restoring backup file: {e}")
            raise
    
    async def list_backups(self) -> List[BackupInfo]:
        """List all backups"""
        return self.backup_history.copy()
    
    async def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Get backup information"""
        return self._get_backup_info(backup_id)
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup"""
        try:
            # Find backup info
            backup_info = self._get_backup_info(backup_id)
            if not backup_info:
                return False
            
            # Find and delete backup file
            backup_file = self._find_backup_file(backup_info)
            if backup_file and os.path.exists(backup_file):
                os.remove(backup_file)
            
            # Remove from history
            self.backup_history = [b for b in self.backup_history if b.id != backup_id]
            self._save_backup_history()
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting backup: {e}")
            return False
    
    async def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Clean up old backups"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            deleted_count = 0
            
            # Find old backups
            old_backups = [
                backup for backup in self.backup_history
                if backup.created_at < cutoff_date
            ]
            
            # Delete old backups
            for backup in old_backups:
                if await self.delete_backup(backup.id):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            return 0
    
    async def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics"""
        try:
            stats = {
                'total_backups': len(self.backup_history),
                'completed_backups': 0,
                'failed_backups': 0,
                'total_size': 0,
                'by_type': {},
                'by_status': {}
            }
            
            for backup in self.backup_history:
                # Count by status
                if backup.status == BackupStatus.COMPLETED:
                    stats['completed_backups'] += 1
                elif backup.status == BackupStatus.FAILED:
                    stats['failed_backups'] += 1
                
                # Count by type
                backup_type = backup.backup_type.value
                stats['by_type'][backup_type] = stats['by_type'].get(backup_type, 0) + 1
                
                # Count by status
                status = backup.status.value
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                
                # Add size
                stats['total_size'] += backup.size
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting backup stats: {e}")
            return {}
    
    async def schedule_backup(
        self,
        name: str,
        backup_config: BackupConfig,
        schedule: str
    ) -> str:
        """Schedule backup (placeholder for future implementation)"""
        # This would integrate with a task scheduler like Celery or APScheduler
        logger.info(f"Scheduled backup: {name} with schedule: {schedule}")
        return f"scheduled_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def cancel_backup(self, backup_id: str) -> bool:
        """Cancel running backup"""
        try:
            backup_info = self._get_backup_info(backup_id)
            if not backup_info:
                return False
            
            if backup_info.status == BackupStatus.RUNNING:
                backup_info.status = BackupStatus.CANCELLED
                self._save_backup_history()
                logger.info(f"Backup cancelled: {backup_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling backup: {e}")
            return False
    
    async def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity"""
        try:
            backup_info = self._get_backup_info(backup_id)
            if not backup_info:
                return False
            
            backup_file = self._find_backup_file(backup_info)
            if not backup_file:
                return False
            
            return self._verify_checksum(backup_file, backup_info.checksum)
            
        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information for backup planning"""
        try:
            # Get disk usage
            disk_usage = psutil.disk_usage('/')
            
            # Get memory info
            memory = psutil.virtual_memory()
            
            return {
                'disk_total': disk_usage.total,
                'disk_used': disk_usage.used,
                'disk_free': disk_usage.free,
                'memory_total': memory.total,
                'memory_used': memory.used,
                'memory_free': memory.available,
                'cpu_count': psutil.cpu_count(),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}

























