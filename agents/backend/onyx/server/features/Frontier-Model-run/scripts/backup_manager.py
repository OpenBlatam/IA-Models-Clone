#!/usr/bin/env python3
"""
Advanced Backup and Recovery System for Frontier Model Training
Provides comprehensive backup, recovery, versioning, and disaster recovery capabilities.
"""

import os
import json
import shutil
import tarfile
import zipfile
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import psutil
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import schedule
import boto3
from botocore.exceptions import ClientError
import paramiko
from cryptography.fernet import Fernet
import schedule
import asyncio
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

console = Console()

class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StorageType(Enum):
    """Storage types for backups."""
    LOCAL = "local"
    S3 = "s3"
    FTP = "ftp"
    SFTP = "sftp"
    AZURE = "azure"
    GCP = "gcp"

@dataclass
class BackupConfig:
    """Backup configuration."""
    backup_name: str
    source_paths: List[str]
    destination_path: str
    backup_type: BackupType = BackupType.FULL
    storage_type: StorageType = StorageType.LOCAL
    compression: bool = True
    encryption: bool = True
    encryption_key: Optional[str] = None
    retention_days: int = 30
    max_backups: int = 10
    schedule: Optional[str] = None  # Cron expression
    exclude_patterns: List[str] = None
    include_patterns: List[str] = None
    verify_backup: bool = True
    parallel_workers: int = 4
    chunk_size: int = 1024 * 1024  # 1MB
    cloud_config: Dict[str, Any] = None

@dataclass
class BackupMetadata:
    """Backup metadata."""
    backup_id: str
    backup_name: str
    backup_type: BackupType
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: BackupStatus = BackupStatus.PENDING
    source_paths: List[str] = None
    destination_path: str = None
    file_count: int = 0
    total_size: int = 0
    compressed_size: int = 0
    checksum: Optional[str] = None
    error_message: Optional[str] = None
    dependencies: List[str] = None

@dataclass
class RecoveryConfig:
    """Recovery configuration."""
    backup_id: str
    destination_path: str
    restore_type: str = "full"  # full, selective, point_in_time
    selective_paths: List[str] = None
    point_in_time: Optional[datetime] = None
    verify_restore: bool = True
    overwrite_existing: bool = False

class FileWatcher(FileSystemEventHandler):
    """File system watcher for real-time backup."""
    
    def __init__(self, backup_manager):
        self.backup_manager = backup_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self.backup_manager._queue_incremental_backup(event.src_path)
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self.backup_manager._queue_incremental_backup(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self.backup_manager._log_file_deletion(event.src_path)

class BackupManager:
    """Main backup and recovery manager."""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metadata_db = self._init_metadata_db()
        self.encryption_key = self._get_encryption_key()
        self.file_index = {}
        self.backup_queue = []
        self.running_backups = {}
        
        # Initialize storage
        self.storage = self._init_storage()
        
        # Initialize file watcher
        self.file_watcher = FileWatcher(self)
        self.observer = Observer()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _init_metadata_db(self) -> str:
        """Initialize metadata database."""
        db_path = Path("./backup_metadata.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backups (
                    backup_id TEXT PRIMARY KEY,
                    backup_name TEXT NOT NULL,
                    backup_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    source_paths TEXT NOT NULL,
                    destination_path TEXT NOT NULL,
                    file_count INTEGER DEFAULT 0,
                    total_size INTEGER DEFAULT 0,
                    compressed_size INTEGER DEFAULT 0,
                    checksum TEXT,
                    error_message TEXT,
                    dependencies TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_index (
                    file_path TEXT PRIMARY KEY,
                    backup_id TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    modified_time TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    FOREIGN KEY (backup_id) REFERENCES backups (backup_id)
                )
            """)
        
        return str(db_path)
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key."""
        if self.config.encryption_key:
            return self.config.encryption_key.encode()
        else:
            return Fernet.generate_key()
    
    def _init_storage(self):
        """Initialize storage backend."""
        if self.config.storage_type == StorageType.S3:
            return S3Storage(self.config.cloud_config)
        elif self.config.storage_type == StorageType.FTP:
            return FTPStorage(self.config.cloud_config)
        elif self.config.storage_type == StorageType.SFTP:
            return SFTPStorage(self.config.cloud_config)
        else:
            return LocalStorage(self.config.destination_path)
    
    def _start_background_tasks(self):
        """Start background tasks."""
        # Start file watcher
        for source_path in self.config.source_paths:
            if Path(source_path).exists():
                self.observer.schedule(self.file_watcher, source_path, recursive=True)
        
        self.observer.start()
        
        # Start backup scheduler
        if self.config.schedule:
            schedule.every().day.at("02:00").do(self.create_backup)
        
        # Start background thread for scheduled tasks
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def _run_scheduler(self):
        """Run scheduler in background."""
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def create_backup(self, backup_type: Optional[BackupType] = None) -> str:
        """Create a new backup."""
        backup_type = backup_type or self.config.backup_type
        backup_id = self._generate_backup_id()
        
        # Create backup metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_name=self.config.backup_name,
            backup_type=backup_type,
            created_at=datetime.now(),
            source_paths=self.config.source_paths,
            destination_path=self.config.destination_path
        )
        
        # Save metadata
        self._save_backup_metadata(metadata)
        
        # Start backup in background
        backup_thread = threading.Thread(
            target=self._perform_backup,
            args=(metadata,),
            daemon=True
        )
        backup_thread.start()
        
        self.running_backups[backup_id] = metadata
        
        console.print(f"[blue]Backup started: {backup_id}[/blue]")
        return backup_id
    
    def _perform_backup(self, metadata: BackupMetadata):
        """Perform the actual backup."""
        try:
            metadata.status = BackupStatus.RUNNING
            self._update_backup_status(metadata)
            
            # Collect files to backup
            files_to_backup = self._collect_files(metadata)
            metadata.file_count = len(files_to_backup)
            
            # Calculate total size
            total_size = sum(file_info['size'] for file_info in files_to_backup)
            metadata.total_size = total_size
            
            # Create backup archive
            backup_path = self._create_backup_archive(metadata, files_to_backup)
            
            # Compress if enabled
            if self.config.compression:
                compressed_path = self._compress_backup(backup_path)
                metadata.compressed_size = Path(compressed_path).stat().st_size
                backup_path = compressed_path
            
            # Encrypt if enabled
            if self.config.encryption:
                encrypted_path = self._encrypt_backup(backup_path)
                backup_path = encrypted_path
            
            # Upload to storage
            self.storage.upload_backup(backup_path, metadata.backup_id)
            
            # Verify backup
            if self.config.verify_backup:
                self._verify_backup(metadata, files_to_backup)
            
            # Update metadata
            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.checksum = self._calculate_checksum(backup_path)
            
            self._update_backup_status(metadata)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            console.print(f"[green]Backup completed: {metadata.backup_id}[/green]")
            
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            self._update_backup_status(metadata)
            self.logger.error(f"Backup failed: {e}")
            console.print(f"[red]Backup failed: {e}[/red]")
        
        finally:
            # Remove from running backups
            if metadata.backup_id in self.running_backups:
                del self.running_backups[metadata.backup_id]
    
    def _collect_files(self, metadata: BackupMetadata) -> List[Dict[str, Any]]:
        """Collect files to backup."""
        files_to_backup = []
        
        for source_path in metadata.source_paths:
            if not Path(source_path).exists():
                self.logger.warning(f"Source path does not exist: {source_path}")
                continue
            
            for file_path in Path(source_path).rglob('*'):
                if file_path.is_file():
                    # Check exclude patterns
                    if self._should_exclude_file(file_path):
                        continue
                    
                    # Check include patterns
                    if not self._should_include_file(file_path):
                        continue
                    
                    file_info = {
                        'path': str(file_path),
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                        'checksum': self._calculate_file_checksum(file_path)
                    }
                    
                    files_to_backup.append(file_info)
        
        return files_to_backup
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded."""
        if not self.config.exclude_patterns:
            return False
        
        file_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if pattern in file_str or file_path.match(pattern):
                return True
        
        return False
    
    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included."""
        if not self.config.include_patterns:
            return True
        
        file_str = str(file_path)
        for pattern in self.config.include_patterns:
            if pattern in file_str or file_path.match(pattern):
                return True
        
        return False
    
    def _create_backup_archive(self, metadata: BackupMetadata, 
                              files_to_backup: List[Dict[str, Any]]) -> str:
        """Create backup archive."""
        archive_path = Path(self.config.destination_path) / f"{metadata.backup_id}.tar"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(archive_path, 'w') as tar:
            for file_info in files_to_backup:
                tar.add(file_info['path'], arcname=file_info['path'])
        
        return str(archive_path)
    
    def _compress_backup(self, backup_path: str) -> str:
        """Compress backup archive."""
        compressed_path = backup_path + '.gz'
        
        with open(backup_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original
        Path(backup_path).unlink()
        
        return compressed_path
    
    def _encrypt_backup(self, backup_path: str) -> str:
        """Encrypt backup archive."""
        encrypted_path = backup_path + '.encrypted'
        
        fernet = Fernet(self.encryption_key)
        
        with open(backup_path, 'rb') as f_in:
            encrypted_data = fernet.encrypt(f_in.read())
        
        with open(encrypted_path, 'wb') as f_out:
            f_out.write(encrypted_data)
        
        # Remove original
        Path(backup_path).unlink()
        
        return encrypted_path
    
    def _verify_backup(self, metadata: BackupMetadata, 
                      files_to_backup: List[Dict[str, Any]]):
        """Verify backup integrity."""
        # Download backup
        local_path = self.storage.download_backup(metadata.backup_id)
        
        # Decrypt if needed
        if self.config.encryption:
            local_path = self._decrypt_backup(local_path)
        
        # Decompress if needed
        if self.config.compression:
            local_path = self._decompress_backup(local_path)
        
        # Verify files
        with tarfile.open(local_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # Check if file exists in original list
                    original_file = next(
                        (f for f in files_to_backup if f['path'] == member.name),
                        None
                    )
                    
                    if original_file:
                        # Verify checksum
                        file_data = tar.extractfile(member)
                        if file_data:
                            checksum = hashlib.md5(file_data.read()).hexdigest()
                            if checksum != original_file['checksum']:
                                raise ValueError(f"Checksum mismatch for {member.name}")
        
        # Cleanup
        Path(local_path).unlink()
    
    def _decrypt_backup(self, encrypted_path: str) -> str:
        """Decrypt backup archive."""
        decrypted_path = encrypted_path.replace('.encrypted', '')
        
        fernet = Fernet(self.encryption_key)
        
        with open(encrypted_path, 'rb') as f_in:
            decrypted_data = fernet.decrypt(f_in.read())
        
        with open(decrypted_path, 'wb') as f_out:
            f_out.write(decrypted_data)
        
        return decrypted_path
    
    def _decompress_backup(self, compressed_path: str) -> str:
        """Decompress backup archive."""
        decompressed_path = compressed_path.replace('.gz', '')
        
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return decompressed_path
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate individual file checksum."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = secrets.token_hex(4)
        return f"{self.config.backup_name}_{timestamp}_{random_suffix}"
    
    def _save_backup_metadata(self, metadata: BackupMetadata):
        """Save backup metadata to database."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT INTO backups (
                    backup_id, backup_name, backup_type, created_at, completed_at,
                    status, source_paths, destination_path, file_count, total_size,
                    compressed_size, checksum, error_message, dependencies
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.backup_id,
                metadata.backup_name,
                metadata.backup_type.value,
                metadata.created_at.isoformat(),
                metadata.completed_at.isoformat() if metadata.completed_at else None,
                metadata.status.value,
                json.dumps(metadata.source_paths),
                metadata.destination_path,
                metadata.file_count,
                metadata.total_size,
                metadata.compressed_size,
                metadata.checksum,
                metadata.error_message,
                json.dumps(metadata.dependencies) if metadata.dependencies else None
            ))
    
    def _update_backup_status(self, metadata: BackupMetadata):
        """Update backup status in database."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                UPDATE backups SET
                    status = ?, completed_at = ?, file_count = ?, total_size = ?,
                    compressed_size = ?, checksum = ?, error_message = ?
                WHERE backup_id = ?
            """, (
                metadata.status.value,
                metadata.completed_at.isoformat() if metadata.completed_at else None,
                metadata.file_count,
                metadata.total_size,
                metadata.compressed_size,
                metadata.checksum,
                metadata.error_message,
                metadata.backup_id
            ))
    
    def _cleanup_old_backups(self):
        """Cleanup old backups based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        with sqlite3.connect(self.metadata_db) as conn:
            # Get old backups
            cursor = conn.execute("""
                SELECT backup_id FROM backups
                WHERE created_at < ? AND status = 'completed'
                ORDER BY created_at ASC
            """, (cutoff_date.isoformat(),))
            
            old_backups = cursor.fetchall()
            
            # Keep only max_backups
            if len(old_backups) > self.config.max_backups:
                backups_to_delete = old_backups[:-self.config.max_backups]
                
                for backup_id, in backups_to_delete:
                    # Delete from storage
                    self.storage.delete_backup(backup_id[0])
                    
                    # Delete from database
                    conn.execute("DELETE FROM backups WHERE backup_id = ?", (backup_id[0],))
                    conn.execute("DELETE FROM file_index WHERE backup_id = ?", (backup_id[0],))
    
    def restore_backup(self, config: RecoveryConfig) -> bool:
        """Restore from backup."""
        try:
            console.print(f"[blue]Starting restore from backup: {config.backup_id}[/blue]")
            
            # Get backup metadata
            metadata = self._get_backup_metadata(config.backup_id)
            if not metadata:
                raise ValueError(f"Backup not found: {config.backup_id}")
            
            # Download backup
            local_path = self.storage.download_backup(config.backup_id)
            
            # Decrypt if needed
            if self.config.encryption:
                local_path = self._decrypt_backup(local_path)
            
            # Decompress if needed
            if self.config.compression:
                local_path = self._decompress_backup(local_path)
            
            # Restore files
            self._restore_files(local_path, config)
            
            # Verify restore
            if config.verify_restore:
                self._verify_restore(config)
            
            # Cleanup
            Path(local_path).unlink()
            
            console.print(f"[green]Restore completed successfully[/green]")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            console.print(f"[red]Restore failed: {e}[/red]")
            return False
    
    def _get_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get backup metadata from database."""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM backups WHERE backup_id = ?
            """, (backup_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return BackupMetadata(
                backup_id=row[0],
                backup_name=row[1],
                backup_type=BackupType(row[2]),
                created_at=datetime.fromisoformat(row[3]),
                completed_at=datetime.fromisoformat(row[4]) if row[4] else None,
                status=BackupStatus(row[5]),
                source_paths=json.loads(row[6]),
                destination_path=row[7],
                file_count=row[8],
                total_size=row[9],
                compressed_size=row[10],
                checksum=row[11],
                error_message=row[12],
                dependencies=json.loads(row[13]) if row[13] else None
            )
    
    def _restore_files(self, archive_path: str, config: RecoveryConfig):
        """Restore files from archive."""
        with tarfile.open(archive_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # Check if file should be restored
                    if config.selective_paths:
                        if not any(member.name.startswith(path) for path in config.selective_paths):
                            continue
                    
                    # Extract file
                    member.name = member.name.replace(member.name.split('/')[0], 
                                                   config.destination_path, 1)
                    
                    if config.overwrite_existing or not Path(member.name).exists():
                        tar.extract(member)
    
    def _verify_restore(self, config: RecoveryConfig):
        """Verify restored files."""
        # This would implement verification logic
        pass
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all backups."""
        backups = []
        
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM backups ORDER BY created_at DESC
            """)
            
            for row in cursor.fetchall():
                metadata = BackupMetadata(
                    backup_id=row[0],
                    backup_name=row[1],
                    backup_type=BackupType(row[2]),
                    created_at=datetime.fromisoformat(row[3]),
                    completed_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    status=BackupStatus(row[5]),
                    source_paths=json.loads(row[6]),
                    destination_path=row[7],
                    file_count=row[8],
                    total_size=row[9],
                    compressed_size=row[10],
                    checksum=row[11],
                    error_message=row[12],
                    dependencies=json.loads(row[13]) if row[13] else None
                )
                backups.append(metadata)
        
        return backups
    
    def get_backup_status(self, backup_id: str) -> Optional[BackupStatus]:
        """Get backup status."""
        metadata = self._get_backup_metadata(backup_id)
        return metadata.status if metadata else None
    
    def cancel_backup(self, backup_id: str) -> bool:
        """Cancel running backup."""
        if backup_id in self.running_backups:
            metadata = self.running_backups[backup_id]
            metadata.status = BackupStatus.CANCELLED
            self._update_backup_status(metadata)
            del self.running_backups[backup_id]
            return True
        return False
    
    def _queue_incremental_backup(self, file_path: str):
        """Queue file for incremental backup."""
        self.backup_queue.append(file_path)
    
    def _log_file_deletion(self, file_path: str):
        """Log file deletion."""
        self.logger.info(f"File deleted: {file_path}")

class StorageInterface:
    """Storage interface for different backends."""
    
    def upload_backup(self, local_path: str, backup_id: str):
        """Upload backup to storage."""
        raise NotImplementedError
    
    def download_backup(self, backup_id: str) -> str:
        """Download backup from storage."""
        raise NotImplementedError
    
    def delete_backup(self, backup_id: str):
        """Delete backup from storage."""
        raise NotImplementedError

class LocalStorage(StorageInterface):
    """Local storage implementation."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload_backup(self, local_path: str, backup_id: str):
        """Upload backup to local storage."""
        dest_path = self.base_path / f"{backup_id}.backup"
        shutil.copy2(local_path, dest_path)
    
    def download_backup(self, backup_id: str) -> str:
        """Download backup from local storage."""
        source_path = self.base_path / f"{backup_id}.backup"
        if not source_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_id}")
        
        temp_path = f"/tmp/{backup_id}.backup"
        shutil.copy2(source_path, temp_path)
        return temp_path
    
    def delete_backup(self, backup_id: str):
        """Delete backup from local storage."""
        backup_path = self.base_path / f"{backup_id}.backup"
        if backup_path.exists():
            backup_path.unlink()

class S3Storage(StorageInterface):
    """S3 storage implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.get('access_key'),
            aws_secret_access_key=config.get('secret_key'),
            region_name=config.get('region', 'us-east-1')
        )
        self.bucket_name = config['bucket_name']
    
    def upload_backup(self, local_path: str, backup_id: str):
        """Upload backup to S3."""
        key = f"backups/{backup_id}.backup"
        self.s3_client.upload_file(local_path, self.bucket_name, key)
    
    def download_backup(self, backup_id: str) -> str:
        """Download backup from S3."""
        key = f"backups/{backup_id}.backup"
        temp_path = f"/tmp/{backup_id}.backup"
        
        try:
            self.s3_client.download_file(self.bucket_name, key, temp_path)
            return temp_path
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise FileNotFoundError(f"Backup not found: {backup_id}")
            raise
    
    def delete_backup(self, backup_id: str):
        """Delete backup from S3."""
        key = f"backups/{backup_id}.backup"
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)

class FTPStorage(StorageInterface):
    """FTP storage implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.host = config['host']
        self.username = config['username']
        self.password = config['password']
        self.port = config.get('port', 21)
    
    def upload_backup(self, local_path: str, backup_id: str):
        """Upload backup to FTP."""
        import ftplib
        
        with ftplib.FTP() as ftp:
            ftp.connect(self.host, self.port)
            ftp.login(self.username, self.password)
            
            with open(local_path, 'rb') as f:
                ftp.storbinary(f'STOR {backup_id}.backup', f)
    
    def download_backup(self, backup_id: str) -> str:
        """Download backup from FTP."""
        import ftplib
        
        temp_path = f"/tmp/{backup_id}.backup"
        
        with ftplib.FTP() as ftp:
            ftp.connect(self.host, self.port)
            ftp.login(self.username, self.password)
            
            with open(temp_path, 'wb') as f:
                ftp.retrbinary(f'RETR {backup_id}.backup', f.write)
        
        return temp_path
    
    def delete_backup(self, backup_id: str):
        """Delete backup from FTP."""
        import ftplib
        
        with ftplib.FTP() as ftp:
            ftp.connect(self.host, self.port)
            ftp.login(self.username, self.password)
            ftp.delete(f'{backup_id}.backup')

class SFTPStorage(StorageInterface):
    """SFTP storage implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.host = config['host']
        self.username = config['username']
        self.password = config['password']
        self.port = config.get('port', 22)
    
    def upload_backup(self, local_path: str, backup_id: str):
        """Upload backup to SFTP."""
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, self.port, self.username, self.password)
            
            with ssh.open_sftp() as sftp:
                sftp.put(local_path, f'/backups/{backup_id}.backup')
    
    def download_backup(self, backup_id: str) -> str:
        """Download backup from SFTP."""
        temp_path = f"/tmp/{backup_id}.backup"
        
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, self.port, self.username, self.password)
            
            with ssh.open_sftp() as sftp:
                sftp.get(f'/backups/{backup_id}.backup', temp_path)
        
        return temp_path
    
    def delete_backup(self, backup_id: str):
        """Delete backup from SFTP."""
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, self.port, self.username, self.password)
            
            with ssh.open_sftp() as sftp:
                sftp.remove(f'/backups/{backup_id}.backup')

def main():
    """Main function for backup CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup and Recovery System")
    parser.add_argument("--action", type=str,
                       choices=["create", "restore", "list", "status", "cancel"],
                       required=True, help="Action to perform")
    parser.add_argument("--backup-name", type=str, default="frontier-model",
                       help="Backup name")
    parser.add_argument("--source-paths", type=str, nargs="+",
                       help="Source paths to backup")
    parser.add_argument("--destination-path", type=str, default="./backups",
                       help="Destination path for backups")
    parser.add_argument("--backup-type", type=str,
                       choices=["full", "incremental", "differential"],
                       default="full", help="Type of backup")
    parser.add_argument("--storage-type", type=str,
                       choices=["local", "s3", "ftp", "sftp"],
                       default="local", help="Storage type")
    parser.add_argument("--backup-id", type=str, help="Backup ID for restore/status")
    parser.add_argument("--restore-path", type=str, help="Restore destination path")
    parser.add_argument("--schedule", type=str, help="Backup schedule (cron format)")
    parser.add_argument("--retention-days", type=int, default=30,
                       help="Backup retention in days")
    
    args = parser.parse_args()
    
    # Create backup configuration
    config = BackupConfig(
        backup_name=args.backup_name,
        source_paths=args.source_paths or ["./models", "./data"],
        destination_path=args.destination_path,
        backup_type=BackupType(args.backup_type),
        storage_type=StorageType(args.storage_type),
        schedule=args.schedule,
        retention_days=args.retention_days
    )
    
    # Create backup manager
    backup_manager = BackupManager(config)
    
    if args.action == "create":
        backup_id = backup_manager.create_backup()
        console.print(f"[green]Backup created: {backup_id}[/green]")
    
    elif args.action == "restore":
        if not args.backup_id or not args.restore_path:
            console.print("[red]Backup ID and restore path are required[/red]")
            return
        
        recovery_config = RecoveryConfig(
            backup_id=args.backup_id,
            destination_path=args.restore_path
        )
        
        success = backup_manager.restore_backup(recovery_config)
        if success:
            console.print("[green]Restore completed successfully[/green]")
        else:
            console.print("[red]Restore failed[/red]")
    
    elif args.action == "list":
        backups = backup_manager.list_backups()
        
        table = Table(title="Available Backups")
        table.add_column("Backup ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="blue")
        table.add_column("Size", style="red")
        
        for backup in backups:
            table.add_row(
                backup.backup_id,
                backup.backup_name,
                backup.backup_type.value,
                backup.status.value,
                backup.created_at.strftime("%Y-%m-%d %H:%M"),
                f"{backup.total_size / 1024 / 1024:.1f} MB"
            )
        
        console.print(table)
    
    elif args.action == "status":
        if not args.backup_id:
            console.print("[red]Backup ID is required[/red]")
            return
        
        status = backup_manager.get_backup_status(args.backup_id)
        if status:
            console.print(f"[blue]Backup status: {status.value}[/blue]")
        else:
            console.print("[red]Backup not found[/red]")
    
    elif args.action == "cancel":
        if not args.backup_id:
            console.print("[red]Backup ID is required[/red]")
            return
        
        success = backup_manager.cancel_backup(args.backup_id)
        if success:
            console.print("[green]Backup cancelled[/green]")
        else:
            console.print("[red]Backup not found or not running[/red]")

if __name__ == "__main__":
    main()
