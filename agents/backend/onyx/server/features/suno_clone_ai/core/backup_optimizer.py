"""
Backup and Recovery Optimizations

Optimizations for:
- Incremental backups
- Fast recovery
- Backup compression
- Backup verification
- Automated backups
"""

import logging
import os
import shutil
import gzip
import tarfile
from typing import Optional, Dict, Any, List
from pathlib import Path
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class BackupManager:
    """Optimized backup management."""
    
    def __init__(self, backup_dir: str = "./backups"):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Backup directory
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_metadata: Dict[str, Dict[str, Any]] = {}
    
    def create_backup(
        self,
        source_path: str,
        backup_name: Optional[str] = None,
        compress: bool = True,
        incremental: bool = True
    ) -> str:
        """
        Create backup.
        
        Args:
            source_path: Source path to backup
            backup_name: Backup name (auto-generated if None)
            compress: Compress backup
            incremental: Use incremental backup
            
        Returns:
            Backup file path
        """
        source = Path(source_path)
        
        if not source.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        
        # Generate backup name
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_path = self.backup_dir / backup_name
        
        # Create backup
        if source.is_file():
            # File backup
            if compress:
                backup_path = backup_path.with_suffix('.tar.gz')
                with tarfile.open(backup_path, 'w:gz') as tar:
                    tar.add(source, arcname=source.name)
            else:
                shutil.copy2(source, backup_path)
        else:
            # Directory backup
            if compress:
                backup_path = backup_path.with_suffix('.tar.gz')
                with tarfile.open(backup_path, 'w:gz') as tar:
                    tar.add(source, arcname=source.name)
            else:
                shutil.copytree(source, backup_path, dirs_exist_ok=True)
        
        # Store metadata
        backup_hash = self._calculate_hash(backup_path)
        self.backup_metadata[str(backup_path)] = {
            'source': str(source),
            'created_at': datetime.now().isoformat(),
            'size': backup_path.stat().st_size,
            'hash': backup_hash,
            'compressed': compress
        }
        
        logger.info(f"Backup created: {backup_path}")
        return str(backup_path)
    
    def restore_backup(
        self,
        backup_path: str,
        target_path: str,
        verify: bool = True
    ) -> None:
        """
        Restore from backup.
        
        Args:
            backup_path: Backup file path
            target_path: Target restoration path
            verify: Verify backup integrity
        """
        backup = Path(backup_path)
        target = Path(target_path)
        
        if not backup.exists():
            raise ValueError(f"Backup does not exist: {backup_path}")
        
        # Verify if requested
        if verify:
            if not self.verify_backup(backup_path):
                raise ValueError("Backup verification failed")
        
        # Restore
        if backup.suffix == '.gz' or backup.suffixes == ['.tar', '.gz']:
            # Compressed backup
            with tarfile.open(backup, 'r:gz') as tar:
                tar.extractall(target.parent)
        else:
            # Regular backup
            if backup.is_file():
                shutil.copy2(backup, target)
            else:
                shutil.copytree(backup, target, dirs_exist_ok=True)
        
        logger.info(f"Backup restored to: {target_path}")
    
    def verify_backup(self, backup_path: str) -> bool:
        """
        Verify backup integrity.
        
        Args:
            backup_path: Backup file path
            
        Returns:
            True if backup is valid
        """
        backup = Path(backup_path)
        
        if not backup.exists():
            return False
        
        # Check if metadata exists
        if str(backup) in self.backup_metadata:
            expected_hash = self.backup_metadata[str(backup)]['hash']
            actual_hash = self._calculate_hash(backup)
            return expected_hash == actual_hash
        
        # Basic check: file exists and has size
        return backup.stat().st_size > 0
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate file hash."""
        hash_md5 = hashlib.md5()
        
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            # For directories, hash all files
            for file in file_path.rglob('*'):
                if file.is_file():
                    with open(file, 'rb') as f:
                        hash_md5.update(f.read())
        
        return hash_md5.hexdigest()
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all backups."""
        backups = []
        
        for backup_file in self.backup_dir.glob('backup_*'):
            if backup_file.is_file():
                metadata = self.backup_metadata.get(str(backup_file), {})
                backups.append({
                    'path': str(backup_file),
                    'size': backup_file.stat().st_size,
                    'created_at': metadata.get('created_at'),
                    'source': metadata.get('source')
                })
        
        return sorted(backups, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def cleanup_old_backups(self, keep_count: int = 10) -> None:
        """
        Cleanup old backups, keeping only the most recent.
        
        Args:
            keep_count: Number of backups to keep
        """
        backups = self.list_backups()
        
        if len(backups) > keep_count:
            to_delete = backups[keep_count:]
            for backup in to_delete:
                backup_path = Path(backup['path'])
                backup_path.unlink()
                if str(backup_path) in self.backup_metadata:
                    del self.backup_metadata[str(backup_path)]
                logger.info(f"Deleted old backup: {backup_path}")


class IncrementalBackup:
    """Incremental backup optimization."""
    
    def __init__(self, base_backup_path: str):
        """
        Initialize incremental backup.
        
        Args:
            base_backup_path: Path to base backup
        """
        self.base_backup_path = Path(base_backup_path)
        self.changes: List[Dict[str, Any]] = []
    
    def track_change(self, file_path: str, change_type: str) -> None:
        """
        Track file change.
        
        Args:
            file_path: File path
            change_type: Type of change (created, modified, deleted)
        """
        self.changes.append({
            'path': file_path,
            'type': change_type,
            'timestamp': datetime.now().isoformat()
        })
    
    def create_incremental(self, output_path: str) -> str:
        """
        Create incremental backup.
        
        Args:
            output_path: Output backup path
            
        Returns:
            Backup file path
        """
        # Only backup changed files
        changed_files = [
            change['path'] for change in self.changes
            if change['type'] in ['created', 'modified']
        ]
        
        if not changed_files:
            return None
        
        output = Path(output_path)
        with tarfile.open(output, 'w:gz') as tar:
            for file_path in changed_files:
                path = Path(file_path)
                if path.exists():
                    tar.add(path, arcname=path.name)
        
        return str(output)








