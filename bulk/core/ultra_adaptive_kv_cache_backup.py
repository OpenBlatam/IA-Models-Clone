"""
Backup and Restore System for Ultra-Adaptive K/V Cache Engine
"""

import json
import pickle
import time
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import tarfile
import gzip

try:
    from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine
except ImportError:
    UltraAdaptiveKVCacheEngine = None

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage backups of engine state and cache."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine, backup_dir: str = "./backups"):
        self.engine = engine
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, name: Optional[str] = None, 
                     include_cache: bool = True,
                     include_sessions: bool = True,
                     include_checkpoints: bool = True,
                     compress: bool = True) -> str:
        """
        Create a backup of engine state.
        
        Returns:
            Path to backup file
        """
        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"backup_{timestamp}"
        
        backup_path = self.backup_dir / name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        backup_data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'engine_config': None,
            'performance_metrics': None,
            'gpu_workloads': None
        }
        
        try:
            # Backup engine stats
            stats = self.engine.get_performance_stats()
            backup_data['performance_metrics'] = stats.get('engine_stats', {})
            backup_data['gpu_workloads'] = stats.get('gpu_workloads', {})
            
            # Backup config
            if hasattr(self.engine, 'config'):
                from dataclasses import asdict
                try:
                    backup_data['engine_config'] = asdict(self.engine.config)
                except:
                    backup_data['engine_config'] = str(self.engine.config)
            
            # Save metadata
            metadata_file = backup_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            # Backup cache if enabled
            if include_cache and hasattr(self.engine, 'cache_path') and self.engine.cache_path:
                cache_backup_path = backup_path / "cache"
                if Path(self.engine.cache_path).exists():
                    shutil.copytree(self.engine.cache_path, cache_backup_path, dirs_exist_ok=True)
            
            # Backup sessions
            if include_sessions and hasattr(self.engine, 'active_sessions'):
                sessions_file = backup_path / "sessions.pkl"
                with open(sessions_file, 'wb') as f:
                    pickle.dump(self.engine.active_sessions, f)
            
            # Backup checkpoints
            if include_checkpoints and hasattr(self.engine, 'cache_path') and self.engine.cache_path:
                checkpoints_source = Path(self.engine.cache_path) / "checkpoints"
                if checkpoints_source.exists():
                    checkpoints_backup = backup_path / "checkpoints"
                    shutil.copytree(checkpoints_source, checkpoints_backup, dirs_exist_ok=True)
            
            # Compress if requested
            if compress:
                compressed_path = f"{backup_path}.tar.gz"
                with tarfile.open(compressed_path, "w:gz") as tar:
                    tar.add(backup_path, arcname=name)
                
                # Remove uncompressed directory
                shutil.rmtree(backup_path)
                backup_path = Path(compressed_path)
            
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
        
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            # Cleanup on error
            if backup_path.exists():
                if backup_path.is_file():
                    backup_path.unlink()
                else:
                    shutil.rmtree(backup_path)
            raise
    
    def restore_backup(self, backup_path: str, 
                      restore_cache: bool = True,
                      restore_sessions: bool = True,
                      restore_checkpoints: bool = True):
        """Restore from backup."""
        backup_path_obj = Path(backup_path)
        
        # Check if compressed
        if backup_path.endswith('.tar.gz'):
            # Extract
            extract_dir = backup_path_obj.parent / backup_path_obj.stem.replace('.tar', '')
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            
            backup_path_obj = extract_dir / backup_path_obj.stem.replace('.tar', '')
        
        if not backup_path_obj.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        try:
            # Load metadata
            metadata_file = backup_path_obj / "metadata.json"
            if not metadata_file.exists():
                raise ValueError("Backup metadata not found")
            
            with open(metadata_file) as f:
                backup_data = json.load(f)
            
            logger.info(f"Restoring backup from {backup_data.get('datetime', 'unknown')}")
            
            # Restore cache
            if restore_cache:
                cache_backup = backup_path_obj / "cache"
                if cache_backup.exists() and hasattr(self.engine, 'cache_path'):
                    if self.engine.cache_path:
                        target_cache = Path(self.engine.cache_path)
                        if target_cache.exists():
                            shutil.rmtree(target_cache)
                        shutil.copytree(cache_backup, target_cache)
                        logger.info("Cache restored")
            
            # Restore sessions
            if restore_sessions:
                sessions_file = backup_path_obj / "sessions.pkl"
                if sessions_file.exists():
                    with open(sessions_file, 'rb') as f:
                        sessions = pickle.load(f)
                    
                    if hasattr(self.engine, 'active_sessions'):
                        self.engine.active_sessions.update(sessions)
                        logger.info(f"Sessions restored: {len(sessions)}")
            
            # Restore checkpoints
            if restore_checkpoints:
                checkpoints_backup = backup_path_obj / "checkpoints"
                if checkpoints_backup.exists() and hasattr(self.engine, 'cache_path'):
                    if self.engine.cache_path:
                        target_checkpoints = Path(self.engine.cache_path) / "checkpoints"
                        target_checkpoints.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(checkpoints_backup, target_checkpoints, dirs_exist_ok=True)
                        logger.info("Checkpoints restored")
            
            logger.info("Backup restored successfully")
        
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        # Check for compressed backups
        for backup_file in self.backup_dir.glob("*.tar.gz"):
            try:
                # Try to extract metadata
                with tarfile.open(backup_file, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith("metadata.json"):
                            metadata_str = tar.extractfile(member).read()
                            metadata = json.loads(metadata_str)
                            backups.append({
                                'name': backup_file.name,
                                'path': str(backup_file),
                                'datetime': metadata.get('datetime'),
                                'timestamp': metadata.get('timestamp'),
                                'compressed': True
                            })
                            break
            except:
                backups.append({
                    'name': backup_file.name,
                    'path': str(backup_file),
                    'compressed': True
                })
        
        # Check for uncompressed backups
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir() and backup_dir.name.startswith("backup_"):
                metadata_file = backup_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    
                    backups.append({
                        'name': backup_dir.name,
                        'path': str(backup_dir),
                        'datetime': metadata.get('datetime'),
                        'timestamp': metadata.get('timestamp'),
                        'compressed': False
                    })
        
        # Sort by timestamp
        backups.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return backups
    
    def delete_backup(self, backup_name: str):
        """Delete a backup."""
        backup_path = self.backup_dir / backup_name
        
        if backup_path.exists():
            if backup_path.is_file():
                backup_path.unlink()
            else:
                shutil.rmtree(backup_path)
            logger.info(f"Backup deleted: {backup_name}")
        else:
            raise FileNotFoundError(f"Backup not found: {backup_name}")
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """Cleanup old backups, keeping only the most recent N."""
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return
        
        # Delete oldest backups
        to_delete = backups[keep_count:]
        for backup in to_delete:
            try:
                self.delete_backup(backup['name'])
            except Exception as e:
                logger.warning(f"Failed to delete backup {backup['name']}: {e}")


class ScheduledBackup:
    """Scheduled backup automation."""
    
    def __init__(self, backup_manager: BackupManager, interval_hours: float = 24):
        self.backup_manager = backup_manager
        self.interval_seconds = interval_hours * 3600
        self.running = False
    
    async def start(self):
        """Start scheduled backups."""
        self.running = True
        logger.info(f"Starting scheduled backups (interval: {self.interval_seconds/3600:.1f} hours)")
        
        while self.running:
            try:
                await asyncio.sleep(self.interval_seconds)
                
                if self.running:
                    logger.info("Creating scheduled backup...")
                    backup_path = self.backup_manager.create_backup(compress=True)
                    logger.info(f"Scheduled backup created: {backup_path}")
                    
                    # Cleanup old backups
                    self.backup_manager.cleanup_old_backups(keep_count=10)
            
            except Exception as e:
                logger.error(f"Error in scheduled backup: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    def stop(self):
        """Stop scheduled backups."""
        self.running = False
        logger.info("Stopped scheduled backups")

