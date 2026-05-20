"""
Snapshot and backup system for KV cache.

This module provides snapshot creation, backup management, and restore capabilities.
"""

import time
import threading
import json
import pickle
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SnapshotFormat(Enum):
    """Snapshot formats."""
    JSON = "json"
    PICKLE = "pickle"
    BINARY = "binary"
    CUSTOM = "custom"


@dataclass
class Snapshot:
    """A cache snapshot."""
    snapshot_id: str
    timestamp: float
    cache_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    format: SnapshotFormat = SnapshotFormat.PICKLE
    size_bytes: int = 0


@dataclass
class BackupConfig:
    """Backup configuration."""
    backup_directory: str
    format: SnapshotFormat = SnapshotFormat.PICKLE
    compression: bool = True
    max_backups: int = 10
    auto_backup: bool = False
    backup_interval: float = 3600.0  # 1 hour


class CacheSnapshotManager:
    """Manages cache snapshots."""
    
    def __init__(self, cache: Any, config: BackupConfig):
        self.cache = cache
        self.config = config
        self._snapshots: Dict[str, Snapshot] = {}
        self._lock = threading.Lock()
        
        # Create backup directory
        Path(config.backup_directory).mkdir(parents=True, exist_ok=True)
        
    def create_snapshot(
        self,
        snapshot_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Snapshot:
        """Create a snapshot of current cache state."""
        if snapshot_id is None:
            snapshot_id = f"snapshot_{int(time.time())}"
            
        # Get all cache data
        cache_data = {}
        if hasattr(self.cache, '_cache'):
            cache_data = dict(self.cache._cache)
            
        # Calculate size
        size_bytes = len(str(cache_data).encode('utf-8'))
        
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            cache_data=cache_data,
            metadata=metadata or {},
            format=self.config.format,
            size_bytes=size_bytes
        )
        
        with self._lock:
            self._snapshots[snapshot_id] = snapshot
            
        # Save to disk
        self._save_snapshot(snapshot)
        
        # Cleanup old backups
        self._cleanup_old_backups()
        
        return snapshot
        
    def _save_snapshot(self, snapshot: Snapshot) -> None:
        """Save snapshot to disk."""
        filepath = Path(self.config.backup_directory) / f"{snapshot.snapshot_id}.{snapshot.format.value}"
        
        if snapshot.format == SnapshotFormat.JSON:
            with open(filepath, 'w') as f:
                json.dump({
                    'snapshot_id': snapshot.snapshot_id,
                    'timestamp': snapshot.timestamp,
                    'cache_data': snapshot.cache_data,
                    'metadata': snapshot.metadata
                }, f)
        elif snapshot.format == SnapshotFormat.PICKLE:
            with open(filepath, 'wb') as f:
                pickle.dump(snapshot, f)
        else:
            # Binary format
            with open(filepath, 'wb') as f:
                pickle.dump(snapshot, f)
                
    def load_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        """Load snapshot from disk."""
        filepath = Path(self.config.backup_directory) / f"{snapshot_id}.{self.config.format.value}"
        
        if not filepath.exists():
            return None
            
        try:
            if self.config.format == SnapshotFormat.JSON:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    return Snapshot(**data)
            else:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading snapshot: {e}")
            return None
            
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore cache from snapshot."""
        snapshot = self.load_snapshot(snapshot_id)
        if not snapshot:
            return False
            
        # Restore cache data
        if hasattr(self.cache, '_cache'):
            self.cache._cache.clear()
            self.cache._cache.update(snapshot.cache_data)
            
        return True
        
    def list_snapshots(self) -> List[str]:
        """List all snapshot IDs."""
        backup_dir = Path(self.config.backup_directory)
        snapshots = []
        
        for file in backup_dir.glob(f"*.{self.config.format.value}"):
            snapshot_id = file.stem
            snapshots.append(snapshot_id)
            
        return sorted(snapshots)
        
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        filepath = Path(self.config.backup_directory) / f"{snapshot_id}.{self.config.format.value}"
        
        if filepath.exists():
            filepath.unlink()
            with self._lock:
                self._snapshots.pop(snapshot_id, None)
            return True
        return False
        
    def _cleanup_old_backups(self) -> None:
        """Cleanup old backups keeping only max_backups."""
        snapshots = self.list_snapshots()
        
        if len(snapshots) > self.config.max_backups:
            # Delete oldest
            to_delete = snapshots[:len(snapshots) - self.config.max_backups]
            for snapshot_id in to_delete:
                self.delete_snapshot(snapshot_id)
                
    def get_snapshot_info(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a snapshot."""
        snapshot = self.load_snapshot(snapshot_id)
        if not snapshot:
            return None
            
        return {
            'snapshot_id': snapshot.snapshot_id,
            'timestamp': snapshot.timestamp,
            'size_bytes': snapshot.size_bytes,
            'format': snapshot.format.value,
            'metadata': snapshot.metadata,
            'key_count': len(snapshot.cache_data)
        }


class BackupManager:
    """Manages automatic backups."""
    
    def __init__(self, cache: Any, config: BackupConfig):
        self.cache = cache
        self.config = config
        self.snapshot_manager = CacheSnapshotManager(cache, config)
        self._backup_thread: Optional[threading.Thread] = None
        self._running = False
        
        if config.auto_backup:
            self.start_backup()
            
    def start_backup(self) -> None:
        """Start automatic backup thread."""
        if self._running:
            return
            
        self._running = True
        self._backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self._backup_thread.start()
        
    def stop_backup(self) -> None:
        """Stop automatic backup thread."""
        self._running = False
        if self._backup_thread:
            self._backup_thread.join(timeout=5.0)
            
    def _backup_loop(self) -> None:
        """Backup loop."""
        while self._running:
            try:
                self.snapshot_manager.create_snapshot(
                    metadata={'auto_backup': True}
                )
                time.sleep(self.config.backup_interval)
            except Exception as e:
                print(f"Error in backup loop: {e}")
                time.sleep(60)  # Wait before retrying
                
    def create_backup(self, metadata: Optional[Dict[str, Any]] = None) -> Snapshot:
        """Create a manual backup."""
        return self.snapshot_manager.create_snapshot(metadata=metadata)
        
    def restore_backup(self, snapshot_id: str) -> bool:
        """Restore from backup."""
        return self.snapshot_manager.restore_snapshot(snapshot_id)
        
    def list_backups(self) -> List[str]:
        """List all backups."""
        return self.snapshot_manager.list_snapshots()














