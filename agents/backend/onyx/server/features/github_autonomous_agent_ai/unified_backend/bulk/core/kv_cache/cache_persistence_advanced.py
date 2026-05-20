"""
Advanced persistence system for KV cache.

This module provides advanced persistence capabilities including
incremental saves, checkpoints, and recovery mechanisms.
"""

import time
import threading
import json
import pickle
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PersistenceMode(Enum):
    """Persistence modes."""
    SYNC = "sync"  # Synchronous writes
    ASYNC = "async"  # Asynchronous writes
    LAZY = "lazy"  # Write on demand
    INCREMENTAL = "incremental"  # Incremental saves


@dataclass
class PersistenceConfig:
    """Persistence configuration."""
    mode: PersistenceMode
    file_path: str
    format: str = "pickle"  # "pickle" or "json"
    save_interval: float = 300.0  # 5 minutes
    incremental: bool = True
    compression: bool = False


@dataclass
class Checkpoint:
    """A persistence checkpoint."""
    checkpoint_id: str
    timestamp: float
    cache_size: int
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedPersistenceManager:
    """Advanced persistence manager."""
    
    def __init__(self, cache: Any, config: PersistenceConfig):
        self.cache = cache
        self.config = config
        self._checkpoints: List[Checkpoint] = []
        self._dirty_keys: set = set()
        self._lock = threading.Lock()
        self._save_thread: Optional[threading.Thread] = None
        self._running = False
        
        if config.mode == PersistenceMode.ASYNC:
            self.start_async_save()
            
    def start_async_save(self) -> None:
        """Start asynchronous save thread."""
        if self._running:
            return
            
        self._running = True
        self._save_thread = threading.Thread(target=self._async_save_loop, daemon=True)
        self._save_thread.start()
        
    def stop_async_save(self) -> None:
        """Stop asynchronous save thread."""
        self._running = False
        if self._save_thread:
            self._save_thread.join(timeout=5.0)
            
    def _async_save_loop(self) -> None:
        """Asynchronous save loop."""
        while self._running:
            try:
                if self.config.incremental:
                    self._save_incremental()
                else:
                    self._save_full()
                time.sleep(self.config.save_interval)
            except Exception as e:
                print(f"Error in async save loop: {e}")
                time.sleep(60)
                
    def save(self, checkpoint_id: Optional[str] = None) -> Checkpoint:
        """Save cache to disk."""
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{int(time.time())}"
            
        file_path = f"{self.config.file_path}_{checkpoint_id}.{self.config.format}"
        
        # Get cache data
        cache_data = {}
        if hasattr(self.cache, '_cache'):
            cache_data = dict(self.cache._cache)
            
        # Save based on format
        if self.config.format == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(cache_data, f)
        else:
            with open(file_path, 'w') as f:
                json.dump(cache_data, f)
                
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            cache_size=len(cache_data),
            file_path=file_path
        )
        
        with self._lock:
            self._checkpoints.append(checkpoint)
            self._dirty_keys.clear()
            
        return checkpoint
        
    def _save_full(self) -> None:
        """Save full cache."""
        self.save()
        
    def _save_incremental(self) -> None:
        """Save only changed keys."""
        if not self._dirty_keys:
            return
            
        # Save incremental changes
        incremental_data = {}
        for key in self._dirty_keys:
            value = self.cache.get(key)
            if value is not None:
                incremental_data[key] = value
                
        if incremental_data:
            file_path = f"{self.config.file_path}_incremental_{int(time.time())}.{self.config.format}"
            
            if self.config.format == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(incremental_data, f)
            else:
                with open(file_path, 'w') as f:
                    json.dump(incremental_data, f)
                    
        with self._lock:
            self._dirty_keys.clear()
            
    def load(self, checkpoint_id: Optional[str] = None) -> bool:
        """Load cache from disk."""
        if checkpoint_id:
            # Load specific checkpoint
            file_path = f"{self.config.file_path}_{checkpoint_id}.{self.config.format}"
        else:
            # Load latest checkpoint
            checkpoints = self.list_checkpoints()
            if not checkpoints:
                return False
            file_path = checkpoints[-1].file_path
            
        if not Path(file_path).exists():
            return False
            
        try:
            if self.config.format == "pickle":
                with open(file_path, 'rb') as f:
                    cache_data = pickle.load(f)
            else:
                with open(file_path, 'r') as f:
                    cache_data = json.load(f)
                    
            # Restore to cache
            if hasattr(self.cache, '_cache'):
                self.cache._cache.clear()
                self.cache._cache.update(cache_data)
                
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
            
    def mark_dirty(self, key: str) -> None:
        """Mark key as dirty for incremental save."""
        with self._lock:
            self._dirty_keys.add(key)
            
    def list_checkpoints(self) -> List[Checkpoint]:
        """List all checkpoints."""
        return self._checkpoints.copy()


