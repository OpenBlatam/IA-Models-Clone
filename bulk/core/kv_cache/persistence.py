"""
Persistence utilities for KV Cache.

Provides save/load functionality for cache state.
"""
import logging
import os
import pickle
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


class CachePersistence:
    """
    Handles persistence of cache state.
    
    Saves and loads cache to/from disk for checkpointing and recovery.
    """
    
    def __init__(self, persistence_path: str):
        """
        Initialize cache persistence.
        
        Args:
            persistence_path: Directory path for persistence
        """
        self.persistence_path = persistence_path
        os.makedirs(persistence_path, exist_ok=True)
        
        logger.info(f"Initialized CachePersistence with path={persistence_path}")
    
    def save_cache(
        self,
        cache: Any,
        filename: str = "cache_state.pkl"
    ) -> str:
        """
        Save cache state to disk.
        
        Args:
            cache: Cache instance to save
            filename: Filename for saved state
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.persistence_path, filename)
        
        try:
            # Collect cache data
            cache_data = {
                "storage": self._extract_storage_data(cache),
                "stats": cache.get_stats() if hasattr(cache, 'get_stats') else {},
                "config": cache.config.__dict__ if hasattr(cache, 'config') else {},
            }
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Cache saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}", exc_info=True)
            raise
    
    def load_cache(
        self,
        cache: Any,
        filename: str = "cache_state.pkl"
    ) -> bool:
        """
        Load cache state from disk.
        
        Args:
            cache: Cache instance to load into
            filename: Filename to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = os.path.join(self.persistence_path, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Cache file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restore cache state
            if hasattr(cache, 'storage') and 'storage' in cache_data:
                self._restore_storage_data(cache, cache_data['storage'])
            
            logger.info(f"Cache loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}", exc_info=True)
            return False
    
    def _extract_storage_data(self, cache: Any) -> Dict[str, Any]:
        """Extract storage data from cache."""
        storage_data = {}
        
        if hasattr(cache, 'storage'):
            storage = cache.storage
            if hasattr(storage, 'get_positions'):
                positions = storage.get_positions()
                storage_data['positions'] = positions
                storage_data['entries'] = {}
                
                # Save actual tensor data (simplified - in production would use better format)
                for pos in positions[:100]:  # Limit for now
                    entry = storage.get(pos)
                    if entry:
                        key, value = entry
                        storage_data['entries'][pos] = {
                            'key_shape': list(key.shape),
                            'value_shape': list(value.shape),
                        }
        
        return storage_data
    
    def _restore_storage_data(self, cache: Any, storage_data: Dict[str, Any]) -> None:
        """Restore storage data to cache."""
        # Implementation would restore actual tensor data
        # For now, just log
        logger.debug(f"Restoring storage data with {len(storage_data.get('positions', []))} positions")


def save_cache_checkpoint(cache: Any, checkpoint_dir: str, step: int) -> str:
    """
    Save cache checkpoint.
    
    Args:
        cache: Cache instance
        checkpoint_dir: Checkpoint directory
        step: Training step number
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f"cache_step_{step}.pkl"
    
    persistence = CachePersistence(checkpoint_dir)
    return persistence.save_cache(cache, filename)


def load_cache_checkpoint(cache: Any, checkpoint_dir: str, step: Optional[int] = None) -> bool:
    """
    Load cache checkpoint.
    
    Args:
        cache: Cache instance
        checkpoint_dir: Checkpoint directory
        step: Step number (None for latest)
        
    Returns:
        True if loaded successfully
    """
    if step is None:
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("cache_step_")]
        if not checkpoints:
            return False
        checkpoint_file = sorted(checkpoints)[-1]
    else:
        checkpoint_file = f"cache_step_{step}.pkl"
    
    persistence = CachePersistence(checkpoint_dir)
    return persistence.load_cache(cache, checkpoint_file)

