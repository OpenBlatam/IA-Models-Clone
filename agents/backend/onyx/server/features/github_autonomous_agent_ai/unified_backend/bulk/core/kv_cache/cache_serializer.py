"""
Cache serialization utilities.

Provides efficient serialization for cache state.
"""
from __future__ import annotations

import logging
import pickle
from typing import Optional, Dict, Any
import torch

from kv_cache.types import StatsDict

logger = logging.getLogger(__name__)


class CacheSerializer:
    """
    Cache serializer for state persistence.
    
    Provides efficient serialization of cache state.
    """
    
    def __init__(self, compress: bool = True, use_pickle: bool = True):
        """
        Initialize cache serializer.
        
        Args:
            compress: Whether to compress serialized data
            use_pickle: Whether to use pickle (vs custom format)
        """
        self.compress = compress
        self.use_pickle = use_pickle
    
    def serialize_cache(
        self,
        cache: Any,
        include_stats: bool = True
    ) -> bytes:
        """
        Serialize cache state.
        
        Args:
            cache: Cache instance
            include_stats: Whether to include statistics
            
        Returns:
            Serialized cache state as bytes
        """
        state = {
            "config": cache.config.to_dict(),
            "entries": {}
        }
        
        # Serialize cache entries
        storage = cache.storage
        positions = storage.get_positions()
        
        for pos in positions:
            entry = storage.get(pos)
            if entry:
                key, value = entry
                state["entries"][pos] = {
                    "key": key.cpu().detach(),
                    "value": value.cpu().detach()
                }
        
        # Include stats if requested
        if include_stats:
            state["stats"] = cache.get_stats()
        
        # Serialize
        if self.use_pickle:
            data = pickle.dumps(state)
        else:
            # Custom serialization (placeholder)
            data = pickle.dumps(state)
        
        # Compress if requested
        if self.compress:
            import gzip
            data = gzip.compress(data)
        
        return data
    
    def deserialize_cache(
        self,
        data: bytes,
        cache: Any
    ) -> Dict[str, Any]:
        """
        Deserialize cache state.
        
        Args:
            data: Serialized cache state
            cache: Cache instance to restore to
            
        Returns:
            Dictionary with restored state info
        """
        # Decompress if needed
        if self.compress:
            import gzip
            try:
                data = gzip.decompress(data)
            except:
                # Assume not compressed
                pass
        
        # Deserialize
        if self.use_pickle:
            state = pickle.loads(data)
        else:
            state = pickle.loads(data)
        
        # Restore entries
        restored = 0
        for pos, entry_data in state.get("entries", {}).items():
            try:
                key = entry_data["key"]
                value = entry_data["value"]
                cache.put(int(pos), key, value)
                restored += 1
            except Exception as e:
                logger.warning(f"Failed to restore position {pos}: {e}")
        
        return {
            "restored_entries": restored,
            "config": state.get("config"),
            "stats": state.get("stats")
        }
    
    def serialize_stats(self, stats: StatsDict) -> bytes:
        """
        Serialize statistics only.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Serialized stats as bytes
        """
        data = pickle.dumps(stats)
        
        if self.compress:
            import gzip
            data = gzip.compress(data)
        
        return data
    
    def deserialize_stats(self, data: bytes) -> StatsDict:
        """
        Deserialize statistics.
        
        Args:
            data: Serialized stats
            
        Returns:
            Statistics dictionary
        """
        if self.compress:
            import gzip
            try:
                data = gzip.decompress(data)
            except:
                pass
        
        return pickle.loads(data)

