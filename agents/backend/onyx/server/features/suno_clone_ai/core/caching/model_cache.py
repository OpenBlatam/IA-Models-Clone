"""
Model Caching

Utilities for caching models.
"""

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelCache:
    """Cache for models."""
    
    def __init__(self, cache_dir: str = "./cache/models"):
        """
        Initialize model cache.
        
        Args:
            cache_dir: Cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index: Dict[str, str] = {}
    
    def _get_cache_key(self, model_name: str, **kwargs) -> str:
        """
        Generate cache key.
        
        Args:
            model_name: Model name
            **kwargs: Additional parameters
            
        Returns:
            Cache key
        """
        key_data = f"{model_name}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cache(
        self,
        model: nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Cache model.
        
        Args:
            model: Model to cache
            model_name: Model name
            metadata: Optional metadata
            **kwargs: Additional parameters
            
        Returns:
            Cache path
        """
        cache_key = self._get_cache_key(model_name, **kwargs)
        cache_path = self.cache_dir / f"{cache_key}.pt"
        
        # Save model state dict
        torch.save({
            'state_dict': model.state_dict(),
            'metadata': metadata or {}
        }, cache_path)
        
        self.cache_index[model_name] = str(cache_path)
        logger.info(f"Cached model: {model_name} -> {cache_path}")
        
        return str(cache_path)
    
    def get(
        self,
        model_name: str,
        model_class: Optional[type] = None,
        **kwargs
    ) -> Optional[nn.Module]:
        """
        Get cached model.
        
        Args:
            model_name: Model name
            model_class: Model class to instantiate
            **kwargs: Additional parameters
            
        Returns:
            Cached model or None
        """
        cache_key = self._get_cache_key(model_name, **kwargs)
        cache_path = self.cache_dir / f"{cache_key}.pt"
        
        if not cache_path.exists():
            logger.warning(f"Cache not found: {cache_path}")
            return None
        
        try:
            checkpoint = torch.load(cache_path)
            
            if model_class:
                model = model_class(**kwargs)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Return state dict if no model class provided
                return checkpoint['state_dict']
            
            logger.info(f"Loaded cached model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading cached model: {e}")
            return None
    
    def clear(self, model_name: Optional[str] = None) -> None:
        """
        Clear cache.
        
        Args:
            model_name: Specific model to clear (None = all)
        """
        if model_name:
            cache_key = self._get_cache_key(model_name)
            cache_path = self.cache_dir / f"{cache_key}.pt"
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache for: {model_name}")
        else:
            # Clear all
            for cache_file in self.cache_dir.glob("*.pt"):
                cache_file.unlink()
            self.cache_index.clear()
            logger.info("Cleared all cache")


# Global cache instance
_global_cache = ModelCache()


def cache_model(
    model: nn.Module,
    model_name: str,
    **kwargs
) -> str:
    """Cache model in global cache."""
    return _global_cache.cache(model, model_name, **kwargs)


def get_cached_model(
    model_name: str,
    **kwargs
) -> Optional[nn.Module]:
    """Get model from global cache."""
    return _global_cache.get(model_name, **kwargs)


def clear_cache(model_name: Optional[str] = None) -> None:
    """Clear global cache."""
    _global_cache.clear(model_name)



