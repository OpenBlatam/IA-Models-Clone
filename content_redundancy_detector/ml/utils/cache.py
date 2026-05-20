"""
Caching Utilities
Caching for models and computations
"""

import torch
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Cache for model checkpoints
    """
    
    def __init__(self, cache_dir: Path = Path(".cache")):
        """
        Initialize model cache
        
        Args:
            cache_dir: Cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, model_name: str, config: dict) -> str:
        """
        Generate cache key
        
        Args:
            model_name: Model name
            config: Model configuration
            
        Returns:
            Cache key
        """
        config_str = str(sorted(config.items()))
        key_string = f"{model_name}_{config_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, model_name: str, config: dict) -> Optional[torch.nn.Module]:
        """
        Get model from cache
        
        Args:
            model_name: Model name
            config: Model configuration
            
        Returns:
            Cached model or None
        """
        cache_key = self._get_cache_key(model_name, config)
        cache_file = self.cache_dir / f"{cache_key}.pth"
        
        if cache_file.exists():
            try:
                model = torch.load(cache_file, map_location='cpu')
                logger.info(f"Loaded model from cache: {cache_file}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        return None
    
    def put(self, model: torch.nn.Module, model_name: str, config: dict) -> None:
        """
        Put model in cache
        
        Args:
            model: Model to cache
            model_name: Model name
            config: Model configuration
        """
        cache_key = self._get_cache_key(model_name, config)
        cache_file = self.cache_dir / f"{cache_key}.pth"
        
        try:
            torch.save(model.state_dict(), cache_file)
            logger.info(f"Cached model: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache model: {e}")


class ComputationCache:
    """
    Cache for computation results
    """
    
    def __init__(self, cache_dir: Path = Path(".cache")):
        """
        Initialize computation cache
        
        Args:
            cache_dir: Cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """
        Generate cache key from function arguments
        
        Args:
            func_name: Function name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        key_string = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, func_name: str, *args, **kwargs) -> Optional[Any]:
        """
        Get result from cache
        
        Args:
            func_name: Function name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cached result or None
        """
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                logger.debug(f"Loaded from cache: {cache_file}")
                return result
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        return None
    
    def put(self, result: Any, func_name: str, *args, **kwargs) -> None:
        """
        Put result in cache
        
        Args:
            result: Result to cache
            func_name: Function name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Cached result: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")


def cached(cache_dir: Path = Path(".cache")):
    """
    Decorator for caching function results
    
    Args:
        cache_dir: Cache directory
        
    Returns:
        Decorator function
    """
    cache = ComputationCache(cache_dir)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Try to get from cache
            result = cache.get(func.__name__, *args, **kwargs)
            if result is not None:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache.put(result, func.__name__, *args, **kwargs)
            
            return result
        
        return wrapper
    return decorator



