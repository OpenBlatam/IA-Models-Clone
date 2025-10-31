from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
from typing import Any, Dict, ClassVar, Optional
from functools import lru_cache
import time
from .base_types import CACHE_TTL, CACHE_SIZE
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cache Mixin - Onyx Integration
Caching functionality for models.
"""

class CacheMixin:
    """Mixin for caching functionality."""
    
    _validation_cache: ClassVar[Dict[str, Any]] = {}
    _cache_timestamp: ClassVar[Dict[str, float]] = {}
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._validation_cache:
            if time.time() - self._cache_timestamp.get(key, 0) < CACHE_TTL:
                return self._validation_cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp."""
        self._validation_cache[key] = value
        self._cache_timestamp[key] = time.time()
    
    def _clear_cache(self) -> None:
        """Clear all cache."""
        self._validation_cache.clear()
        self._cache_timestamp.clear()
    
    @lru_cache(maxsize=CACHE_SIZE)
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with caching."""
        return self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True
        )
    
    @lru_cache(maxsize=CACHE_SIZE)
    def to_json(self) -> str:
        """Convert to JSON string with caching."""
        return self.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True
        )
    
    def _clear_method_cache(self) -> None:
        """Clear method cache."""
        self.to_dict.cache_clear()
        self.to_json.cache_clear() 