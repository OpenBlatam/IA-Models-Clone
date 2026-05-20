"""
Cache Cleanup Module

Cache cleanup and expiration handling.
"""

from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def cleanup_expired(memory_cache: Dict[str, Dict[str, Any]]) -> None:
    """
    Elimina items expirados del cache.
    
    Args:
        memory_cache: Cache dictionary
    """
    now = datetime.now()
    expired_keys = [
        k for k, v in memory_cache.items()
        if now >= v.get("expires_at", datetime.min)
    ]
    
    for key in expired_keys:
        del memory_cache[key]
    
    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")



