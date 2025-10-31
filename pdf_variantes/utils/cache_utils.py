"""Cache utilities."""

import hashlib
import json
from typing import Any, Optional, Dict
from datetime import datetime, timedelta


def get_cache_key(prefix: str, *args) -> str:
    """Generate cache key from prefix and arguments."""
    key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
    return hashlib.md5(key_data.encode()).hexdigest()


def is_cache_valid(cache_entry: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
    """Check if cache entry is still valid."""
    if not cache_entry:
        return False
    
    created_at = cache_entry.get("created_at")
    if not created_at:
        return False
    
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    expiry_time = created_at + timedelta(seconds=ttl_seconds)
    return datetime.utcnow() < expiry_time


def set_cache_entry(key: str, data: Any, ttl_seconds: int = 3600) -> Dict[str, Any]:
    """Create cache entry with metadata."""
    return {
        "key": key,
        "data": data,
        "created_at": datetime.utcnow().isoformat(),
        "ttl_seconds": ttl_seconds,
        "expires_at": (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat()
    }


def serialize_for_cache(data: Any) -> str:
    """Serialize data for caching."""
    try:
        return json.dumps(data, default=str)
    except Exception:
        return str(data)


def deserialize_from_cache(cached_data: str) -> Any:
    """Deserialize data from cache."""
    try:
        return json.loads(cached_data)
    except Exception:
        return cached_data
