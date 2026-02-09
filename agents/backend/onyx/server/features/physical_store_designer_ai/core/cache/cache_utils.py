"""
Cache Utilities

Utility functions for cache operations.
"""

import hashlib
import json
from typing import Any


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from function arguments"""
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items()) if kwargs else {}
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()




