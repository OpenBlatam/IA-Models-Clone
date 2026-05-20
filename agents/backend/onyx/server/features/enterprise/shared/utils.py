from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import hashlib
import time
import json
from typing import Any, Dict, Optional
from datetime import datetime, timezone
        import orjson
        import orjson
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Shared Utilities
===============

Common utility functions used across the enterprise API.
"""



def generate_cache_key(key: str, prefix: str = "enterprise") -> str:
    """Generate a consistent cache key."""
    return f"{prefix}:{hashlib.md5(key.encode()).hexdigest()}"


def get_current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def serialize_json(data: Any) -> str:
    """Serialize data to JSON string."""
    try:
        return orjson.dumps(data).decode()
    except ImportError:
        return json.dumps(data, default=str)


def deserialize_json(data: str) -> Any:
    """Deserialize JSON string to data."""
    try:
        return orjson.loads(data)
    except ImportError:
        return json.loads(data)


def measure_time(func) -> Any:
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


def safe_get_client_ip(request) -> str:
    """Safely get client IP from request."""
    if hasattr(request, 'client') and request.client:
        return request.client.host
    return "unknown"


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def create_response_headers(additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Create standard response headers."""
    headers = {
        "X-API-Version": "2.0.0",
        "X-Powered-By": "Enterprise API",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    if additional_headers:
        headers.update(additional_headers)
    
    return headers


def validate_json_payload(payload: str, max_size: int = 1024 * 1024) -> bool:
    """Validate JSON payload size and format."""
    if len(payload) > max_size:
        return False
    
    try:
        json.loads(payload)
        return True
    except (json.JSONDecodeError, TypeError):
        return False 