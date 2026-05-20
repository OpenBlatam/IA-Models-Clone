"""
Shared Utilities
Utility functions used across modules
"""

from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import hashlib
import json


def generate_id(prefix: str = "") -> str:
    """Generate unique ID"""
    import uuid
    if prefix:
        return f"{prefix}_{uuid.uuid4()}"
    return str(uuid.uuid4())


def hash_content(content: str) -> str:
    """Hash content for integrity checking"""
    return hashlib.sha256(content.encode()).hexdigest()


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity (0.0 - 1.0)"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()


def paginate(
    items: List[Any],
    page: int = 1,
    limit: int = 20
) -> Dict[str, Any]:
    """Paginate items"""
    offset = (page - 1) * limit
    total = len(items)
    
    paginated_items = items[offset:offset + limit]
    
    return {
        "items": paginated_items,
        "total": total,
        "page": page,
        "limit": limit,
        "offset": offset,
        "total_pages": (total + limit - 1) // limit if limit > 0 else 0,
        "has_next": (offset + limit) < total,
        "has_previous": offset > 0
    }


def format_datetime(dt: datetime) -> str:
    """Format datetime to ISO string"""
    return dt.isoformat() if dt else None


def parse_datetime(iso_string: str) -> Optional[datetime]:
    """Parse ISO string to datetime"""
    if not iso_string:
        return None
    try:
        return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    except:
        return None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename"""
    import re
    # Remove path components
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    # Remove special characters except dots and hyphens
    filename = re.sub(r'[^\w\-_.]', '', filename)
    # Limit length
    return filename[:500]


def validate_email(email: str) -> bool:
    """Validate email format"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks"""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


class RetryHelper:
    """Helper for retry operations"""
    
    @staticmethod
    async def retry_async(
        func: callable,
        max_retries: int = 3,
        delay: float = 1.0,
        exceptions: tuple = (Exception,)
    ):
        """Retry async function"""
        import asyncio
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await func()
            except exceptions as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (attempt + 1))
        
        raise last_exception


class CacheHelper:
    """Helper for caching operations"""
    
    def __init__(self):
        self._cache: Dict[str, tuple[Any, datetime, int]] = {}
    
    def get(self, key: str, ttl: int = 3600) -> Optional[Any]:
        """Get from cache"""
        if key not in self._cache:
            return None
        
        value, stored_at, stored_ttl = self._cache[key]
        
        if (datetime.utcnow() - stored_at).total_seconds() > stored_ttl:
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set in cache"""
        self._cache[key] = (value, datetime.utcnow(), ttl)
    
    def delete(self, key: str) -> bool:
        """Delete from cache"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache"""
        self._cache.clear()






