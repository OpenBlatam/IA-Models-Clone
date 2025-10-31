"""
Helper Utilities
================

Common helper utilities for the application.
"""

from __future__ import annotations
import hashlib
import secrets
import string
import uuid
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime, timezone, timedelta
from functools import wraps
import asyncio
import time

from pydantic import BaseModel


logger = logging.getLogger(__name__)

T = TypeVar('T')


class StringHelpers:
    """String manipulation helpers"""
    
    @staticmethod
    def generate_random_string(length: int = 32, include_symbols: bool = False) -> str:
        """Generate random string"""
        characters = string.ascii_letters + string.digits
        if include_symbols:
            characters += "!@#$%^&*"
        
        return ''.join(secrets.choice(characters) for _ in range(length))
    
    @staticmethod
    def generate_slug(text: str, max_length: int = 50) -> str:
        """Generate URL-friendly slug from text"""
        import re
        
        # Convert to lowercase and replace spaces with hyphens
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        
        # Trim to max length
        if len(slug) > max_length:
            slug = slug[:max_length].rstrip('-')
        
        return slug
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to max length with suffix"""
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def extract_hashtags(text: str) -> List[str]:
        """Extract hashtags from text"""
        import re
        return re.findall(r'#\w+', text)
    
    @staticmethod
    def extract_mentions(text: str) -> List[str]:
        """Extract mentions from text"""
        import re
        return re.findall(r'@\w+', text)
    
    @staticmethod
    def clean_html(html: str) -> str:
        """Remove HTML tags from text"""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', html)
    
    @staticmethod
    def mask_sensitive_data(text: str, visible_chars: int = 4) -> str:
        """Mask sensitive data (e.g., credit cards, emails)"""
        if len(text) <= visible_chars:
            return "*" * len(text)
        
        return text[:visible_chars] + "*" * (len(text) - visible_chars)


class HashHelpers:
    """Hashing utilities"""
    
    @staticmethod
    def generate_hash(data: str, algorithm: str = "sha256") -> str:
        """Generate hash for data"""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @staticmethod
    def generate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
        """Generate hash for file"""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def verify_hash(data: str, hash_value: str, algorithm: str = "sha256") -> bool:
        """Verify hash for data"""
        return HashHelpers.generate_hash(data, algorithm) == hash_value


class UUIDHelpers:
    """UUID utilities"""
    
    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID string"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_uuid1() -> str:
        """Generate UUID1 string"""
        return str(uuid.uuid1())
    
    @staticmethod
    def is_valid_uuid(uuid_string: str) -> bool:
        """Check if string is valid UUID"""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False


class DateTimeHelpers:
    """DateTime utilities"""
    
    @staticmethod
    def now_utc() -> datetime:
        """Get current UTC datetime"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        """Convert datetime to UTC"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime to string"""
        return dt.strftime(format_str)
    
    @staticmethod
    def parse_datetime(date_string: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """Parse string to datetime"""
        return datetime.strptime(date_string, format_str)
    
    @staticmethod
    def add_days(dt: datetime, days: int) -> datetime:
        """Add days to datetime"""
        return dt + timedelta(days=days)
    
    @staticmethod
    def add_hours(dt: datetime, hours: int) -> datetime:
        """Add hours to datetime"""
        return dt + timedelta(hours=hours)
    
    @staticmethod
    def add_minutes(dt: datetime, minutes: int) -> datetime:
        """Add minutes to datetime"""
        return dt + timedelta(minutes=minutes)
    
    @staticmethod
    def time_ago(dt: datetime) -> str:
        """Get human-readable time ago string"""
        now = DateTimeHelpers.now_utc()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "just now"


class JSONHelpers:
    """JSON utilities"""
    
    @staticmethod
    def safe_json_loads(json_string: str, default: Any = None) -> Any:
        """Safely parse JSON string"""
        try:
            return json.loads(json_string)
        except (json.JSONDecodeError, TypeError):
            return default
    
    @staticmethod
    def safe_json_dumps(data: Any, default: str = "{}") -> str:
        """Safely serialize data to JSON"""
        try:
            return json.dumps(data, default=str)
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def pretty_json(data: Any) -> str:
        """Pretty print JSON"""
        return json.dumps(data, indent=2, default=str)
    
    @staticmethod
    def flatten_json(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """Flatten nested JSON"""
        def _flatten(obj, parent_key="", sep="."):
            items = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    items.extend(_flatten(v, new_key, sep=sep).items())
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                    items.extend(_flatten(v, new_key, sep=sep).items())
            else:
                items.append((parent_key, obj))
            return dict(items)
        
        return _flatten(data, sep=separator)


class RetryHelpers:
    """Retry utilities"""
    
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Retry decorator"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"All {max_attempts} attempts failed. Last error: {e}")
                
                raise last_exception
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"All {max_attempts} attempts failed. Last error: {e}")
                
                raise last_exception
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator


class PerformanceHelpers:
    """Performance monitoring utilities"""
    
    @staticmethod
    def measure_time(func: Callable) -> Callable:
        """Measure execution time decorator"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    @staticmethod
    def measure_memory(func: Callable) -> Callable:
        """Measure memory usage decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            logger.info(f"{func.__name__} used {memory_used:.2f} MB of memory")
            return result
        
        return wrapper


class DataHelpers:
    """Data manipulation utilities"""
    
    @staticmethod
    def chunk_list(data: List[T], chunk_size: int) -> List[List[T]]:
        """Split list into chunks"""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    @staticmethod
    def remove_duplicates(data: List[T], key: Optional[Callable] = None) -> List[T]:
        """Remove duplicates from list"""
        if key is None:
            return list(dict.fromkeys(data))
        else:
            seen = set()
            result = []
            for item in data:
                key_value = key(item)
                if key_value not in seen:
                    seen.add(key_value)
                    result.append(item)
            return result
    
    @staticmethod
    def group_by(data: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group list of dictionaries by key"""
        groups = {}
        for item in data:
            group_key = item.get(key)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        return groups
    
    @staticmethod
    def sort_by(data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
        """Sort list of dictionaries by key"""
        return sorted(data, key=lambda x: x.get(key, 0), reverse=reverse)
    
    @staticmethod
    def filter_by(data: List[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
        """Filter list of dictionaries by key-value pair"""
        return [item for item in data if item.get(key) == value]


class SecurityHelpers:
    """Security utilities"""
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """Generate API key"""
        return StringHelpers.generate_random_string(length, include_symbols=False)
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate session token"""
        return StringHelpers.generate_random_string(64, include_symbols=True)
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input"""
        import html
        return html.escape(text.strip())
    
    @staticmethod
    def is_safe_filename(filename: str) -> bool:
        """Check if filename is safe"""
        import os
        import re
        
        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return False
        
        # Check for invalid characters
        if re.search(r'[<>:"|?*]', filename):
            return False
        
        # Check length
        if len(filename) > 255:
            return False
        
        return True


class PaginationHelpers:
    """Pagination utilities"""
    
    @staticmethod
    def paginate(data: List[T], page: int, page_size: int) -> Dict[str, Any]:
        """Paginate data"""
        total_count = len(data)
        total_pages = (total_count + page_size - 1) // page_size
        
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        
        paginated_data = data[start_index:end_index]
        
        return {
            "data": paginated_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }
    
    @staticmethod
    def calculate_offset(page: int, page_size: int) -> int:
        """Calculate offset for pagination"""
        return (page - 1) * page_size


class CacheHelpers:
    """Cache utilities"""
    
    @staticmethod
    def generate_cache_key(prefix: str, *args, **kwargs) -> str:
        """Generate cache key"""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            key_parts.append(str(arg))
        
        # Add keyword arguments
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")
        
        return ":".join(key_parts)
    
    @staticmethod
    def calculate_ttl(seconds: int) -> int:
        """Calculate TTL with jitter"""
        import random
        jitter = random.randint(0, 60)  # Add up to 60 seconds of jitter
        return seconds + jitter


class ErrorHelpers:
    """Error handling utilities"""
    
    @staticmethod
    def format_exception(e: Exception) -> Dict[str, Any]:
        """Format exception for logging"""
        import traceback
        
        return {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
    
    @staticmethod
    def is_retryable_error(e: Exception) -> bool:
        """Check if error is retryable"""
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            OSError,
            asyncio.TimeoutError
        )
        return isinstance(e, retryable_errors)


class ValidationHelpers:
    """Validation utilities"""
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate required fields"""
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        return missing_fields
    
    @staticmethod
    def validate_field_types(data: Dict[str, Any], field_types: Dict[str, type]) -> List[str]:
        """Validate field types"""
        type_errors = []
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                type_errors.append(f"{field} must be of type {expected_type.__name__}")
        return type_errors




