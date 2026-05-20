"""
Utils Service Implementation
"""

import re
import asyncio
import logging
import random
import uuid
import functools
from typing import Any, Callable, Dict, List
from datetime import datetime

from .base import (
    UtilityBase,
    ValidationError,
    RetryConfig,
    ConfigurationError
)

logger = logging.getLogger(__name__)


class UtilityService(UtilityBase):
    """Utility service implementation"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email"""
        if not email or not isinstance(email, str):
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string"""
        if not isinstance(value, str):
            return str(value)
        
        # Remove leading/trailing whitespace
        value = value.strip()
        
        # Remove control characters
        value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        return value
    
    @staticmethod
    async def retry(
        func: Callable,
        config: RetryConfig,
        *args,
        **kwargs
    ) -> Any:
        """Retry function with exponential backoff"""
        last_exception = None
        
        for attempt in range(config.max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except config.retry_on as e:
                last_exception = e
                if attempt < config.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = config.backoff_factor * (2 ** attempt)
                    
                    if config.jitter:
                        wait_time = wait_time * (0.5 + random.random() * 0.5)
                    
                    if config.max_wait_time:
                        wait_time = min(wait_time, config.max_wait_time)
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{config.max_retries} "
                        f"after {wait_time:.2f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for {func.__name__}: {e}")
                    raise
        
        if last_exception:
            raise last_exception
    
    @staticmethod
    def generate_id(prefix: str = "") -> str:
        """Generate unique ID"""
        id_str = str(uuid.uuid4())
        if prefix:
            return f"{prefix}_{id_str}"
        return id_str
    
    @staticmethod
    def format_datetime(dt: datetime, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime"""
        return dt.strftime(format)
    
    @staticmethod
    def parse_datetime(date_string: str, format: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """Parse datetime string"""
        return datetime.strptime(date_string, format)
    
    @staticmethod
    def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = UtilityService.deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Chunk list into smaller lists"""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL"""
        pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(pattern.match(url))
    
    @staticmethod
    def truncate_string(value: str, max_length: int, suffix: str = "...") -> str:
        """Truncate string to max length"""
        if len(value) <= max_length:
            return value
        return value[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def safe_get(dictionary: Dict, *keys, default: Any = None) -> Any:
        """Safely get nested dictionary value"""
        result = dictionary
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
                if result is None:
                    return default
            else:
                return default
        return result


# Decorators
def retry_on_failure(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for automatic retry on failure"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                backoff_factor=backoff_factor
            )
            return await UtilityService.retry(func, config, *args, **kwargs)
        return wrapper
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log execution time"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper


def validate_input(**validators):
    """Decorator to validate function inputs"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate kwargs
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    if not validator(kwargs[param_name]):
                        raise ValidationError(f"Invalid value for {param_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

