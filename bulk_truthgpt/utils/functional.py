"""
Functional Utilities
===================

Ultra-modular functional utilities following Flask best practices.
"""

import time
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar, Generic
from functools import wraps, lru_cache
from datetime import datetime, timedelta
import uuid
import re
import os

logger = logging.getLogger(__name__)
T = TypeVar('T')

# Validation utilities
def validate_email(email: str) -> bool:
    """
    Validate email address.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> bool:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not password or len(password) < 8:
        return False
    
    # Check for at least one uppercase, lowercase, digit, and special character
    has_upper = bool(re.search(r'[A-Z]', password))
    has_lower = bool(re.search(r'[a-z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    
    return has_upper and has_lower and has_digit and has_special

def validate_username(username: str) -> bool:
    """
    Validate username.
    
    Args:
        username: Username to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not username or len(username) < 3 or len(username) > 30:
        return False
    
    # Only alphanumeric and underscores allowed
    pattern = r'^[a-zA-Z0-9_]+$'
    return bool(re.match(pattern, username))

# String utilities
def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())

def generate_hash(data: str, algorithm: str = 'sha256') -> str:
    """
    Generate hash for data.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hash string
    """
    if algorithm == 'md5':
        return hashlib.md5(data.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(data.encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(data.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def sanitize_string(text: str) -> str:
    """
    Sanitize string for safe usage.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)
    return sanitized.strip()

# Time utilities
def get_current_timestamp() -> float:
    """Get current timestamp."""
    return time.time()

def format_timestamp(timestamp: float, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format timestamp to string.
    
    Args:
        timestamp: Timestamp to format
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.fromtimestamp(timestamp).strftime(format_str)

def is_timestamp_valid(timestamp: float, max_age_seconds: int = 3600) -> bool:
    """
    Check if timestamp is valid (not too old).
    
    Args:
        timestamp: Timestamp to check
        max_age_seconds: Maximum age in seconds
        
    Returns:
        True if valid, False otherwise
    """
    current_time = get_current_timestamp()
    return (current_time - timestamp) <= max_age_seconds

# Data utilities
def safe_json_loads(data: str, default: Any = None) -> Any:
    """
    Safely load JSON data.
    
    Args:
        data: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON data or default value
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default: str = '{}') -> str:
    """
    Safely dump data to JSON.
    
    Args:
        data: Data to serialize
        default: Default JSON string if serialization fails
        
    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return default

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

# Functional programming utilities
def pipe(*functions: Callable) -> Callable:
    """
    Create a function pipeline.
    
    Args:
        functions: Functions to chain
        
    Returns:
        Chained function
    """
    def pipeline(value):
        for func in functions:
            value = func(value)
        return value
    return pipeline

def compose(*functions: Callable) -> Callable:
    """
    Compose functions (right to left).
    
    Args:
        functions: Functions to compose
        
    Returns:
        Composed function
    """
    def composed(value):
        for func in reversed(functions):
            value = func(value)
        return value
    return composed

def curry(func: Callable, *args, **kwargs) -> Callable:
    """
    Curry a function with partial arguments.
    
    Args:
        func: Function to curry
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Curried function
    """
    @wraps(func)
    def curried(*new_args, **new_kwargs):
        return func(*(args + new_args), **{**kwargs, **new_kwargs})
    return curried

# Caching utilities
@lru_cache(maxsize=128)
def cached_function(func: Callable) -> Callable:
    """
    Cache function results.
    
    Args:
        func: Function to cache
        
    Returns:
        Cached function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def memoize(func: Callable) -> Callable:
    """
    Memoize function results.
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

# Error handling utilities
def handle_errors(func: Callable) -> Callable:
    """
    Handle errors in function execution.
    
    Args:
        func: Function to wrap
        
    Returns:
        Error-handled function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0) -> Callable:
    """
    Retry function on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries
        
    Returns:
        Retry decorator
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}): {str(e)}")
                    time.sleep(delay)
        return wrapper
    return decorator

# Performance utilities
def measure_time(func: Callable) -> Callable:
    """
    Measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Time-measured function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"Function {func.__name__} took {end_time - start_time:.3f} seconds")
        return result
    return wrapper

def rate_limit(calls_per_second: float = 1.0) -> Callable:
    """
    Rate limit function calls.
    
    Args:
        calls_per_second: Maximum calls per second
        
    Returns:
        Rate-limited function
    """
    last_called = [0.0]
    min_interval = 1.0 / calls_per_second
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            time_since_last = now - last_called[0]
            
            if time_since_last < min_interval:
                time.sleep(min_interval - time_since_last)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Data transformation utilities
def transform_data(data: Any, transformer: Callable) -> Any:
    """
    Transform data using a transformer function.
    
    Args:
        data: Data to transform
        transformer: Transformation function
        
    Returns:
        Transformed data
    """
    try:
        return transformer(data)
    except Exception as e:
        logger.error(f"Data transformation failed: {str(e)}")
        return data

def filter_data(data: List[Any], predicate: Callable) -> List[Any]:
    """
    Filter data using a predicate function.
    
    Args:
        data: Data to filter
        predicate: Predicate function
        
    Returns:
        Filtered data
    """
    return [item for item in data if predicate(item)]

def map_data(data: List[Any], mapper: Callable) -> List[Any]:
    """
    Map data using a mapper function.
    
    Args:
        data: Data to map
        mapper: Mapper function
        
    Returns:
        Mapped data
    """
    return [mapper(item) for item in data]

def reduce_data(data: List[Any], reducer: Callable, initial: Any = None) -> Any:
    """
    Reduce data using a reducer function.
    
    Args:
        data: Data to reduce
        reducer: Reducer function
        initial: Initial value
        
    Returns:
        Reduced data
    """
    if not data:
        return initial
    
    result = initial
    for item in data:
        result = reducer(result, item)
    return result

# Environment utilities
def get_env_var(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable with validation.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        Environment variable value
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} not set")
    
    return value

def is_development() -> bool:
    """Check if running in development mode."""
    return os.getenv('FLASK_ENV', 'development') == 'development'

def is_production() -> bool:
    """Check if running in production mode."""
    return os.getenv('FLASK_ENV', 'development') == 'production'

def is_testing() -> bool:
    """Check if running in testing mode."""
    return os.getenv('FLASK_ENV', 'development') == 'testing'

# Logging utilities
def log_function_call(func: Callable) -> Callable:
    """
    Log function calls.
    
    Args:
        func: Function to log
        
    Returns:
        Logged function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper

def log_performance(func: Callable) -> Callable:
    """
    Log function performance.
    
    Args:
        func: Function to log
        
    Returns:
        Performance-logged function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"Function {func.__name__} performance: {end_time - start_time:.3f}s")
        return result
    return wrapper