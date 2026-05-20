"""
Real-World Utilities for BUL API
===============================

Production-ready utility functions following FastAPI best practices:
- Pure functional programming
- Early returns and guard clauses
- Async/await throughout
- Real-world error handling
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from functools import wraps, lru_cache
from datetime import datetime
import logging
import re

# Real-world validation functions
def validate_email(email: str) -> bool:
    """Validate email address with early returns"""
    if not email or '@' not in email:
        return False
    
    if len(email) > 254:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone: str) -> bool:
    """Validate phone number with early returns"""
    if not phone:
        return False
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Check length
    if len(digits) < 10 or len(digits) > 15:
        return False
    
    return True

def validate_url(url: str) -> bool:
    """Validate URL with early returns"""
    if not url or not url.startswith(('http://', 'https://')):
        return False
    
    if len(url) > 2048:
        return False
    
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    return bool(re.match(pattern, url))

# Real-world string processing
def normalize_text(text: str) -> str:
    """Normalize text for processing"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text"""
    if not text:
        return []
    
    # Normalize text
    normalized = normalize_text(text)
    
    # Split into words
    words = normalized.split()
    
    # Filter by length and remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    return list(set(keywords))  # Remove duplicates

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using Jaccard similarity"""
    if not text1 or not text2:
        return 0.0
    
    # Extract keywords
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    if not keywords1 and not keywords2:
        return 1.0
    
    # Calculate Jaccard similarity
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0

# Real-world hashing functions
def generate_hash(data: str, algorithm: str = 'sha256') -> str:
    """Generate hash for data"""
    if not data:
        return ""
    
    hash_func = getattr(hashlib, algorithm)
    return hash_func(data.encode('utf-8')).hexdigest()

def generate_token(length: int = 32) -> str:
    """Generate secure random token"""
    import secrets
    return secrets.token_urlsafe(length)

# Real-world caching
@lru_cache(maxsize=1000)
def get_cached_value(key: str) -> Any:
    """Get cached value"""
    return None

def set_cached_value(key: str, value: Any, ttl: int = 3600) -> None:
    """Set cached value"""
    # This would integrate with actual cache system
    pass

# Real-world async utilities
async def async_map(func: Callable[[Any], Any], items: List[Any]) -> List[Any]:
    """Apply function to each item in list asynchronously"""
    if not items:
        return []
    
    return await asyncio.gather(*[func(item) for item in items])

async def async_filter(predicate: Callable[[Any], bool], items: List[Any]) -> List[Any]:
    """Filter list asynchronously"""
    if not items:
        return []
    
    results = await async_map(predicate, items)
    return [item for item, result in zip(items, results) if result]

async def async_batch_process(
    items: List[Any], 
    processor: Callable[[Any], Any], 
    batch_size: int = 10
) -> List[Any]:
    """Process items in batches asynchronously"""
    if not items:
        return []
    
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await async_map(processor, batch)
        results.extend(batch_results)
    return results

# Real-world error handling
def handle_error(error: Exception, context: str = "") -> Dict[str, Any]:
    """Handle errors with context"""
    return {
        "error": str(error),
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "success": False
    }

def validate_required_fields(data: Dict[str, Any], required: List[str]) -> None:
    """Validate required fields with early returns"""
    for field in required:
        if field not in data or not data[field]:
            raise ValueError(f"Required field missing: {field}")

def validate_field_types(data: Dict[str, Any], types: Dict[str, type]) -> None:
    """Validate field types with early returns"""
    for field, expected_type in types.items():
        if field in data and not isinstance(data[field], expected_type):
            raise ValueError(f"Field {field} must be of type {expected_type.__name__}")

# Real-world performance utilities
def measure_time(func: Callable) -> Callable:
    """Measure function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        logging.info(f"{func.__name__} took {duration:.4f}s")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logging.info(f"{func.__name__} took {duration:.4f}s")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def cache_result(ttl: int = 3600):
    """Cache function results"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            if cache_key in cache:
                cached_time, result = cache[cache_key]
                if time.time() - cached_time < ttl:
                    return result
            
            result = await func(*args, **kwargs)
            cache[cache_key] = (time.time(), result)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            if cache_key in cache:
                cached_time, result = cache[cache_key]
                if time.time() - cached_time < ttl:
                    return result
            
            result = func(*args, **kwargs)
            cache[cache_key] = (time.time(), result)
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Real-world data processing
def process_data(data: List[Dict[str, Any]], processor: Callable[[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process data with error handling"""
    if not data:
        return []
    
    results = []
    for item in data:
        try:
            result = processor(item)
            results.append(result)
        except Exception as e:
            logging.error(f"Data processing error: {e}")
            continue
    
    return results

def filter_data(data: List[Dict[str, Any]], filters: Dict[str, Callable]) -> List[Dict[str, Any]]:
    """Filter data with error handling"""
    if not data:
        return []
    
    results = []
    for item in data:
        include = True
        
        for key, filter_func in filters.items():
            if key in item:
                try:
                    if not filter_func(item[key]):
                        include = False
                        break
                except Exception as e:
                    logging.error(f"Filter error for {key}: {e}")
                    include = False
                    break
        
        if include:
            results.append(item)
    
    return results

def aggregate_data(data: List[Dict[str, Any]], group_by: str, aggregations: Dict[str, Callable]) -> Dict[Any, Dict[str, Any]]:
    """Aggregate data with error handling"""
    if not data:
        return {}
    
    grouped = {}
    for item in data:
        key = item.get(group_by)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)
    
    result = {}
    for key, items in grouped.items():
        aggregated = {}
        
        for field, agg_func in aggregations.items():
            values = [item.get(field) for item in items if field in item]
            if values:
                try:
                    aggregated[field] = agg_func(values)
                except Exception as e:
                    logging.error(f"Aggregation error for {field}: {e}")
                    aggregated[field] = None
        
        result[key] = aggregated
    
    return result

# Real-world configuration
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    if not Path(config_path).exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Config loading error: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """Save configuration to file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Config saving error: {e}")
        return False

# Real-world logging
def log_info(message: str, **kwargs) -> None:
    """Log info message"""
    log_data = {
        "message": message,
        "level": "INFO",
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    logging.info(json.dumps(log_data))

def log_error(message: str, **kwargs) -> None:
    """Log error message"""
    log_data = {
        "message": message,
        "level": "ERROR",
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    logging.error(json.dumps(log_data))

def log_warning(message: str, **kwargs) -> None:
    """Log warning message"""
    log_data = {
        "message": message,
        "level": "WARNING",
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    logging.warning(json.dumps(log_data))

# Export functions
__all__ = [
    # Validation
    "validate_email",
    "validate_phone",
    "validate_url",
    
    # String processing
    "normalize_text",
    "extract_keywords",
    "calculate_similarity",
    
    # Hashing
    "generate_hash",
    "generate_token",
    
    # Caching
    "get_cached_value",
    "set_cached_value",
    
    # Async utilities
    "async_map",
    "async_filter",
    "async_batch_process",
    
    # Error handling
    "handle_error",
    "validate_required_fields",
    "validate_field_types",
    
    # Performance
    "measure_time",
    "cache_result",
    
    # Data processing
    "process_data",
    "filter_data",
    "aggregate_data",
    
    # Configuration
    "load_config",
    "save_config",
    
    # Logging
    "log_info",
    "log_error",
    "log_warning"
]












