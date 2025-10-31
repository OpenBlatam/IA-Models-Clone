"""
Ultra-Fast Utilities - Maximum Performance
=========================================

Ultra-optimized utility functions following expert guidelines:
- Pure functional programming
- Maximum async performance
- Minimal overhead
- Ultra-fast execution
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from functools import wraps, lru_cache, partial
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
from pathlib import Path

# Ultra-fast async utilities
T = TypeVar('T')
R = TypeVar('R')

# Ultra-fast data processing
def process_data_ultra_fast(data: List[Dict[str, Any]], processor: Callable[[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ultra-fast data processing with early returns"""
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

async def async_process_data_ultra_fast(data: List[Dict[str, Any]], processor: Callable[[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ultra-fast async data processing"""
    if not data:
        return []
    
    tasks = [processor(item) for item in data]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = [result for result in results if not isinstance(result, Exception)]
    return valid_results

# Ultra-fast validation
def validate_email_ultra_fast(email: str) -> bool:
    """Ultra-fast email validation with early returns"""
    if not email or '@' not in email:
        return False
    
    if len(email) > 254:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone_ultra_fast(phone: str) -> bool:
    """Ultra-fast phone validation with early returns"""
    if not phone:
        return False
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Check length
    if len(digits) < 10 or len(digits) > 15:
        return False
    
    return True

def validate_url_ultra_fast(url: str) -> bool:
    """Ultra-fast URL validation with early returns"""
    if not url or not url.startswith(('http://', 'https://')):
        return False
    
    if len(url) > 2048:
        return False
    
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    return bool(re.match(pattern, url))

# Ultra-fast string processing
def normalize_text_ultra_fast(text: str) -> str:
    """Ultra-fast text normalization"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def extract_keywords_ultra_fast(text: str, min_length: int = 3) -> List[str]:
    """Ultra-fast keyword extraction"""
    if not text:
        return []
    
    # Normalize text
    normalized = normalize_text_ultra_fast(text)
    
    # Split into words
    words = normalized.split()
    
    # Filter by length and remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    return list(set(keywords))  # Remove duplicates

def calculate_similarity_ultra_fast(text1: str, text2: str) -> float:
    """Ultra-fast text similarity calculation"""
    if not text1 or not text2:
        return 0.0
    
    # Extract keywords
    keywords1 = set(extract_keywords_ultra_fast(text1))
    keywords2 = set(extract_keywords_ultra_fast(text2))
    
    if not keywords1 and not keywords2:
        return 1.0
    
    # Calculate Jaccard similarity
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0

# Ultra-fast hashing
def generate_hash_ultra_fast(data: str, algorithm: str = 'sha256') -> str:
    """Ultra-fast hash generation"""
    if not data:
        return ""
    
    hash_func = getattr(hashlib, algorithm)
    return hash_func(data.encode('utf-8')).hexdigest()

def generate_token_ultra_fast(length: int = 32) -> str:
    """Ultra-fast token generation"""
    import secrets
    return secrets.token_urlsafe(length)

# Ultra-fast caching
@lru_cache(maxsize=1000)
def get_cached_value(key: str) -> Any:
    """Ultra-fast cached value retrieval"""
    return None

def set_cached_value(key: str, value: Any, ttl: int = 3600) -> None:
    """Ultra-fast cached value setting"""
    # This would integrate with actual cache system
    pass

# Ultra-fast async utilities
async def async_map_ultra_fast(func: Callable[[T], R], items: List[T]) -> List[R]:
    """Ultra-fast async mapping"""
    if not items:
        return []
    
    return await asyncio.gather(*[func(item) for item in items])

async def async_filter_ultra_fast(predicate: Callable[[T], bool], items: List[T]) -> List[T]:
    """Ultra-fast async filtering"""
    if not items:
        return []
    
    results = await async_map_ultra_fast(predicate, items)
    return [item for item, result in zip(items, results) if result]

async def async_reduce_ultra_fast(func: Callable[[R, T], R], items: List[T], initial: R) -> R:
    """Ultra-fast async reduction"""
    result = initial
    for item in items:
        result = await func(result, item)
    return result

async def async_batch_process_ultra_fast(
    items: List[T], 
    processor: Callable[[T], R], 
    batch_size: int = 10
) -> List[R]:
    """Ultra-fast async batch processing"""
    if not items:
        return []
    
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await async_map_ultra_fast(processor, batch)
        results.extend(batch_results)
    return results

# Ultra-fast error handling
def handle_error_ultra_fast(error: Exception, context: str = "") -> Dict[str, Any]:
    """Ultra-fast error handling"""
    return {
        "error": str(error),
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "success": False
    }

def validate_required_fields_ultra_fast(data: Dict[str, Any], required: List[str]) -> None:
    """Ultra-fast required field validation with early returns"""
    for field in required:
        if field not in data or not data[field]:
            raise ValueError(f"Required field missing: {field}")

def validate_field_types_ultra_fast(data: Dict[str, Any], types: Dict[str, type]) -> None:
    """Ultra-fast field type validation with early returns"""
    for field, expected_type in types.items():
        if field in data and not isinstance(data[field], expected_type):
            raise ValueError(f"Field {field} must be of type {expected_type.__name__}")

# Ultra-fast performance utilities
def measure_time_ultra_fast(func: Callable) -> Callable:
    """Ultra-fast time measurement decorator"""
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

def cache_result_ultra_fast(ttl: int = 3600):
    """Ultra-fast result caching decorator"""
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

# Ultra-fast data transformation
def transform_data_ultra_fast(data: List[Dict[str, Any]], transformations: Dict[str, Callable]) -> List[Dict[str, Any]]:
    """Ultra-fast data transformation"""
    if not data:
        return []
    
    results = []
    for item in data:
        transformed_item = {}
        
        for key, value in item.items():
            if key in transformations:
                try:
                    transformed_item[key] = transformations[key](value)
                except Exception as e:
                    logging.error(f"Transformation error for {key}: {e}")
                    transformed_item[key] = value
            else:
                transformed_item[key] = value
        
        results.append(transformed_item)
    
    return results

def filter_data_ultra_fast(data: List[Dict[str, Any]], filters: Dict[str, Callable]) -> List[Dict[str, Any]]:
    """Ultra-fast data filtering"""
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

def aggregate_data_ultra_fast(data: List[Dict[str, Any]], group_by: str, aggregations: Dict[str, Callable]) -> Dict[Any, Dict[str, Any]]:
    """Ultra-fast data aggregation"""
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

# Ultra-fast configuration
def load_config_ultra_fast(config_path: str) -> Dict[str, Any]:
    """Ultra-fast configuration loading"""
    if not Path(config_path).exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Config loading error: {e}")
        return {}

def save_config_ultra_fast(config: Dict[str, Any], config_path: str) -> bool:
    """Ultra-fast configuration saving"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Config saving error: {e}")
        return False

# Ultra-fast logging
def log_ultra_fast(message: str, level: str = "INFO", **kwargs) -> None:
    """Ultra-fast logging"""
    log_data = {
        "message": message,
        "level": level,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    
    if level == "ERROR":
        logging.error(json.dumps(log_data))
    elif level == "WARNING":
        logging.warning(json.dumps(log_data))
    else:
        logging.info(json.dumps(log_data))

# Export ultra-fast utilities
__all__ = [
    # Data processing
    "process_data_ultra_fast",
    "async_process_data_ultra_fast",
    
    # Validation
    "validate_email_ultra_fast",
    "validate_phone_ultra_fast",
    "validate_url_ultra_fast",
    
    # String processing
    "normalize_text_ultra_fast",
    "extract_keywords_ultra_fast",
    "calculate_similarity_ultra_fast",
    
    # Hashing
    "generate_hash_ultra_fast",
    "generate_token_ultra_fast",
    
    # Caching
    "get_cached_value",
    "set_cached_value",
    
    # Async utilities
    "async_map_ultra_fast",
    "async_filter_ultra_fast",
    "async_reduce_ultra_fast",
    "async_batch_process_ultra_fast",
    
    # Error handling
    "handle_error_ultra_fast",
    "validate_required_fields_ultra_fast",
    "validate_field_types_ultra_fast",
    
    # Performance
    "measure_time_ultra_fast",
    "cache_result_ultra_fast",
    
    # Data transformation
    "transform_data_ultra_fast",
    "filter_data_ultra_fast",
    "aggregate_data_ultra_fast",
    
    # Configuration
    "load_config_ultra_fast",
    "save_config_ultra_fast",
    
    # Logging
    "log_ultra_fast"
]












