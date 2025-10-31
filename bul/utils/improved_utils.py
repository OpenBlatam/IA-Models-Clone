"""
Improved Utilities for BUL API
=============================

Enhanced utility functions with additional real-world functionality:
- Advanced validation
- Business logic utilities
- Performance optimizations
- Real-world integrations
"""

import asyncio
import time
import json
import hashlib
import re
from typing import Dict, List, Optional, Any, Union, Callable
from functools import wraps, lru_cache
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Enhanced validation functions
def validate_enhanced_email(email: str) -> Dict[str, Any]:
    """Enhanced email validation with detailed results"""
    if not email:
        return {"valid": False, "error": "Email is required"}
    
    if '@' not in email:
        return {"valid": False, "error": "Email must contain @ symbol"}
    
    if len(email) > 254:
        return {"valid": False, "error": "Email too long"}
    
    # Check for valid email pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return {"valid": False, "error": "Invalid email format"}
    
    # Extract domain for additional validation
    domain = email.split('@')[1]
    if len(domain) < 3:
        return {"valid": False, "error": "Invalid domain"}
    
    return {
        "valid": True,
        "email": email,
        "domain": domain,
        "local_part": email.split('@')[0]
    }

def validate_enhanced_phone(phone: str) -> Dict[str, Any]:
    """Enhanced phone validation with detailed results"""
    if not phone:
        return {"valid": False, "error": "Phone number is required"}
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Check length
    if len(digits) < 10:
        return {"valid": False, "error": "Phone number too short"}
    
    if len(digits) > 15:
        return {"valid": False, "error": "Phone number too long"}
    
    # Check for valid phone patterns
    if len(digits) == 10:
        return {"valid": True, "phone": phone, "digits": digits, "format": "US"}
    elif len(digits) == 11 and digits.startswith('1'):
        return {"valid": True, "phone": phone, "digits": digits, "format": "US"}
    else:
        return {"valid": True, "phone": phone, "digits": digits, "format": "International"}

def validate_enhanced_url(url: str) -> Dict[str, Any]:
    """Enhanced URL validation with detailed results"""
    if not url:
        return {"valid": False, "error": "URL is required"}
    
    if not url.startswith(('http://', 'https://')):
        return {"valid": False, "error": "URL must start with http:// or https://"}
    
    if len(url) > 2048:
        return {"valid": False, "error": "URL too long"}
    
    # Enhanced URL pattern
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    if not re.match(pattern, url):
        return {"valid": False, "error": "Invalid URL format"}
    
    # Extract URL components
    from urllib.parse import urlparse
    parsed = urlparse(url)
    
    return {
        "valid": True,
        "url": url,
        "scheme": parsed.scheme,
        "domain": parsed.netloc,
        "path": parsed.path,
        "query": parsed.query,
        "fragment": parsed.fragment
    }

# Enhanced string processing
def normalize_enhanced_text(text: str) -> Dict[str, Any]:
    """Enhanced text normalization with detailed results"""
    if not text:
        return {"normalized": "", "original_length": 0, "normalized_length": 0}
    
    original_length = len(text)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', text.strip())
    
    # Convert to lowercase
    normalized = normalized.lower()
    
    # Remove special characters but keep alphanumeric and basic punctuation
    normalized = re.sub(r'[^\w\s.,!?;:]', '', normalized)
    
    return {
        "normalized": normalized,
        "original_length": original_length,
        "normalized_length": len(normalized),
        "reduction_percentage": round((1 - len(normalized) / original_length) * 100, 2) if original_length > 0 else 0
    }

def extract_enhanced_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> Dict[str, Any]:
    """Enhanced keyword extraction with detailed results"""
    if not text:
        return {"keywords": [], "count": 0, "text_length": 0}
    
    # Normalize text
    normalized_result = normalize_enhanced_text(text)
    normalized = normalized_result["normalized"]
    
    # Split into words
    words = normalized.split()
    
    # Filter by length and remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    # Count frequency
    keyword_freq = {}
    for keyword in keywords:
        keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
    
    # Sort by frequency and limit
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [kw[0] for kw in sorted_keywords[:max_keywords]]
    
    return {
        "keywords": top_keywords,
        "count": len(top_keywords),
        "text_length": len(text),
        "keyword_density": round(len(top_keywords) / len(words) * 100, 2) if words else 0,
        "frequency": dict(sorted_keywords[:max_keywords])
    }

def calculate_enhanced_similarity(text1: str, text2: str) -> Dict[str, Any]:
    """Enhanced text similarity calculation with detailed results"""
    if not text1 or not text2:
        return {"similarity": 0.0, "method": "jaccard", "details": "Empty text provided"}
    
    # Extract keywords for both texts
    keywords1_result = extract_enhanced_keywords(text1)
    keywords2_result = extract_enhanced_keywords(text2)
    
    keywords1 = set(keywords1_result["keywords"])
    keywords2 = set(keywords2_result["keywords"])
    
    if not keywords1 and not keywords2:
        return {"similarity": 1.0, "method": "jaccard", "details": "Both texts have no keywords"}
    
    # Calculate Jaccard similarity
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    jaccard_similarity = intersection / union if union > 0 else 0.0
    
    # Calculate additional metrics
    common_keywords = list(keywords1.intersection(keywords2))
    unique_keywords1 = list(keywords1 - keywords2)
    unique_keywords2 = list(keywords2 - keywords1)
    
    return {
        "similarity": round(jaccard_similarity, 3),
        "method": "jaccard",
        "common_keywords": common_keywords,
        "unique_to_text1": unique_keywords1,
        "unique_to_text2": unique_keywords2,
        "intersection_size": intersection,
        "union_size": union
    }

# Enhanced hashing functions
def generate_enhanced_hash(data: str, algorithm: str = 'sha256') -> Dict[str, Any]:
    """Enhanced hash generation with detailed results"""
    if not data:
        return {"hash": "", "algorithm": algorithm, "data_length": 0}
    
    hash_func = getattr(hashlib, algorithm)
    hash_value = hash_func(data.encode('utf-8')).hexdigest()
    
    return {
        "hash": hash_value,
        "algorithm": algorithm,
        "data_length": len(data),
        "hash_length": len(hash_value)
    }

def generate_enhanced_token(length: int = 32, include_symbols: bool = True) -> Dict[str, Any]:
    """Enhanced token generation with detailed results"""
    import secrets
    import string
    
    if include_symbols:
        characters = string.ascii_letters + string.digits + string.punctuation
    else:
        characters = string.ascii_letters + string.digits
    
    token = ''.join(secrets.choice(characters) for _ in range(length))
    
    return {
        "token": token,
        "length": length,
        "includes_symbols": include_symbols,
        "character_set_size": len(characters)
    }

# Enhanced async utilities
async def async_map_enhanced(func: Callable[[Any], Any], items: List[Any], max_concurrent: int = 10) -> List[Any]:
    """Enhanced async mapping with concurrency control"""
    if not items:
        return []
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_func(item):
        async with semaphore:
            return await func(item)
    
    return await asyncio.gather(*[limited_func(item) for item in items])

async def async_filter_enhanced(predicate: Callable[[Any], bool], items: List[Any], max_concurrent: int = 10) -> List[Any]:
    """Enhanced async filtering with concurrency control"""
    if not items:
        return []
    
    results = await async_map_enhanced(predicate, items, max_concurrent)
    return [item for item, result in zip(items, results) if result]

async def async_batch_process_enhanced(
    items: List[Any], 
    processor: Callable[[Any], Any], 
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """Enhanced async batch processing with concurrency control"""
    if not items:
        return []
    
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await async_map_enhanced(processor, batch, max_concurrent)
        results.extend(batch_results)
    return results

# Enhanced error handling
def handle_enhanced_error(error: Exception, context: str = "", include_traceback: bool = False) -> Dict[str, Any]:
    """Enhanced error handling with detailed context"""
    error_info = {
        "error": str(error),
        "error_type": type(error).__name__,
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "success": False
    }
    
    if include_traceback:
        import traceback
        error_info["traceback"] = traceback.format_exc()
    
    return error_info

def validate_enhanced_required_fields(data: Dict[str, Any], required: List[str]) -> None:
    """Enhanced required field validation with detailed error messages"""
    missing_fields = []
    invalid_fields = []
    
    for field in required:
        if field not in data:
            missing_fields.append(field)
        elif not data[field]:
            invalid_fields.append(field)
    
    if missing_fields or invalid_fields:
        error_msg = "Validation failed: "
        if missing_fields:
            error_msg += f"Missing fields: {', '.join(missing_fields)}"
        if invalid_fields:
            error_msg += f"Invalid fields: {', '.join(invalid_fields)}"
        raise ValueError(error_msg)

def validate_enhanced_field_types(data: Dict[str, Any], types: Dict[str, type]) -> None:
    """Enhanced field type validation with detailed error messages"""
    type_errors = []
    
    for field, expected_type in types.items():
        if field in data:
            if not isinstance(data[field], expected_type):
                type_errors.append(f"{field} (expected {expected_type.__name__}, got {type(data[field]).__name__})")
    
    if type_errors:
        raise ValueError(f"Type validation failed: {', '.join(type_errors)}")

# Enhanced performance utilities
def measure_enhanced_time(func: Callable) -> Callable:
    """Enhanced time measurement with detailed metrics"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        result = await func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = _get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logging.info(f"{func.__name__} - Duration: {duration:.4f}s, Memory: {memory_delta:.2f}MB")
        
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = _get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logging.info(f"{func.__name__} - Duration: {duration:.4f}s, Memory: {memory_delta:.2f}MB")
        
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def cache_enhanced_result(ttl: int = 3600, max_size: int = 1000):
    """Enhanced result caching with size limits"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            now = time.time()
            
            # Check cache
            if cache_key in cache:
                cached_time = cache_times.get(cache_key, 0)
                if now - cached_time < ttl:
                    return cache[cache_key]
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = result
            cache_times[cache_key] = now
            
            # Clean old entries if cache is too large
            if len(cache) > max_size:
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            now = time.time()
            
            # Check cache
            if cache_key in cache:
                cached_time = cache_times.get(cache_key, 0)
                if now - cached_time < ttl:
                    return cache[cache_key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = result
            cache_times[cache_key] = now
            
            # Clean old entries if cache is too large
            if len(cache) > max_size:
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Enhanced data processing
def process_enhanced_data(data: List[Dict[str, Any]], processor: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """Enhanced data processing with detailed results"""
    if not data:
        return {"results": [], "processed_count": 0, "error_count": 0, "success_rate": 0.0}
    
    results = []
    errors = []
    
    for i, item in enumerate(data):
        try:
            result = processor(item)
            results.append(result)
        except Exception as e:
            errors.append({"index": i, "error": str(e), "item": item})
            logging.error(f"Data processing error at index {i}: {e}")
    
    processed_count = len(results)
    error_count = len(errors)
    success_rate = processed_count / len(data) if data else 0.0
    
    return {
        "results": results,
        "processed_count": processed_count,
        "error_count": error_count,
        "success_rate": round(success_rate, 3),
        "errors": errors
    }

def filter_enhanced_data(data: List[Dict[str, Any]], filters: Dict[str, Callable]) -> Dict[str, Any]:
    """Enhanced data filtering with detailed results"""
    if not data:
        return {"filtered_data": [], "original_count": 0, "filtered_count": 0, "filter_rate": 0.0}
    
    filtered_data = []
    filter_stats = {}
    
    for item in data:
        include = True
        item_stats = {}
        
        for key, filter_func in filters.items():
            if key in item:
                try:
                    filter_result = filter_func(item[key])
                    item_stats[key] = filter_result
                    if not filter_result:
                        include = False
                        break
                except Exception as e:
                    logging.error(f"Filter error for {key}: {e}")
                    item_stats[key] = False
                    include = False
                    break
        
        if include:
            filtered_data.append(item)
        
        # Update filter stats
        for key, result in item_stats.items():
            if key not in filter_stats:
                filter_stats[key] = {"passed": 0, "failed": 0}
            if result:
                filter_stats[key]["passed"] += 1
            else:
                filter_stats[key]["failed"] += 1
    
    return {
        "filtered_data": filtered_data,
        "original_count": len(data),
        "filtered_count": len(filtered_data),
        "filter_rate": round(len(filtered_data) / len(data), 3) if data else 0.0,
        "filter_stats": filter_stats
    }

# Enhanced configuration
def load_enhanced_config(config_path: str) -> Dict[str, Any]:
    """Enhanced configuration loading with validation"""
    if not Path(config_path).exists():
        logging.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ["version", "environment"]
        for field in required_fields:
            if field not in config:
                logging.warning(f"Missing required config field: {field}")
        
        return config
        
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        return {}
    except Exception as e:
        logging.error(f"Config loading error: {e}")
        return {}

def save_enhanced_config(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """Enhanced configuration saving with validation"""
    try:
        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        # Add metadata
        config_with_metadata = {
            **config,
            "last_updated": datetime.now().isoformat(),
            "version": config.get("version", "1.0.0")
        }
        
        # Save to file
        with open(config_path, 'w') as f:
            json.dump(config_with_metadata, f, indent=2)
        
        return {"success": True, "path": config_path, "size": len(json.dumps(config_with_metadata))}
        
    except Exception as e:
        logging.error(f"Config saving error: {e}")
        return {"success": False, "error": str(e)}

# Enhanced logging
def log_enhanced_info(message: str, **kwargs) -> None:
    """Enhanced info logging with structured data"""
    log_data = {
        "message": message,
        "level": "INFO",
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    logging.info(json.dumps(log_data))

def log_enhanced_error(message: str, error: Exception = None, **kwargs) -> None:
    """Enhanced error logging with structured data"""
    log_data = {
        "message": message,
        "level": "ERROR",
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    
    if error:
        log_data["error_type"] = type(error).__name__
        log_data["error_message"] = str(error)
    
    logging.error(json.dumps(log_data))

def log_enhanced_warning(message: str, **kwargs) -> None:
    """Enhanced warning logging with structured data"""
    log_data = {
        "message": message,
        "level": "WARNING",
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    logging.warning(json.dumps(log_data))

# Export enhanced functions
__all__ = [
    # Enhanced validation
    "validate_enhanced_email",
    "validate_enhanced_phone",
    "validate_enhanced_url",
    
    # Enhanced string processing
    "normalize_enhanced_text",
    "extract_enhanced_keywords",
    "calculate_enhanced_similarity",
    
    # Enhanced hashing
    "generate_enhanced_hash",
    "generate_enhanced_token",
    
    # Enhanced async utilities
    "async_map_enhanced",
    "async_filter_enhanced",
    "async_batch_process_enhanced",
    
    # Enhanced error handling
    "handle_enhanced_error",
    "validate_enhanced_required_fields",
    "validate_enhanced_field_types",
    
    # Enhanced performance
    "measure_enhanced_time",
    "cache_enhanced_result",
    
    # Enhanced data processing
    "process_enhanced_data",
    "filter_enhanced_data",
    
    # Enhanced configuration
    "load_enhanced_config",
    "save_enhanced_config",
    
    # Enhanced logging
    "log_enhanced_info",
    "log_enhanced_error",
    "log_enhanced_warning"
]












