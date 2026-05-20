"""
Performance optimization utilities.
"""

from typing import List, Callable, Any, Optional
from functools import lru_cache
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def memoize_with_ttl(ttl_seconds: int = 300):
    """
    Decorator to memoize function results with TTL.
    
    Args:
        ttl_seconds: Time to live in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if cached and still valid
            if key in cache:
                cache_time = cache_times.get(key)
                if cache_time and (datetime.now() - cache_time).total_seconds() < ttl_seconds:
                    return cache[key]
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = datetime.now()
            
            # Clean old entries
            now = datetime.now()
            expired_keys = [
                k for k, t in cache_times.items()
                if (now - t).total_seconds() >= ttl_seconds
            ]
            for k in expired_keys:
                cache.pop(k, None)
                cache_times.pop(k, None)
            
            return result
        
        return wrapper
    return decorator


def batch_get(
    items: List[Any],
    batch_size: int = 100,
    get_func: Optional[Callable] = None
) -> List[Any]:
    """
    Get items in batches to avoid memory issues.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        get_func: Optional function to apply to each batch
        
    Returns:
        List of processed items
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if get_func:
            results.extend(get_func(batch))
        else:
            results.extend(batch)
    
    return results


def optimize_query(query, limit: Optional[int] = None):
    """
    Optimize a query by adding limits and optimizations.
    
    Args:
        query: SQLAlchemy query
        limit: Optional limit to apply
        
    Returns:
        Optimized query
    """
    if limit:
        query = query.limit(limit)
    
    # Add optimizations
    # query = query.options(joinedload(...))  # Eager loading if needed
    
    return query


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def batch_fetch(
    items: List[Any],
    fetch_func: Callable,
    batch_size: int = 100
) -> List[Any]:
    """
    Fetch items in batches using a fetch function.
    
    Args:
        items: List of items/IDs to fetch
        fetch_func: Function to fetch a batch of items
        batch_size: Size of each batch
        
    Returns:
        List of fetched items
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = fetch_func(batch)
        if batch_results:
            results.extend(batch_results)
            
    return results






