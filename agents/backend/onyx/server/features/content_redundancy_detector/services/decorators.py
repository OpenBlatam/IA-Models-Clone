"""
Service Decorators - Cross-cutting concerns for services
Handles caching, webhooks, analytics, and error handling
Supports both sync and async functions
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, Any, Optional, Dict

logger = logging.getLogger(__name__)


def with_caching(cache_get_func: Optional[Callable] = None, cache_set_func: Optional[Callable] = None, cache_key_func: Optional[Callable] = None):
    """
    Decorator to add caching to service functions (works with both sync and async)
    
    Args:
        cache_get_func: Optional function to get cached result (defaults to auto-detect based on function name)
        cache_set_func: Optional function to cache result (defaults to auto-detect based on function name)
        cache_key_func: Optional function to generate cache key from function args
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Auto-detect cache functions based on function name if not provided
                if cache_get_func is None or cache_set_func is None:
                    try:
                        from cache import (
                            get_cached_analysis_result, cache_analysis_result,
                            get_cached_similarity_result, cache_similarity_result,
                            get_cached_quality_result, cache_quality_result
                        )
                        
                        # Map function names to cache functions
                        func_name = func.__name__
                        if "analyze" in func_name.lower() or "content" in func_name.lower():
                            cache_get = cache_get_func or get_cached_analysis_result
                            cache_set = cache_set_func or cache_analysis_result
                        elif "similarity" in func_name.lower() or "similar" in func_name.lower():
                            cache_get = cache_get_func or get_cached_similarity_result
                            cache_set = cache_set_func or cache_similarity_result
                        elif "quality" in func_name.lower():
                            cache_get = cache_get_func or get_cached_quality_result
                            cache_set = cache_set_func or cache_quality_result
                        else:
                            cache_get = cache_get_func or get_cached_analysis_result
                            cache_set = cache_set_func or cache_analysis_result
                    except ImportError:
                        cache_get = cache_get_func
                        cache_set = cache_set_func
                else:
                    cache_get = cache_get_func
                    cache_set = cache_set_func
                
                # Generate cache key
                cache_key = None
                if cache_get and cache_set:
                    if cache_key_func:
                        cache_key = cache_key_func(*args, **kwargs)
                    else:
                        # Default: use first string argument(s) as key
                        str_args = [arg for arg in args if isinstance(arg, str)]
                        if len(str_args) >= 2:
                            # For similarity functions, use both texts
                            cache_key = f"{str_args[0]}:{str_args[1]}"
                            if len(args) > 2 and isinstance(args[2], (int, float)):
                                cache_key += f":{args[2]}"  # Include threshold
                        elif len(str_args) >= 1:
                            cache_key = str_args[0]
                    
                    if cache_key:
                        try:
                            # Try different cache function signatures
                            if "similarity" in func.__name__.lower():
                                # For similarity, pass text1, text2, threshold
                                if len(args) >= 3:
                                    cached = cache_get(args[0], args[1], args[2]) if not asyncio.iscoroutinefunction(cache_get) else await cache_get(args[0], args[1], args[2])
                                else:
                                    cached = cache_get(cache_key) if not asyncio.iscoroutinefunction(cache_get) else await cache_get(cache_key)
                            else:
                                cached = (cache_get(cache_key) if not asyncio.iscoroutinefunction(cache_get) else await cache_get(cache_key)) if cache_key else None
                            
                            if cached is not None:
                                logger.debug(f"Cache hit for {func.__name__}")
                                return cached
                        except Exception as e:
                            logger.debug(f"Cache check failed: {e}")
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                if cache_set and cache_key:
                    try:
                        if "similarity" in func.__name__.lower() and len(args) >= 3:
                            # For similarity, pass text1, text2, threshold, result
                            if asyncio.iscoroutinefunction(cache_set):
                                await cache_set(args[0], args[1], args[2], result)
                            else:
                                cache_set(args[0], args[1], args[2], result)
                        else:
                            if asyncio.iscoroutinefunction(cache_set):
                                await cache_set(cache_key, result)
                            else:
                                cache_set(cache_key, result)
                    except Exception as e:
                        logger.warning(f"Failed to cache result: {e}")
                
                return result
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Auto-detect cache functions based on function name if not provided
                if cache_get_func is None or cache_set_func is None:
                    try:
                        from cache import (
                            get_cached_analysis_result, cache_analysis_result,
                            get_cached_similarity_result, cache_similarity_result,
                            get_cached_quality_result, cache_quality_result
                        )
                        
                        # Map function names to cache functions
                        func_name = func.__name__
                        if "analyze" in func_name.lower() or "content" in func_name.lower():
                            cache_get = cache_get_func or get_cached_analysis_result
                            cache_set = cache_set_func or cache_analysis_result
                        elif "similarity" in func_name.lower() or "similar" in func_name.lower():
                            cache_get = cache_get_func or get_cached_similarity_result
                            cache_set = cache_set_func or cache_similarity_result
                        elif "quality" in func_name.lower():
                            cache_get = cache_get_func or get_cached_quality_result
                            cache_set = cache_set_func or cache_quality_result
                        else:
                            cache_get = cache_get_func or get_cached_analysis_result
                            cache_set = cache_set_func or cache_analysis_result
                    except ImportError:
                        cache_get = cache_get_func
                        cache_set = cache_set_func
                else:
                    cache_get = cache_get_func
                    cache_set = cache_set_func
                
                # Generate cache key
                cache_key = None
                if cache_get and cache_set:
                    if cache_key_func:
                        cache_key = cache_key_func(*args, **kwargs)
                    else:
                        # Default: use first string argument(s) as key
                        str_args = [arg for arg in args if isinstance(arg, str)]
                        if len(str_args) >= 2:
                            # For similarity functions, use both texts
                            cache_key = f"{str_args[0]}:{str_args[1]}"
                            if len(args) > 2 and isinstance(args[2], (int, float)):
                                cache_key += f":{args[2]}"  # Include threshold
                        elif len(str_args) >= 1:
                            cache_key = str_args[0]
                    
                    if cache_key:
                        try:
                            # Try different cache function signatures
                            if "similarity" in func.__name__.lower():
                                # For similarity, pass text1, text2, threshold
                                if len(args) >= 3:
                                    cached = cache_get(args[0], args[1], args[2])
                                else:
                                    cached = cache_get(cache_key)
                            else:
                                cached = cache_get(cache_key) if cache_key else None
                            
                            if cached is not None:
                                logger.debug(f"Cache hit for {func.__name__}")
                                return cached
                        except Exception as e:
                            logger.debug(f"Cache check failed: {e}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                if cache_set and cache_key:
                    try:
                        if "similarity" in func.__name__.lower() and len(args) >= 3:
                            # For similarity, pass text1, text2, threshold, result
                            cache_set(args[0], args[1], args[2], result)
                        else:
                            cache_set(cache_key, result)
                    except Exception as e:
                        logger.warning(f"Failed to cache result: {e}")
                
                return result
            
            return sync_wrapper
    return decorator


def with_webhooks(event_type: str):
    """
    Decorator to add webhook notifications to service functions (works with both sync and async)
    
    Args:
        event_type: Webhook event type to send
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Execute function
                result = await func(*args, **kwargs)
                
                # Send webhook asynchronously
                try:
                    from webhooks import send_webhook, WebhookEvent
                    request_id = kwargs.get("request_id")
                    user_id = kwargs.get("user_id")
                    
                    asyncio.create_task(send_webhook(
                        getattr(WebhookEvent, event_type, event_type),
                        {"result": result},
                        request_id=request_id,
                        user_id=user_id
                    ))
                except Exception as e:
                    logger.warning(f"Failed to send webhook: {e}")
                
                return result
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Execute function
                result = func(*args, **kwargs)
                
                # Send webhook asynchronously (fire and forget)
                try:
                    from webhooks import send_webhook, WebhookEvent
                    request_id = kwargs.get("request_id")
                    user_id = kwargs.get("user_id")
                    
                    # Extract request_id and user_id from args if not in kwargs
                    if request_id is None and len(args) > 1:
                        # Check if second or third arg is request_id (string)
                        for arg in args[1:]:
                            if isinstance(arg, str) and len(arg) > 10:  # Likely a request_id
                                request_id = arg
                                break
                    
                    asyncio.create_task(send_webhook(
                        getattr(WebhookEvent, event_type, event_type),
                        {"result": result},
                        request_id=request_id,
                        user_id=user_id
                    ))
                except Exception as e:
                    logger.warning(f"Failed to send webhook: {e}")
                
                return result
            
            return sync_wrapper
    return decorator


def with_analytics(analytics_type: Optional[str] = None):
    """
    Decorator to add analytics tracking to service functions (works with both sync and async)
    
    Args:
        analytics_type: Type of analytics event to record (auto-detected from function name if None)
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Execute function
                result = await func(*args, **kwargs)
                
                # Record analytics
                try:
                    from analytics import record_analysis
                    event_type = analytics_type or func.__name__.replace("_", "-")
                    if asyncio.iscoroutinefunction(record_analysis):
                        await record_analysis(event_type, result)
                    else:
                        record_analysis(event_type, result)
                except Exception as e:
                    logger.warning(f"Failed to record analytics: {e}")
                
                return result
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Execute function
                result = func(*args, **kwargs)
                
                # Record analytics
                try:
                    from analytics import record_analysis
                    event_type = analytics_type or func.__name__.replace("_", "-")
                    record_analysis(event_type, result)
                except Exception as e:
                    logger.warning(f"Failed to record analytics: {e}")
                
                return result
            
            return sync_wrapper
    return decorator


def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors consistently across services (works with both sync and async)
    """
    is_async = asyncio.iscoroutinefunction(func)
    
    if is_async:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ValueError as e:
                logger.error(f"Validation error in {func.__name__}: {e}")
                raise
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Send error webhook
                try:
                    from webhooks import send_webhook, WebhookEvent
                    request_id = kwargs.get("request_id")
                    user_id = kwargs.get("user_id")
                    
                    asyncio.create_task(send_webhook(
                        WebhookEvent.SYSTEM_ERROR,
                        {"error": error_msg, "operation": func.__name__},
                        request_id=request_id,
                        user_id=user_id
                    ))
                except Exception:
                    pass
                
                raise ValueError(error_msg) from e
        
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                logger.error(f"Validation error in {func.__name__}: {e}")
                raise
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Send error webhook
                try:
                    from webhooks import send_webhook, WebhookEvent
                    request_id = kwargs.get("request_id")
                    user_id = kwargs.get("user_id")
                    
                    asyncio.create_task(send_webhook(
                        WebhookEvent.SYSTEM_ERROR,
                        {"error": error_msg, "operation": func.__name__},
                        request_id=request_id,
                        user_id=user_id
                    ))
                except Exception:
                    pass
                
                raise ValueError(error_msg) from e
        
        return sync_wrapper
