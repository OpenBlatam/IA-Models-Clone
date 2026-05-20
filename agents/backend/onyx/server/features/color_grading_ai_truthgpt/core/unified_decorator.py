"""
Unified Decorator for Color Grading AI
=======================================

Unified decorator combining tracing, performance, error handling, validation, and caching.
"""

import logging
import functools
import inspect
from typing import Callable, Any, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedDecorator:
    """
    Unified decorator combining multiple cross-cutting concerns.
    
    Features:
    - Distributed tracing
    - Performance tracking
    - Error handling
    - Input validation
    - Result caching
    - Metrics collection
    """
    
    def __init__(
        self,
        operation_name: Optional[str] = None,
        enable_tracing: bool = True,
        enable_performance: bool = True,
        enable_error_handling: bool = True,
        enable_validation: bool = False,
        enable_caching: bool = False,
        validator: Optional[Callable] = None,
        cache_key_func: Optional[Callable] = None,
        cache_ttl: int = 3600,
        default_return: Any = None,
        log_errors: bool = True,
    ):
        """
        Initialize unified decorator.
        
        Args:
            operation_name: Operation name for tracing/metrics
            enable_tracing: Enable distributed tracing
            enable_performance: Enable performance tracking
            enable_error_handling: Enable error handling
            enable_validation: Enable input validation
            enable_caching: Enable result caching
            validator: Validation function
            cache_key_func: Cache key generation function
            cache_ttl: Cache TTL in seconds
            default_return: Default return value on error
            log_errors: Whether to log errors
        """
        self.operation_name = operation_name
        self.enable_tracing = enable_tracing
        self.enable_performance = enable_performance
        self.enable_error_handling = enable_error_handling
        self.enable_validation = enable_validation
        self.enable_caching = enable_caching
        self.validator = validator
        self.cache_key_func = cache_key_func
        self.cache_ttl = cache_ttl
        self.default_return = default_return
        self.log_errors = log_errors
    
    def __call__(self, func: Callable) -> Callable:
        """Apply decorator to function."""
        operation_name = self.operation_name or func.__name__
        
        if inspect.iscoroutinefunction(func):
            return self._async_wrapper(func, operation_name)
        else:
            return self._sync_wrapper(func, operation_name)
    
    def _async_wrapper(self, func: Callable, operation_name: str):
        """Async wrapper."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get services from first argument if available
            services = self._get_services(args)
            
            # Validation
            if self.enable_validation and self.validator:
                if not self.validator(*args, **kwargs):
                    raise ValueError(f"Validation failed for {operation_name}")
            
            # Check cache
            cache_key = None
            if self.enable_caching and services.get("cache"):
                cache_key = self._get_cache_key(func, args, kwargs)
                cached = await services["cache"].get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for {operation_name}")
                    return cached
            
            # Start tracing
            span = None
            if self.enable_tracing and services.get("tracer"):
                span = services["tracer"].start_span(operation_name)
            
            # Start performance tracking
            perf_tracker = services.get("performance_tracker")
            if self.enable_performance and perf_tracker:
                perf_start = datetime.now()
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Record success metrics
                duration = None
                if perf_tracker and self.enable_performance:
                    duration = (datetime.now() - perf_start).total_seconds()
                    perf_tracker.record_timing(operation_name, duration)
                
                # End tracing
                if span:
                    if duration:
                        services["tracer"].set_attribute(span.span_id, "duration_ms", duration * 1000)
                    from ..services.distributed_tracing import SpanStatus
                    services["tracer"].end_span(span.span_id, status=SpanStatus.OK)
                
                # Cache result
                if self.enable_caching and cache_key and services.get("cache"):
                    await services["cache"].set(cache_key, result, ttl=self.cache_ttl)
                
                return result
            
            except Exception as e:
                # Record error metrics
                duration = None
                if perf_tracker and self.enable_performance:
                    duration = (datetime.now() - perf_start).total_seconds()
                    perf_tracker.record_timing(operation_name, duration)
                
                # Handle error
                if self.enable_error_handling:
                    error_handler = services.get("error_handler")
                    if error_handler:
                        from .error_handler import ErrorContext
                        context = ErrorContext(operation=operation_name)
                        error_handler.handle_error(e, context, reraise=False)
                    
                    if self.log_errors:
                        logger.error(f"Error in {operation_name}: {e}")
                    
                    # End tracing with error
                    if span:
                        from ..services.distributed_tracing import SpanStatus
                        services["tracer"].end_span(span.span_id, status=SpanStatus.ERROR, error=e)
                    
                    return self.default_return
                else:
                    # End tracing with error
                    if span:
                        from ..services.distributed_tracing import SpanStatus
                        services["tracer"].end_span(span.span_id, status=SpanStatus.ERROR, error=e)
                    raise
        
        return wrapper
    
    def _sync_wrapper(self, func: Callable, operation_name: str):
        """Sync wrapper."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Similar to async but without await
            services = self._get_services(args)
            
            # Validation
            if self.enable_validation and self.validator:
                if not self.validator(*args, **kwargs):
                    raise ValueError(f"Validation failed for {operation_name}")
            
            # Check cache (sync)
            cache_key = None
            if self.enable_caching and services.get("cache"):
                cache_key = self._get_cache_key(func, args, kwargs)
                cached = services["cache"].get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for {operation_name}")
                    return cached
            
            # Start tracing
            span = None
            if self.enable_tracing and services.get("tracer"):
                span = services["tracer"].start_span(operation_name)
            
            # Start performance tracking
            perf_tracker = services.get("performance_tracker")
            if self.enable_performance and perf_tracker:
                perf_start = datetime.now()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success metrics
                duration = None
                if perf_tracker and self.enable_performance:
                    duration = (datetime.now() - perf_start).total_seconds()
                    perf_tracker.record_timing(operation_name, duration)
                
                # End tracing
                if span:
                    if duration:
                        services["tracer"].set_attribute(span.span_id, "duration_ms", duration * 1000)
                    from ..services.distributed_tracing import SpanStatus
                    services["tracer"].end_span(span.span_id, status=SpanStatus.OK)
                
                # Cache result
                if self.enable_caching and cache_key and services.get("cache"):
                    services["cache"].set(cache_key, result, ttl=self.cache_ttl)
                
                return result
            
            except Exception as e:
                # Record error metrics
                duration = None
                if perf_tracker and self.enable_performance:
                    duration = (datetime.now() - perf_start).total_seconds()
                    perf_tracker.record_timing(operation_name, duration)
                
                # Handle error
                if self.enable_error_handling:
                    error_handler = services.get("error_handler")
                    if error_handler:
                        from .error_handler import ErrorContext
                        context = ErrorContext(operation=operation_name)
                        error_handler.handle_error(e, context, reraise=False)
                    
                    if self.log_errors:
                        logger.error(f"Error in {operation_name}: {e}")
                    
                    # End tracing with error
                    if span:
                        from ..services.distributed_tracing import SpanStatus
                        services["tracer"].end_span(span.span_id, status=SpanStatus.ERROR, error=e)
                    
                    return self.default_return
                else:
                    # End tracing with error
                    if span:
                        from ..services.distributed_tracing import SpanStatus
                        services["tracer"].end_span(span.span_id, status=SpanStatus.ERROR, error=e)
                    raise
        
        return wrapper
    
    def _get_services(self, args: tuple) -> Dict[str, Any]:
        """Extract services from function arguments."""
        services = {}
        
        if args:
            # Try to get services from first argument (usually self)
            obj = args[0]
            if hasattr(obj, 'services'):
                services = obj.services
            elif hasattr(obj, 'service_manager'):
                services = obj.service_manager.get_all_services()
            elif hasattr(obj, 'service_accessor'):
                # Try to get individual services
                if hasattr(obj.service_accessor, 'cache'):
                    services["cache"] = obj.service_accessor.cache
                if hasattr(obj.service_accessor, 'tracer'):
                    services["tracer"] = obj.service_accessor.tracer
                if hasattr(obj.service_accessor, 'performance_tracker'):
                    services["performance_tracker"] = obj.service_accessor.performance_tracker
                if hasattr(obj.service_accessor, 'error_handler'):
                    services["error_handler"] = obj.service_accessor.error_handler
        
        return services
    
    def _get_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key."""
        if self.cache_key_func:
            return self.cache_key_func(*args, **kwargs)
        
        # Default cache key
        import hashlib
        import json
        key_data = {
            "func": func.__name__,
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


def unified(
    operation_name: Optional[str] = None,
    enable_tracing: bool = True,
    enable_performance: bool = True,
    enable_error_handling: bool = True,
    enable_validation: bool = False,
    enable_caching: bool = False,
    validator: Optional[Callable] = None,
    cache_key_func: Optional[Callable] = None,
    cache_ttl: int = 3600,
    default_return: Any = None,
    log_errors: bool = True,
):
    """
    Unified decorator combining multiple cross-cutting concerns.
    
    Args:
        operation_name: Operation name
        enable_tracing: Enable distributed tracing
        enable_performance: Enable performance tracking
        enable_error_handling: Enable error handling
        enable_validation: Enable input validation
        enable_caching: Enable result caching
        validator: Validation function
        cache_key_func: Cache key generation function
        cache_ttl: Cache TTL in seconds
        default_return: Default return value on error
        log_errors: Whether to log errors
    
    Example:
        @unified(
            operation_name="process_video",
            enable_tracing=True,
            enable_performance=True,
            enable_caching=True,
            cache_ttl=3600
        )
        async def process_video(self, video_path: str):
            # Function implementation
            pass
    """
    decorator = UnifiedDecorator(
        operation_name=operation_name,
        enable_tracing=enable_tracing,
        enable_performance=enable_performance,
        enable_error_handling=enable_error_handling,
        enable_validation=enable_validation,
        enable_caching=enable_caching,
        validator=validator,
        cache_key_func=cache_key_func,
        cache_ttl=cache_ttl,
        default_return=default_return,
        log_errors=log_errors,
    )
    return decorator

