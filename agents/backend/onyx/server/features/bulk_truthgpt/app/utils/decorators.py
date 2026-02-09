"""
Advanced decorators for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import time
import functools
import logging
import asyncio
from typing import Callable, Any, Dict, Optional, Union
from flask import request, jsonify, current_app, g
from marshmallow import ValidationError
from functools import wraps

logger = logging.getLogger(__name__)

def performance_monitor(func: Callable) -> Callable:
    """Decorator for performance monitoring with detailed metrics."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        request_id = getattr(g, 'request_id', 'unknown')
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Log performance metrics
            logger.info(f"⚡ {func.__name__} executed in {execution_time:.3f}s [req:{request_id}]")
            
            # Store metrics in request context
            if not hasattr(g, 'performance_metrics'):
                g.performance_metrics = {}
            g.performance_metrics[func.__name__] = execution_time
            
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s [req:{request_id}]: {e}")
            raise
    return wrapper

def error_handler(func: Callable) -> Callable:
    """Decorator for comprehensive error handling with early returns."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.error(f"❌ Validation error in {func.__name__}: {e}")
            return _create_error_response('validation_error', str(e), func.__name__)
        except PermissionError as e:
            logger.error(f"❌ Permission error in {func.__name__}: {e}")
            return _create_error_response('permission_denied', str(e), func.__name__)
        except FileNotFoundError as e:
            logger.error(f"❌ File not found error in {func.__name__}: {e}")
            return _create_error_response('file_not_found', str(e), func.__name__)
        except Exception as e:
            logger.error(f"❌ Unexpected error in {func.__name__}: {e}")
            return _create_error_response('internal_server_error', str(e), func.__name__)
    return wrapper

def validate_request(schema_class):
    """Decorator for request validation with early returns."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Early return for non-JSON requests
            if not request.is_json:
                return _create_error_response('invalid_content_type', 'Request must be JSON', func.__name__)
            
            try:
                data = request.get_json()
                if not data:
                    return _create_error_response('empty_request', 'Request body is empty', func.__name__)
                
                schema = schema_class()
                validated_data = schema.load(data)
                kwargs['validated_data'] = validated_data
                
                return func(*args, **kwargs)
            except ValidationError as e:
                logger.error(f"❌ Request validation error in {func.__name__}: {e}")
                return _create_error_response('validation_error', str(e), func.__name__)
        return wrapper
    return decorator

def cache_result(ttl: int = 300, key_prefix: str = None):
    """Decorator for intelligent caching with cache invalidation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key with prefix
            prefix = key_prefix or func.__name__
            cache_key = f"{prefix}:{hash(str(args) + str(kwargs))}"
            
            # Early return if cache is not available
            if not hasattr(current_app, 'cache'):
                return func(*args, **kwargs)
            
            # Try to get from cache
            cached_result = current_app.cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"📦 Cache hit for {func.__name__} [key:{cache_key}]")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            current_app.cache.set(cache_key, result, timeout=ttl)
            logger.info(f"💾 Cached result for {func.__name__} [key:{cache_key}]")
            
            return result
        return wrapper
    return decorator

def rate_limit(requests_per_minute: int = 60, key_func: Callable = None):
    """Decorator for intelligent rate limiting with custom key functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate rate limit key
            if key_func:
                rate_key = key_func()
            else:
                client_ip = request.remote_addr
                rate_key = f"rate_limit:{client_ip}:{func.__name__}"
            
            # Early return if cache is not available
            if not hasattr(current_app, 'cache'):
                return func(*args, **kwargs)
            
            # Check rate limit
            current_requests = current_app.cache.get(rate_key) or 0
            if current_requests >= requests_per_minute:
                logger.warning(f"🚫 Rate limit exceeded for {rate_key}")
                return _create_error_response('rate_limit_exceeded', 
                    f'Maximum {requests_per_minute} requests per minute allowed', func.__name__)
            
            # Increment counter and execute function
            current_app.cache.set(rate_key, current_requests + 1, timeout=60)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def async_performance_monitor(func: Callable) -> Callable:
    """Decorator for async performance monitoring."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        request_id = getattr(g, 'request_id', 'unknown')
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.info(f"⚡ {func.__name__} executed in {execution_time:.3f}s [req:{request_id}]")
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s [req:{request_id}]: {e}")
            raise
    return wrapper

def async_error_handler(func: Callable) -> Callable:
    """Decorator for async error handling with early returns."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            logger.error(f"❌ Validation error in {func.__name__}: {e}")
            return _create_error_response('validation_error', str(e), func.__name__)
        except Exception as e:
            logger.error(f"❌ Error in {func.__name__}: {e}")
            return _create_error_response('internal_server_error', str(e), func.__name__)
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator for retrying on failure with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} for {func.__name__} in {wait_time:.2f}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ All retries failed for {func.__name__}: {e}")
                        raise last_exception
            
            return None
        return wrapper
    return decorator

def async_retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator for async retrying on failure with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} for {func.__name__} in {wait_time:.2f}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ All retries failed for {func.__name__}: {e}")
                        raise last_exception
            
            return None
        return wrapper
    return decorator

def require_auth(func: Callable) -> Callable:
    """Decorator for JWT authentication with early returns."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Early return if JWT is not available
        if not hasattr(current_app, 'jwt'):
            return _create_error_response('auth_not_configured', 'JWT not configured', func.__name__)
        
        try:
            from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
            verify_jwt_in_request()
            user_id = get_jwt_identity()
            
            # Store user info in request context
            g.current_user_id = user_id
            
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"❌ Authentication error in {func.__name__}: {e}")
            return _create_error_response('authentication_required', 'Valid JWT token required', func.__name__)
    return wrapper

def require_permissions(*required_permissions: str):
    """Decorator for permission-based authorization with early returns."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Early return if user is not authenticated
            if not hasattr(g, 'current_user_id'):
                return _create_error_response('authentication_required', 'User must be authenticated', func.__name__)
            
            # Check permissions (mock implementation)
            user_permissions = _get_user_permissions(g.current_user_id)
            missing_permissions = [perm for perm in required_permissions if perm not in user_permissions]
            
            if missing_permissions:
                logger.warning(f"❌ Insufficient permissions for {func.__name__}: missing {missing_permissions}")
                return _create_error_response('insufficient_permissions', 
                    f'Missing permissions: {", ".join(missing_permissions)}', func.__name__)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_content_type(*allowed_types: str):
    """Decorator for content type validation with early returns."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            content_type = request.content_type
            if content_type not in allowed_types:
                return _create_error_response('invalid_content_type', 
                    f'Content-Type must be one of: {", ".join(allowed_types)}', func.__name__)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_request(func: Callable) -> Callable:
    """Decorator for comprehensive request logging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        request_id = getattr(g, 'request_id', 'unknown')
        user_id = getattr(g, 'current_user_id', 'anonymous')
        
        logger.info(f"📥 Request started: {request.method} {request.path} [req:{request_id}] [user:{user_id}]")
        
        # Log request details
        if request.is_json:
            logger.debug(f"📄 Request body: {request.get_json()}")
        
        result = func(*args, **kwargs)
        
        logger.info(f"📤 Request completed: {request.method} {request.path} [req:{request_id}] [user:{user_id}]")
        return result
    return wrapper

def timeout(seconds: float):
    """Decorator for function timeout with early returns."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore old handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

def circuit_breaker(failure_threshold: int = 5, timeout: float = 60.0):
    """Decorator for circuit breaker pattern with early returns."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            circuit_key = f"circuit_breaker:{func.__name__}"
            
            # Check circuit state
            if hasattr(current_app, 'cache'):
                circuit_state = current_app.cache.get(circuit_key)
                if circuit_state and circuit_state.get('state') == 'open':
                    if time.time() - circuit_state.get('last_failure', 0) < timeout:
                        return _create_error_response('circuit_breaker_open', 
                            'Service temporarily unavailable', func.__name__)
                    else:
                        # Reset circuit
                        current_app.cache.delete(circuit_key)
            
            try:
                result = func(*args, **kwargs)
                
                # Reset failure count on success
                if hasattr(current_app, 'cache'):
                    current_app.cache.delete(circuit_key)
                
                return result
            except Exception as e:
                # Increment failure count
                if hasattr(current_app, 'cache'):
                    failure_count = current_app.cache.get(f"{circuit_key}:failures") or 0
                    failure_count += 1
                    current_app.cache.set(f"{circuit_key}:failures", failure_count, timeout=timeout)
                    
                    if failure_count >= failure_threshold:
                        current_app.cache.set(circuit_key, {
                            'state': 'open',
                            'last_failure': time.time()
                        }, timeout=timeout)
                
                raise e
        return wrapper
    return decorator

def _create_error_response(error_type: str, message: str, function_name: str) -> Dict[str, Any]:
    """Create standardized error response with early returns."""
    return {
        'success': False,
        'error_type': error_type,
        'message': message,
        'function': function_name,
        'timestamp': time.time(),
        'request_id': getattr(g, 'request_id', 'unknown')
    }

def _get_user_permissions(user_id: str) -> list:
    """Get user permissions (mock implementation)."""
    # Mock implementation - in real app, this would query database
    return ['read', 'write', 'admin'] if user_id else []

# Utility functions following RORO pattern
def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Create standardized success response."""
    return {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': time.time(),
        'request_id': getattr(g, 'request_id', 'unknown')
    }

def create_paginated_response(data: list, page: int, per_page: int, total: int) -> Dict[str, Any]:
    """Create paginated response."""
    return {
        'success': True,
        'data': data,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        },
        'timestamp': time.time(),
        'request_id': getattr(g, 'request_id', 'unknown')
    }