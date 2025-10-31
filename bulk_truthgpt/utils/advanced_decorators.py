"""
Advanced Decorators
==================

Ultra-advanced decorators following Flask best practices.
"""

import time
import functools
import logging
import asyncio
import threading
from typing import Callable, Any, Optional, Dict, List, Union, TypeVar, Generic
from flask import request, jsonify, g, current_app, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import TooManyRequests
import redis
import json
import hashlib
import uuid
from datetime import datetime, timedelta
import psutil
import os

logger = logging.getLogger(__name__)
T = TypeVar('T')

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

def ultra_rate_limit(limit: str = "100 per hour", key_func: Optional[Callable] = None, 
                    burst_limit: Optional[str] = None, skip_successful: bool = False):
    """
    Ultra-advanced rate limiting decorator.
    
    Args:
        limit: Rate limit string (e.g., "100 per hour")
        key_func: Custom key function for rate limiting
        burst_limit: Burst limit for handling traffic spikes
        skip_successful: Skip rate limiting for successful requests
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Apply burst limit if specified
                if burst_limit:
                    limiter.limit(burst_limit, key_func=key_func)(f)(*args, **kwargs)
                
                # Apply main rate limit
                if key_func:
                    limiter.limit(limit, key_func=key_func)(f)(*args, **kwargs)
                else:
                    limiter.limit(limit)(f)(*args, **kwargs)
                
                # Execute function
                response = f(*args, **kwargs)
                
                # Skip rate limiting for successful responses if enabled
                if skip_successful and hasattr(response, 'status_code') and response.status_code < 400:
                    return response
                
                return response
            except TooManyRequests:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Limit: {limit}',
                    'retry_after': 60
                }), 429
            except Exception as e:
                current_app.logger.error(f"Rate limiting error: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        return decorated_function
    return decorator

def ultra_json_validation(schema_class: Optional[type] = None, 
                         strict: bool = True, 
                         unknown: str = 'raise',
                         partial: bool = False):
    """
    Ultra-advanced JSON validation decorator.
    
    Args:
        schema_class: Marshmallow schema class for validation
        strict: Whether to use strict validation
        unknown: How to handle unknown fields
        partial: Whether to allow partial validation
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Check if request has JSON data
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400
                
                # Validate JSON data if schema provided
                if schema_class:
                    schema = schema_class()
                    try:
                        validated_data = schema.load(
                            request.json, 
                            strict=strict, 
                            unknown=unknown, 
                            partial=partial
                        )
                        # Add validated data to request context
                        g.validated_data = validated_data
                    except Exception as e:
                        return jsonify({
                            'error': 'Validation error',
                            'details': str(e),
                            'field_errors': getattr(e, 'messages', {})
                        }), 400
                
                return f(*args, **kwargs)
            except Exception as e:
                current_app.logger.error(f"JSON validation error: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        return decorated_function
    return decorator

def ultra_response_caching(ttl: int = 300, key_prefix: str = "response",
                          vary_headers: Optional[List[str]] = None,
                          cache_control: Optional[str] = None,
                          etag: bool = True):
    """
    Ultra-advanced response caching decorator.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Cache key prefix
        vary_headers: Headers to vary cache by
        cache_control: Cache control header
        etag: Whether to generate ETags
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Generate cache key with headers
                cache_key = _generate_advanced_cache_key(
                    f.__name__, request, key_prefix, vary_headers
                )
                
                # Try to get from cache
                cached_response = _get_from_advanced_cache(cache_key)
                if cached_response:
                    # Add cache headers
                    response = jsonify(cached_response)
                    if cache_control:
                        response.headers['Cache-Control'] = cache_control
                    if etag:
                        response.headers['ETag'] = _generate_etag(cached_response)
                    return response
                
                # Execute function
                response = f(*args, **kwargs)
                
                # Cache response if successful
                if hasattr(response, 'status_code') and response.status_code < 400:
                    _set_advanced_cache(cache_key, response, ttl)
                
                return response
            except Exception as e:
                current_app.logger.error(f"Cache error: {str(e)}")
                # Fallback to original function
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def ultra_performance_monitoring(metric_name: Optional[str] = None,
                                track_memory: bool = True,
                                track_cpu: bool = True,
                                track_database: bool = True,
                                custom_metrics: Optional[Dict[str, Callable]] = None):
    """
    Ultra-advanced performance monitoring decorator.
    
    Args:
        metric_name: Custom metric name
        track_memory: Whether to track memory usage
        track_cpu: Whether to track CPU usage
        track_database: Whether to track database queries
        custom_metrics: Custom metrics to track
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss if track_memory else 0
            start_cpu = psutil.cpu_percent() if track_cpu else 0
            
            metric_name_final = metric_name or f"{f.__name__}_execution_time"
            
            try:
                # Execute function
                response = f(*args, **kwargs)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss if track_memory else 0
                end_cpu = psutil.cpu_percent() if track_cpu else 0
                
                # Log performance metrics
                _log_ultra_performance_metrics(
                    metric_name_final, execution_time, 
                    end_memory - start_memory, end_cpu - start_cpu,
                    custom_metrics
                )
                
                return response
            except Exception as e:
                # Log error metric
                execution_time = time.time() - start_time
                _log_ultra_performance_metrics(
                    f"{metric_name_final}_error", execution_time, 0, 0, custom_metrics
                )
                raise e
        
        return decorated_function
    return decorator

def ultra_authentication(roles: Optional[List[str]] = None,
                        permissions: Optional[List[str]] = None,
                        require_fresh_token: bool = False,
                        allow_anonymous: bool = False):
    """
    Ultra-advanced authentication decorator.
    
    Args:
        roles: Required user roles
        permissions: Required user permissions
        require_fresh_token: Whether to require fresh token
        allow_anonymous: Whether to allow anonymous access
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Check if user is authenticated
                if not hasattr(g, 'current_user') or not g.current_user:
                    if allow_anonymous:
                        return f(*args, **kwargs)
                    return jsonify({'error': 'Authentication required'}), 401
                
                # Check roles if specified
                if roles and g.current_user.role not in roles:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                # Check permissions if specified
                if permissions:
                    user_permissions = getattr(g.current_user, 'permissions', [])
                    if not all(perm in user_permissions for perm in permissions):
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                # Check fresh token if required
                if require_fresh_token and not getattr(g, 'fresh_token', False):
                    return jsonify({'error': 'Fresh token required'}), 401
                
                return f(*args, **kwargs)
            except Exception as e:
                current_app.logger.error(f"Authentication error: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        return decorated_function
    return decorator

def ultra_error_handling(error_handler: Optional[Callable] = None,
                        log_errors: bool = True,
                        return_errors: bool = True,
                        error_context: bool = True):
    """
    Ultra-advanced error handling decorator.
    
    Args:
        error_handler: Custom error handler function
        log_errors: Whether to log errors
        return_errors: Whether to return error responses
        error_context: Whether to include error context
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    current_app.logger.error(f"Error in {f.__name__}: {str(e)}")
                
                if error_handler:
                    return error_handler(e)
                
                if return_errors:
                    error_response = {
                        'error': 'Internal server error',
                        'message': str(e) if current_app.debug else 'An unexpected error occurred'
                    }
                    
                    if error_context:
                        error_response['context'] = {
                            'function': f.__name__,
                            'timestamp': datetime.utcnow().isoformat(),
                            'request_id': getattr(g, 'request_id', None)
                        }
                    
                    return jsonify(error_response), 500
                
                raise e
        
        return decorated_function
    return decorator

def ultra_activity_logging(activity: str,
                          log_level: str = 'info',
                          include_request_data: bool = True,
                          include_response_data: bool = False,
                          sensitive_fields: Optional[List[str]] = None):
    """
    Ultra-advanced activity logging decorator.
    
    Args:
        activity: Activity description
        log_level: Log level (info, warning, error)
        include_request_data: Whether to include request data
        include_response_data: Whether to include response data
        sensitive_fields: Fields to exclude from logging
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Log activity start
                log_data = {
                    'activity': activity,
                    'function': f.__name__,
                    'timestamp': datetime.utcnow().isoformat(),
                    'request_id': getattr(g, 'request_id', None)
                }
                
                if include_request_data:
                    request_data = _sanitize_request_data(request, sensitive_fields)
                    log_data['request'] = request_data
                
                _log_activity(log_level, f"Activity started: {activity}", log_data)
                
                # Execute function
                response = f(*args, **kwargs)
                
                # Log activity completion
                completion_data = log_data.copy()
                if include_response_data:
                    response_data = _sanitize_response_data(response, sensitive_fields)
                    completion_data['response'] = response_data
                
                _log_activity(log_level, f"Activity completed: {activity}", completion_data)
                
                return response
            except Exception as e:
                # Log activity error
                error_data = {
                    'activity': activity,
                    'function': f.__name__,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat(),
                    'request_id': getattr(g, 'request_id', None)
                }
                _log_activity('error', f"Activity failed: {activity}", error_data)
                raise e
        
        return decorated_function
    return decorator

def ultra_input_validation(schema_class: type,
                          strict: bool = True,
                          unknown: str = 'raise',
                          partial: bool = False,
                          custom_validators: Optional[Dict[str, Callable]] = None):
    """
    Ultra-advanced input validation decorator.
    
    Args:
        schema_class: Marshmallow schema class
        strict: Whether to use strict validation
        unknown: How to handle unknown fields
        partial: Whether to allow partial validation
        custom_validators: Custom validation functions
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Validate input data
                schema = schema_class()
                validated_data = schema.load(
                    request.json or {}, 
                    strict=strict, 
                    unknown=unknown, 
                    partial=partial
                )
                
                # Apply custom validators
                if custom_validators:
                    for field, validator in custom_validators.items():
                        if field in validated_data:
                            validator(validated_data[field])
                
                # Add validated data to kwargs
                kwargs['validated_data'] = validated_data
                
                return f(*args, **kwargs)
            except Exception as e:
                return jsonify({
                    'error': 'Validation error',
                    'details': str(e),
                    'field_errors': getattr(e, 'messages', {})
                }), 400
        
        return decorated_function
    return decorator

def ultra_async_operation(timeout: float = 30.0,
                         max_concurrent: int = 10,
                         retry_attempts: int = 3,
                         retry_delay: float = 1.0):
    """
    Ultra-advanced async operation decorator.
    
    Args:
        timeout: Operation timeout in seconds
        max_concurrent: Maximum concurrent operations
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Execute function with timeout
                result = asyncio.run(
                    asyncio.wait_for(
                        f(*args, **kwargs),
                        timeout=timeout
                    )
                )
                return result
            except asyncio.TimeoutError:
                return jsonify({'error': 'Operation timeout'}), 408
            except Exception as e:
                # Retry logic
                for attempt in range(retry_attempts):
                    try:
                        time.sleep(retry_delay * (attempt + 1))
                        result = asyncio.run(
                            asyncio.wait_for(
                                f(*args, **kwargs),
                                timeout=timeout
                            )
                        )
                        return result
                    except Exception:
                        if attempt == retry_attempts - 1:
                            raise e
                
                return jsonify({'error': 'Operation failed after retries'}), 500
        
        return decorated_function
    return decorator

def ultra_circuit_breaker(failure_threshold: int = 5,
                         recovery_timeout: float = 60.0,
                         expected_exception: type = Exception):
    """
    Ultra-advanced circuit breaker decorator.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type to count as failure
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        circuit_state = {'failures': 0, 'last_failure': None, 'state': 'closed'}
        
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            current_time = time.time()
            
            # Check if circuit is open
            if circuit_state['state'] == 'open':
                if current_time - circuit_state['last_failure'] > recovery_timeout:
                    circuit_state['state'] = 'half-open'
                else:
                    return jsonify({'error': 'Circuit breaker is open'}), 503
            
            try:
                # Execute function
                result = f(*args, **kwargs)
                
                # Reset circuit on success
                if circuit_state['state'] == 'half-open':
                    circuit_state['state'] = 'closed'
                    circuit_state['failures'] = 0
                
                return result
            except expected_exception as e:
                circuit_state['failures'] += 1
                circuit_state['last_failure'] = current_time
                
                if circuit_state['failures'] >= failure_threshold:
                    circuit_state['state'] = 'open'
                
                raise e
        
        return decorated_function
    return decorator

def ultra_retry_logic(max_retries: int = 3,
                     base_delay: float = 1.0,
                     max_delay: float = 60.0,
                     exponential_backoff: bool = True,
                     jitter: bool = True):
    """
    Ultra-advanced retry logic decorator.
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential_backoff: Whether to use exponential backoff
        jitter: Whether to add jitter to delays
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    # Calculate delay
                    if exponential_backoff:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    if jitter:
                        delay *= (0.5 + 0.5 * (hash(str(time.time())) % 100) / 100)
                    
                    time.sleep(delay)
            
            return None
        
        return decorated_function
    return decorator

# Helper functions
def _generate_advanced_cache_key(function_name: str, request, key_prefix: str, 
                                vary_headers: Optional[List[str]]) -> str:
    """Generate advanced cache key."""
    components = [
        key_prefix,
        function_name,
        request.method,
        request.path,
        str(sorted(request.args.items())),
        str(request.json) if request.is_json else ""
    ]
    
    if vary_headers:
        for header in vary_headers:
            components.append(f"{header}:{request.headers.get(header, '')}")
    
    key_string = "|".join(components)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    
    return f"cache:{key_hash}"

def _get_from_advanced_cache(cache_key: str) -> Optional[Any]:
    """Get value from advanced cache."""
    try:
        if hasattr(current_app, 'redis'):
            cached_data = current_app.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
    except Exception as e:
        current_app.logger.error(f"Cache get error: {str(e)}")
    
    return None

def _set_advanced_cache(cache_key: str, value: Any, ttl: int) -> None:
    """Set value in advanced cache."""
    try:
        if hasattr(current_app, 'redis'):
            current_app.redis.setex(cache_key, ttl, json.dumps(value))
    except Exception as e:
        current_app.logger.error(f"Cache set error: {str(e)}")

def _generate_etag(data: Any) -> str:
    """Generate ETag for data."""
    data_string = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_string.encode()).hexdigest()

def _log_ultra_performance_metrics(metric_name: str, execution_time: float, 
                                  memory_delta: int, cpu_delta: float,
                                  custom_metrics: Optional[Dict[str, Callable]]) -> None:
    """Log ultra performance metrics."""
    try:
        metrics = {
            'metric_name': metric_name,
            'execution_time': execution_time,
            'memory_delta': memory_delta,
            'cpu_delta': cpu_delta,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if custom_metrics:
            for name, func in custom_metrics.items():
                try:
                    metrics[name] = func()
                except Exception as e:
                    current_app.logger.error(f"Custom metric error: {str(e)}")
        
        current_app.logger.info(f"Performance metrics: {json.dumps(metrics)}")
    except Exception as e:
        current_app.logger.error(f"Performance logging error: {str(e)}")

def _log_activity(level: str, message: str, data: Dict[str, Any]) -> None:
    """Log activity with structured data."""
    try:
        log_data = {
            'message': message,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if level == 'info':
            current_app.logger.info(json.dumps(log_data))
        elif level == 'warning':
            current_app.logger.warning(json.dumps(log_data))
        elif level == 'error':
            current_app.logger.error(json.dumps(log_data))
    except Exception as e:
        current_app.logger.error(f"Activity logging error: {str(e)}")

def _sanitize_request_data(request, sensitive_fields: Optional[List[str]]) -> Dict[str, Any]:
    """Sanitize request data for logging."""
    try:
        data = {
            'method': request.method,
            'path': request.path,
            'headers': dict(request.headers),
            'args': dict(request.args)
        }
        
        if request.is_json:
            json_data = request.json.copy()
            if sensitive_fields:
                for field in sensitive_fields:
                    if field in json_data:
                        json_data[field] = '[REDACTED]'
            data['json'] = json_data
        
        return data
    except Exception as e:
        current_app.logger.error(f"Request data sanitization error: {str(e)}")
        return {}

def _sanitize_response_data(response, sensitive_fields: Optional[List[str]]) -> Dict[str, Any]:
    """Sanitize response data for logging."""
    try:
        data = {
            'status_code': getattr(response, 'status_code', None),
            'headers': dict(getattr(response, 'headers', {}))
        }
        
        if hasattr(response, 'json'):
            json_data = response.json.copy()
            if sensitive_fields:
                for field in sensitive_fields:
                    if field in json_data:
                        json_data[field] = '[REDACTED]'
            data['json'] = json_data
        
        return data
    except Exception as e:
        current_app.logger.error(f"Response data sanitization error: {str(e)}")
        return {}









