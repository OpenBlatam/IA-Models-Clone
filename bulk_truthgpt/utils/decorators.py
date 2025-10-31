"""
Flask Decorators
===============

Ultra-modular decorators following Flask best practices.
"""

import time
import functools
import logging
from typing import Callable, Any, Optional, Dict, List, Union, TypeVar
from flask import request, jsonify, g, current_app, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import TooManyRequests
import redis
import json
import hashlib

logger = logging.getLogger(__name__)
T = TypeVar('T')

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

def rate_limit(limit: str = "100 per hour", key_func: Optional[Callable] = None):
    """
    Rate limiting decorator.
    
    Args:
        limit: Rate limit string (e.g., "100 per hour")
        key_func: Custom key function for rate limiting
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Apply rate limiting
                if key_func:
                    limiter.limit(limit, key_func=key_func)(f)(*args, **kwargs)
                else:
                    limiter.limit(limit)(f)(*args, **kwargs)
                
                return f(*args, **kwargs)
            except TooManyRequests:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Limit: {limit}'
                }), 429
            except Exception as e:
                current_app.logger.error(f"Rate limiting error: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        return decorated_function
    return decorator

def validate_json(schema_class: Optional[type] = None):
    """
    JSON validation decorator.
    
    Args:
        schema_class: Marshmallow schema class for validation
    
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
                        validated_data = schema.load(request.json)
                        # Add validated data to request context
                        g.validated_data = validated_data
                    except Exception as e:
                        return jsonify({
                            'error': 'Validation error',
                            'details': str(e)
                        }), 400
                
                return f(*args, **kwargs)
            except Exception as e:
                current_app.logger.error(f"JSON validation error: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        return decorated_function
    return decorator

def cache_response(ttl: int = 300, key_prefix: str = "response"):
    """
    Response caching decorator.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Cache key prefix
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Generate cache key
                cache_key = _generate_cache_key(f.__name__, request, key_prefix)
                
                # Try to get from cache
                cached_response = _get_from_cache(cache_key)
                if cached_response:
                    return cached_response
                
                # Execute function
                response = f(*args, **kwargs)
                
                # Cache response
                _set_cache(cache_key, response, ttl)
                
                return response
            except Exception as e:
                current_app.logger.error(f"Cache error: {str(e)}")
                # Fallback to original function
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def monitor_performance(metric_name: Optional[str] = None):
    """
    Performance monitoring decorator.
    
    Args:
        metric_name: Custom metric name
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            metric_name_final = metric_name or f"{f.__name__}_execution_time"
            
            try:
                # Execute function
                response = f(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log performance metric
                _log_performance_metric(metric_name_final, execution_time)
                
                return response
            except Exception as e:
                # Log error metric
                execution_time = time.time() - start_time
                _log_performance_metric(f"{metric_name_final}_error", execution_time)
                raise e
        
        return decorated_function
    return decorator

def require_auth(roles: Optional[List[str]] = None):
    """
    Authentication requirement decorator.
    
    Args:
        roles: Required user roles
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Check if user is authenticated
                if not hasattr(g, 'current_user') or not g.current_user:
                    return jsonify({'error': 'Authentication required'}), 401
                
                # Check roles if specified
                if roles and g.current_user.role not in roles:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                return f(*args, **kwargs)
            except Exception as e:
                current_app.logger.error(f"Authentication error: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        return decorated_function
    return decorator

def handle_errors(error_handler: Optional[Callable] = None):
    """
    Error handling decorator.
    
    Args:
        error_handler: Custom error handler function
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                current_app.logger.error(f"Error in {f.__name__}: {str(e)}")
                
                if error_handler:
                    return error_handler(e)
                
                return jsonify({
                    'error': 'Internal server error',
                    'message': str(e) if current_app.debug else 'An unexpected error occurred'
                }), 500
        
        return decorated_function
    return decorator

def log_activity(activity: str):
    """
    Activity logging decorator.
    
    Args:
        activity: Activity description
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Log activity start
                current_app.logger.info(f"Activity started: {activity}")
                
                # Execute function
                response = f(*args, **kwargs)
                
                # Log activity completion
                current_app.logger.info(f"Activity completed: {activity}")
                
                return response
            except Exception as e:
                # Log activity error
                current_app.logger.error(f"Activity failed: {activity} - {str(e)}")
                raise e
        
        return decorated_function
    return decorator

def validate_input(schema_class: type):
    """
    Input validation decorator.
    
    Args:
        schema_class: Marshmallow schema class
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Validate input data
                schema = schema_class()
                validated_data = schema.load(request.json or {})
                
                # Add validated data to kwargs
                kwargs['validated_data'] = validated_data
                
                return f(*args, **kwargs)
            except Exception as e:
                return jsonify({
                    'error': 'Validation error',
                    'details': str(e)
                }), 400
        
        return decorated_function
    return decorator

def _generate_cache_key(function_name: str, request, key_prefix: str) -> str:
    """Generate cache key."""
    # Create key components
    components = [
        key_prefix,
        function_name,
        request.method,
        request.path,
        str(sorted(request.args.items())),
        str(request.json) if request.is_json else ""
    ]
    
    # Generate hash
    key_string = "|".join(components)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    
    return f"cache:{key_hash}"

def _get_from_cache(cache_key: str) -> Optional[Any]:
    """Get value from cache."""
    try:
        if hasattr(current_app, 'redis'):
            cached_data = current_app.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
    except Exception as e:
        current_app.logger.error(f"Cache get error: {str(e)}")
    
    return None

def _set_cache(cache_key: str, value: Any, ttl: int) -> None:
    """Set value in cache."""
    try:
        if hasattr(current_app, 'redis'):
            current_app.redis.setex(cache_key, ttl, json.dumps(value))
    except Exception as e:
        current_app.logger.error(f"Cache set error: {str(e)}")

def _log_performance_metric(metric_name: str, value: float) -> None:
    """Log performance metric."""
    try:
        # This would integrate with monitoring system
        current_app.logger.info(f"Performance metric: {metric_name} = {value}")
    except Exception as e:
        current_app.logger.error(f"Performance logging error: {str(e)}")