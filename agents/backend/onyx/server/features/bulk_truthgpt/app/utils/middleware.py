"""
Advanced middleware for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from flask import request, response, current_app, g, jsonify
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)

def init_middleware(app) -> None:
    """Initialize middleware with app."""
    # Register middleware functions
    app.before_request(before_request_middleware)
    app.after_request(after_request_middleware)
    app.teardown_request(teardown_request_middleware)
    
    # Register error handlers
    app.errorhandler(400)(handle_bad_request)
    app.errorhandler(401)(handle_unauthorized)
    app.errorhandler(403)(handle_forbidden)
    app.errorhandler(404)(handle_not_found)
    app.errorhandler(405)(handle_method_not_allowed)
    app.errorhandler(429)(handle_rate_limit_exceeded)
    app.errorhandler(500)(handle_internal_server_error)
    
    logger.info("🔧 Middleware initialized")

def before_request_middleware() -> Optional[response.Response]:
    """Before request middleware with early returns."""
    # Set request start time
    g.start_time = time.perf_counter()
    g.request_id = f"req_{int(time.time() * 1000)}"
    
    # Log request
    logger.info(f"📥 Request started: {request.method} {request.path} [req:{g.request_id}]")
    
    # Check request size
    if request.content_length and request.content_length > current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024):
        return jsonify({'success': False, 'message': 'Request too large'}), 413
    
    # Check rate limit
    if not check_rate_limit_middleware():
        return jsonify({'success': False, 'message': 'Rate limit exceeded'}), 429
    
    # Validate content type for POST/PUT requests
    if request.method in ['POST', 'PUT', 'PATCH']:
        if not request.is_json and request.content_type != 'application/json':
            return jsonify({'success': False, 'message': 'Content-Type must be application/json'}), 400
    
    # Add security headers
    add_security_headers()
    
    return None

def after_request_middleware(response: response.Response) -> response.Response:
    """After request middleware with early returns."""
    # Calculate processing time
    if hasattr(g, 'start_time'):
        processing_time = time.perf_counter() - g.start_time
        response.headers['X-Processing-Time'] = str(processing_time)
        
        # Log request completion
        logger.info(f"📤 Request completed: {request.method} {request.path} [req:{g.request_id}] in {processing_time:.3f}s")
    
    # Add CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-API-Key'
    response.headers['Access-Control-Max-Age'] = '3600'
    
    # Add security headers
    add_security_headers(response)
    
    # Add request ID to response
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    
    return response

def teardown_request_middleware(exception: Optional[Exception]) -> None:
    """Teardown request middleware with early returns."""
    if exception:
        logger.error(f"❌ Request failed: {request.method} {request.path} [req:{getattr(g, 'request_id', 'unknown')}] - {exception}")
    else:
        logger.debug(f"✅ Request teardown: {request.method} {request.path} [req:{getattr(g, 'request_id', 'unknown')}]")
    
    # Clean up request context
    if hasattr(g, 'start_time'):
        delattr(g, 'start_time')
    if hasattr(g, 'request_id'):
        delattr(g, 'request_id')
    if hasattr(g, 'current_user'):
        delattr(g, 'current_user')

def check_rate_limit_middleware() -> bool:
    """Check rate limit with early returns."""
    client_ip = request.remote_addr
    endpoint = request.endpoint or 'unknown'
    
    # Mock rate limit check
    # In production, you'd use Redis or similar
    return True

def add_security_headers(response: response.Response = None) -> None:
    """Add security headers with early returns."""
    headers = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }
    
    if response:
        for header, value in headers.items():
            response.headers[header] = value
    else:
        # Add to request context for later use
        g.security_headers = headers

# Error handlers
def handle_bad_request(error: HTTPException) -> response.Response:
    """Handle 400 Bad Request with early returns."""
    logger.warning(f"❌ Bad request: {error}")
    return jsonify({
        'success': False,
        'message': 'Bad request',
        'error': str(error),
        'timestamp': time.time()
    }), 400

def handle_unauthorized(error: HTTPException) -> response.Response:
    """Handle 401 Unauthorized with early returns."""
    logger.warning(f"❌ Unauthorized: {error}")
    return jsonify({
        'success': False,
        'message': 'Unauthorized',
        'error': str(error),
        'timestamp': time.time()
    }), 401

def handle_forbidden(error: HTTPException) -> response.Response:
    """Handle 403 Forbidden with early returns."""
    logger.warning(f"❌ Forbidden: {error}")
    return jsonify({
        'success': False,
        'message': 'Forbidden',
        'error': str(error),
        'timestamp': time.time()
    }), 403

def handle_not_found(error: HTTPException) -> response.Response:
    """Handle 404 Not Found with early returns."""
    logger.warning(f"❌ Not found: {error}")
    return jsonify({
        'success': False,
        'message': 'Not found',
        'error': str(error),
        'timestamp': time.time()
    }), 404

def handle_method_not_allowed(error: HTTPException) -> response.Response:
    """Handle 405 Method Not Allowed with early returns."""
    logger.warning(f"❌ Method not allowed: {error}")
    return jsonify({
        'success': False,
        'message': 'Method not allowed',
        'error': str(error),
        'timestamp': time.time()
    }), 405

def handle_rate_limit_exceeded(error: HTTPException) -> response.Response:
    """Handle 429 Rate Limit Exceeded with early returns."""
    logger.warning(f"❌ Rate limit exceeded: {error}")
    return jsonify({
        'success': False,
        'message': 'Rate limit exceeded',
        'error': str(error),
        'timestamp': time.time()
    }), 429

def handle_internal_server_error(error: HTTPException) -> response.Response:
    """Handle 500 Internal Server Error with early returns."""
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({
        'success': False,
        'message': 'Internal server error',
        'error': str(error),
        'timestamp': time.time()
    }), 500

# Middleware decorators
def middleware_decorator(middleware_func: Callable) -> Callable:
    """Decorator for applying middleware with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply middleware before function
            middleware_result = middleware_func()
            if middleware_result:
                return middleware_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Apply middleware after function
            return result
        return wrapper
    return decorator

def authentication_middleware(func: Callable) -> Callable:
    """Authentication middleware decorator with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'success': False, 'message': 'Authentication required'}), 401
        
        # Mock token validation
        token = auth_header.split(' ')[1]
        if not token:
            return jsonify({'success': False, 'message': 'Invalid token'}), 401
        
        # Store user info in context
        g.current_user = {'id': 'user_123', 'permissions': ['read', 'write']}
        
        return func(*args, **kwargs)
    return wrapper

def permission_middleware(*required_permissions: str):
    """Permission middleware decorator with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if user is authenticated
            if not hasattr(g, 'current_user'):
                return jsonify({'success': False, 'message': 'Authentication required'}), 401
            
            # Check permissions
            user_permissions = g.current_user.get('permissions', [])
            missing_permissions = [perm for perm in required_permissions if perm not in user_permissions]
            
            if missing_permissions:
                return jsonify({'success': False, 'message': f'Missing permissions: {", ".join(missing_permissions)}'}), 403
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit_middleware(max_requests: int = 100, window: int = 3600):
    """Rate limit middleware decorator with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr
            endpoint = request.endpoint or func.__name__
            
            # Mock rate limit check
            if not check_rate_limit_middleware():
                return jsonify({'success': False, 'message': 'Rate limit exceeded'}), 429
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def logging_middleware(func: Callable) -> Callable:
    """Logging middleware decorator with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log request start
        logger.info(f"📥 {func.__name__} started")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"📤 {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ {func.__name__} failed: {e}")
            raise
    return wrapper

def performance_middleware(func: Callable) -> Callable:
    """Performance middleware decorator with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Log performance metrics
            logger.info(f"⚡ {func.__name__} executed in {execution_time:.3f}s")
            
            # Store metrics in context
            if not hasattr(g, 'performance_metrics'):
                g.performance_metrics = {}
            g.performance_metrics[func.__name__] = execution_time
            
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper

def error_handling_middleware(func: Callable) -> Callable:
    """Error handling middleware decorator with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.error(f"❌ ValueError in {func.__name__}: {e}")
            return jsonify({'success': False, 'message': 'Invalid value', 'error': str(e)}), 400
        except PermissionError as e:
            logger.error(f"❌ PermissionError in {func.__name__}: {e}")
            return jsonify({'success': False, 'message': 'Permission denied', 'error': str(e)}), 403
        except FileNotFoundError as e:
            logger.error(f"❌ FileNotFoundError in {func.__name__}: {e}")
            return jsonify({'success': False, 'message': 'File not found', 'error': str(e)}), 404
        except Exception as e:
            logger.error(f"❌ Unexpected error in {func.__name__}: {e}")
            return jsonify({'success': False, 'message': 'Internal server error', 'error': str(e)}), 500
    return wrapper

def validation_middleware(schema_class):
    """Validation middleware decorator with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return jsonify({'success': False, 'message': 'Request must be JSON'}), 400
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'success': False, 'message': 'Request body is empty'}), 400
                
                schema = schema_class()
                validated_data = schema.load(data)
                kwargs['validated_data'] = validated_data
                
                return func(*args, **kwargs)
            except Exception as e:
                return jsonify({'success': False, 'message': 'Validation error', 'error': str(e)}), 400
        return wrapper
    return decorator

def caching_middleware(ttl: int = 300, key_prefix: str = None):
    """Caching middleware decorator with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            prefix = key_prefix or func.__name__
            cache_key = f"{prefix}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            if hasattr(current_app, 'cache'):
                cached_result = current_app.cache.get(cache_key)
                if cached_result is not None:
                    logger.info(f"📦 Cache hit for {func.__name__}")
                    return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            if hasattr(current_app, 'cache'):
                current_app.cache.set(cache_key, result, timeout=ttl)
                logger.info(f"💾 Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator

# Utility functions
def get_request_metrics() -> Dict[str, Any]:
    """Get request metrics with early returns."""
    if not hasattr(g, 'start_time'):
        return {}
    
    processing_time = time.perf_counter() - g.start_time
    
    return {
        'request_id': getattr(g, 'request_id', 'unknown'),
        'processing_time': processing_time,
        'method': request.method,
        'path': request.path,
        'user_agent': request.headers.get('User-Agent'),
        'client_ip': request.remote_addr,
        'timestamp': time.time()
    }

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics with early returns."""
    if not hasattr(g, 'performance_metrics'):
        return {}
    
    return g.performance_metrics

def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log security event with early returns."""
    if not event_type or not details:
        return
    
    logger.warning(f"🔒 Security event: {event_type} - {details}")

def check_request_origin(origin: str, allowed_origins: List[str]) -> bool:
    """Check request origin with early returns."""
    if not origin or not allowed_origins:
        return False
    
    return origin in allowed_origins

def validate_request_size(max_size: int = 1024 * 1024) -> bool:
    """Validate request size with early returns."""
    content_length = request.content_length
    if not content_length:
        return True
    
    return content_length <= max_size









