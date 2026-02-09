"""
Request handlers for Ultimate Enhanced Supreme Production system
"""

import time
import logging
from flask import request, g
from app.utils.decorators import performance_monitor

logger = logging.getLogger(__name__)

@performance_monitor
def before_request_handler():
    """Handle requests before processing."""
    g.start_time = time.perf_counter()
    g.request_id = f"req_{int(time.time() * 1000)}"
    
    logger.info(f"📥 Request started: {request.method} {request.path} [{g.request_id}]")
    
    # Log request details
    if request.is_json:
        logger.debug(f"📄 Request body: {request.get_json()}")
    
    # Add request ID to response headers
    from flask import make_response
    response = make_response()
    response.headers['X-Request-ID'] = g.request_id
    return response

@performance_monitor
def after_request_handler(response):
    """Handle responses after processing."""
    if hasattr(g, 'start_time'):
        processing_time = time.perf_counter() - g.start_time
        logger.info(f"📤 Request completed: {request.method} {request.path} [{g.request_id}] in {processing_time:.3f}s")
        
        # Add processing time to response headers
        response.headers['X-Processing-Time'] = str(processing_time)
    
    # Add CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    return response

@performance_monitor
def teardown_request_handler(exception):
    """Handle request teardown."""
    if exception:
        logger.error(f"❌ Request failed: {request.method} {request.path} [{getattr(g, 'request_id', 'unknown')}] - {exception}")
    else:
        logger.debug(f"✅ Request teardown: {request.method} {request.path} [{getattr(g, 'request_id', 'unknown')}]")
    
    # Clean up request context
    if hasattr(g, 'start_time'):
        delattr(g, 'start_time')
    if hasattr(g, 'request_id'):
        delattr(g, 'request_id')