"""
Error handlers for Ultimate Enhanced Supreme Production system
"""

import time
import logging
from flask import jsonify, request
from marshmallow import ValidationError

logger = logging.getLogger(__name__)

def handle_validation_error(error):
    """Handle validation errors."""
    logger.error(f"❌ Validation error: {error}")
    return jsonify({
        'success': False,
        'message': 'Validation error',
        'error': str(error),
        'timestamp': time.time()
    }), 400

def handle_not_found_error(error):
    """Handle not found errors."""
    logger.error(f"❌ Not found error: {error}")
    return jsonify({
        'success': False,
        'message': 'Resource not found',
        'error': str(error),
        'timestamp': time.time()
    }), 404

def handle_unauthorized_error(error):
    """Handle unauthorized errors."""
    logger.error(f"❌ Unauthorized error: {error}")
    return jsonify({
        'success': False,
        'message': 'Unauthorized access',
        'error': str(error),
        'timestamp': time.time()
    }), 401

def handle_forbidden_error(error):
    """Handle forbidden errors."""
    logger.error(f"❌ Forbidden error: {error}")
    return jsonify({
        'success': False,
        'message': 'Access forbidden',
        'error': str(error),
        'timestamp': time.time()
    }), 403

def handle_internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({
        'success': False,
        'message': 'Internal server error',
        'error': str(error),
        'timestamp': time.time()
    }), 500

def handle_generic_error(error):
    """Handle generic errors."""
    logger.error(f"❌ Generic error: {error}")
    return jsonify({
        'success': False,
        'message': 'An error occurred',
        'error': str(error),
        'timestamp': time.time()
    }), 500