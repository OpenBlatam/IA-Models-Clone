"""
Authentication Routes
====================

Functional authentication routes following Flask best practices.
"""

import logging
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify, g, current_app
from flask_jwt_extended import (
    create_access_token, create_refresh_token, jwt_required,
    get_jwt_identity, get_jwt
)
from werkzeug.security import check_password_hash, generate_password_hash
from marshmallow import Schema, fields, validate, ValidationError
from sqlalchemy.exc import IntegrityError
from app import db
from models import User, SecurityEvent
from utils.functional import (
    validate_email, validate_password, validate_username,
    generate_uuid, sanitize_string, get_current_timestamp,
    handle_errors, measure_time, log_function_call
)
from utils.decorators import rate_limit, validate_json, monitor_performance

logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Schemas
class LoginSchema(Schema):
    """Login request schema."""
    username = fields.Str(required=True, validate=validate.Length(min=3, max=80))
    password = fields.Str(required=True, validate=validate.Length(min=8, max=128))

class RegisterSchema(Schema):
    """Registration request schema."""
    username = fields.Str(required=True, validate=validate.Length(min=3, max=80))
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=8, max=128))
    confirm_password = fields.Str(required=True, validate=validate.Length(min=8, max=128))

class ChangePasswordSchema(Schema):
    """Change password request schema."""
    current_password = fields.Str(required=True)
    new_password = fields.Str(required=True, validate=validate.Length(min=8, max=128))
    confirm_password = fields.Str(required=True)

# Initialize schemas
login_schema = LoginSchema()
register_schema = RegisterSchema()
change_password_schema = ChangePasswordSchema()

# Helper functions
def _log_security_event(event_type: str, severity: str, description: str) -> None:
    """Log security event."""
    try:
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_id=g.get('request_id')
        )
        db.session.add(event)
        db.session.commit()
    except Exception as e:
        current_app.logger.error(f"Failed to log security event: {str(e)}")

def _validate_user_data(data: Dict[str, Any]) -> Dict[str, str]:
    """Validate user data."""
    errors = {}
    
    if not validate_username(data.get('username', '')):
        errors['username'] = 'Invalid username format'
    
    if not validate_email(data.get('email', '')):
        errors['email'] = 'Invalid email format'
    
    if not validate_password(data.get('password', '')):
        errors['password'] = 'Password must be at least 8 characters with uppercase, lowercase, digit, and special character'
    
    return errors

def _create_user_response(user: User) -> Dict[str, Any]:
    """Create user response object."""
    return {
        'id': str(user.id),
        'username': user.username,
        'email': user.email,
        'is_active': user.is_active,
        'is_admin': user.is_admin,
        'created_at': user.created_at.isoformat(),
        'last_login': user.last_login.isoformat() if user.last_login else None
    }

# Route handlers
@auth_bp.route('/login', methods=['POST'])
@rate_limit(limit="5 per minute")
@validate_json
@monitor_performance("user_login")
@handle_errors
@measure_time
@log_function_call
def login() -> Dict[str, Any]:
    """
    User login endpoint.
    
    Returns:
        JSON response with access and refresh tokens
    """
    # Validate request data
    try:
        data = login_schema.load(request.json)
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    
    # Find user
    user = User.query.filter_by(username=data['username']).first()
    if not user or not check_password_hash(user.password_hash, data['password']):
        _log_security_event('failed_login', 'medium', f"Failed login attempt for username: {data['username']}")
        return jsonify({'error': 'Invalid username or password'}), 401
    
    # Check if user is active
    if not user.is_active:
        _log_security_event('account_disabled', 'high', f"Login attempt for disabled account: {user.username}")
        return jsonify({'error': 'Account is disabled'}), 401
    
    # Update last login
    user.last_login = get_current_timestamp()
    db.session.commit()
    
    # Create tokens
    access_token = create_access_token(
        identity=str(user.id),
        expires_delta=timedelta(seconds=current_app.config['JWT_ACCESS_TOKEN_EXPIRES'])
    )
    refresh_token = create_refresh_token(identity=str(user.id))
    
    # Log successful login
    _log_security_event('successful_login', 'low', f"Successful login for user: {user.username}")
    
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': _create_user_response(user)
    }), 200

@auth_bp.route('/register', methods=['POST'])
@rate_limit(limit="3 per minute")
@validate_json
@monitor_performance("user_registration")
@handle_errors
@measure_time
@log_function_call
def register() -> Dict[str, Any]:
    """
    User registration endpoint.
    
    Returns:
        JSON response with user information
    """
    # Validate request data
    try:
        data = register_schema.load(request.json)
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    
    # Check password confirmation
    if data['password'] != data['confirm_password']:
        return jsonify({'error': 'Passwords do not match'}), 400
    
    # Validate user data
    validation_errors = _validate_user_data(data)
    if validation_errors:
        return jsonify({'error': 'Validation failed', 'details': validation_errors}), 400
    
    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    # Create new user
    try:
        user = User(
            username=sanitize_string(data['username']),
            email=sanitize_string(data['email']),
            password_hash=generate_password_hash(data['password'])
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Log registration
        _log_security_event('user_registration', 'low', f"New user registered: {user.username}")
        
        return jsonify({
            'message': 'User registered successfully',
            'user': _create_user_response(user)
        }), 201
        
    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'Username or email already exists'}), 400
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
@monitor_performance("token_refresh")
@handle_errors
@measure_time
@log_function_call
def refresh() -> Dict[str, Any]:
    """
    Refresh access token endpoint.
    
    Returns:
        JSON response with new access token
    """
    current_user_id = get_jwt_identity()
    new_token = create_access_token(identity=current_user_id)
    
    return jsonify({'access_token': new_token}), 200

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
@monitor_performance("user_logout")
@handle_errors
@measure_time
@log_function_call
def logout() -> Dict[str, Any]:
    """
    User logout endpoint.
    
    Returns:
        JSON response with logout confirmation
    """
    # Get current user
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if user:
        # Log logout
        _log_security_event('user_logout', 'low', f"User logged out: {user.username}")
    
    return jsonify({'message': 'Logged out successfully'}), 200

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
@monitor_performance("get_profile")
@handle_errors
@measure_time
@log_function_call
def get_profile() -> Dict[str, Any]:
    """
    Get user profile endpoint.
    
    Returns:
        JSON response with user profile
    """
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 401
    
    return jsonify({'user': _create_user_response(user)}), 200

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
@rate_limit(limit="3 per minute")
@validate_json
@monitor_performance("change_password")
@handle_errors
@measure_time
@log_function_call
def change_password() -> Dict[str, Any]:
    """
    Change password endpoint.
    
    Returns:
        JSON response with change confirmation
    """
    # Validate request data
    try:
        data = change_password_schema.load(request.json)
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    
    # Check password confirmation
    if data['new_password'] != data['confirm_password']:
        return jsonify({'error': 'New passwords do not match'}), 400
    
    # Validate new password
    if not validate_password(data['new_password']):
        return jsonify({'error': 'New password does not meet requirements'}), 400
    
    # Get current user
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 401
    
    # Check current password
    if not check_password_hash(user.password_hash, data['current_password']):
        _log_security_event('failed_password_change', 'medium', f"Failed password change attempt for user: {user.username}")
        return jsonify({'error': 'Current password is incorrect'}), 401
    
    # Update password
    user.password_hash = generate_password_hash(data['new_password'])
    user.updated_at = get_current_timestamp()
    db.session.commit()
    
    # Log password change
    _log_security_event('password_changed', 'medium', f"Password changed for user: {user.username}")
    
    return jsonify({'message': 'Password changed successfully'}), 200

# JWT error handlers
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    """Handle expired token."""
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    """Handle invalid token."""
    return jsonify({'error': 'Invalid token'}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    """Handle missing token."""
    return jsonify({'error': 'Authorization token is required'}), 401

@jwt.needs_fresh_token_loader
def token_not_fresh_callback(jwt_header, jwt_payload):
    """Handle non-fresh token."""
    return jsonify({'error': 'Fresh token required'}), 401

@jwt.revoked_token_loader
def revoked_token_callback(jwt_header, jwt_payload):
    """Handle revoked token."""
    return jsonify({'error': 'Token has been revoked'}), 401









