"""
Authentication Blueprint
========================

Ultra-advanced authentication system with JWT and security.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token, create_refresh_token, jwt_required,
    get_jwt_identity, get_jwt, JWTManager
)
from werkzeug.security import check_password_hash, generate_password_hash
from marshmallow import Schema, fields, validate, ValidationError
from sqlalchemy.exc import IntegrityError
from app import db
from models import User, SecurityEvent
from utils.decorators import rate_limit, validate_json
from utils.exceptions import AuthenticationError, ValidationError as CustomValidationError

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

@auth_bp.route('/login', methods=['POST'])
@rate_limit(limit="5 per minute")
@validate_json
def login():
    """
    User login endpoint.
    
    Returns:
        JSON response with access and refresh tokens
    """
    try:
        # Validate request data
        data = login_schema.load(request.json)
        
        # Find user
        user = User.query.filter_by(username=data['username']).first()
        if not user or not check_password_hash(user.password_hash, data['password']):
            # Log failed login attempt
            _log_security_event('failed_login', 'medium', f"Failed login attempt for username: {data['username']}")
            raise AuthenticationError("Invalid username or password")
        
        # Check if user is active
        if not user.is_active:
            _log_security_event('account_disabled', 'high', f"Login attempt for disabled account: {user.username}")
            raise AuthenticationError("Account is disabled")
        
        # Update last login
        user.last_login = datetime.utcnow()
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
            'user': {
                'id': str(user.id),
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'last_login': user.last_login.isoformat() if user.last_login else None
            }
        }), 200
        
    except ValidationError as e:
        raise CustomValidationError(f"Validation error: {e.messages}")
    except AuthenticationError as e:
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        current_app.logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/register', methods=['POST'])
@rate_limit(limit="3 per minute")
@validate_json
def register():
    """
    User registration endpoint.
    
    Returns:
        JSON response with user information
    """
    try:
        # Validate request data
        data = register_schema.load(request.json)
        
        # Check password confirmation
        if data['password'] != data['confirm_password']:
            raise CustomValidationError("Passwords do not match")
        
        # Check if user already exists
        if User.query.filter_by(username=data['username']).first():
            raise CustomValidationError("Username already exists")
        
        if User.query.filter_by(email=data['email']).first():
            raise CustomValidationError("Email already exists")
        
        # Create new user
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password'])
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Log registration
        _log_security_event('user_registration', 'low', f"New user registered: {user.username}")
        
        return jsonify({
            'message': 'User registered successfully',
            'user': {
                'id': str(user.id),
                'username': user.username,
                'email': user.email
            }
        }), 201
        
    except ValidationError as e:
        raise CustomValidationError(f"Validation error: {e.messages}")
    except IntegrityError:
        db.session.rollback()
        raise CustomValidationError("Username or email already exists")
    except Exception as e:
        current_app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """
    Refresh access token endpoint.
    
    Returns:
        JSON response with new access token
    """
    try:
        current_user_id = get_jwt_identity()
        new_token = create_access_token(identity=current_user_id)
        
        return jsonify({'access_token': new_token}), 200
        
    except Exception as e:
        current_app.logger.error(f"Token refresh error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """
    User logout endpoint.
    
    Returns:
        JSON response with logout confirmation
    """
    try:
        # Get current user
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if user:
            # Log logout
            _log_security_event('user_logout', 'low', f"User logged out: {user.username}")
        
        return jsonify({'message': 'Logged out successfully'}), 200
        
    except Exception as e:
        current_app.logger.error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """
    Get user profile endpoint.
    
    Returns:
        JSON response with user profile
    """
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user:
            raise AuthenticationError("User not found")
        
        return jsonify({
            'user': {
                'id': str(user.id),
                'username': user.username,
                'email': user.email,
                'is_active': user.is_active,
                'is_admin': user.is_admin,
                'created_at': user.created_at.isoformat(),
                'last_login': user.last_login.isoformat() if user.last_login else None
            }
        }), 200
        
    except AuthenticationError as e:
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        current_app.logger.error(f"Get profile error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
@rate_limit(limit="3 per minute")
@validate_json
def change_password():
    """
    Change password endpoint.
    
    Returns:
        JSON response with change confirmation
    """
    try:
        # Validate request data
        data = change_password_schema.load(request.json)
        
        # Check password confirmation
        if data['new_password'] != data['confirm_password']:
            raise CustomValidationError("New passwords do not match")
        
        # Get current user
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user:
            raise AuthenticationError("User not found")
        
        # Check current password
        if not check_password_hash(user.password_hash, data['current_password']):
            _log_security_event('failed_password_change', 'medium', f"Failed password change attempt for user: {user.username}")
            raise AuthenticationError("Current password is incorrect")
        
        # Update password
        user.password_hash = generate_password_hash(data['new_password'])
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        # Log password change
        _log_security_event('password_changed', 'medium', f"Password changed for user: {user.username}")
        
        return jsonify({'message': 'Password changed successfully'}), 200
        
    except ValidationError as e:
        raise CustomValidationError(f"Validation error: {e.messages}")
    except AuthenticationError as e:
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        current_app.logger.error(f"Change password error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

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









