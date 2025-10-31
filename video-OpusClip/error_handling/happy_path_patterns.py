#!/usr/bin/env python3
"""
Happy Path Patterns for Video-OpusClip
Keep the happy path last in function bodies
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta

from .custom_exceptions import (
    ValidationError, InputValidationError, TypeValidationError,
    SecurityError, AuthenticationError, AuthorizationError
)


# ============================================================================
# HAPPY PATH PATTERNS
# ============================================================================

def process_scan_request_happy_path_last(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process scan request with happy path last - avoiding nested conditionals
    
    Args:
        target: Target to scan
        scan_type: Type of scan
        options: Scan options
        
    Returns:
        Scan results
        
    Raises:
        Various exceptions for error conditions
    """
    # Early returns for error conditions (guard clauses)
    
    # Check if target is provided
    if target is None:
        raise InputValidationError("Scan target is required", "target", target)
    
    # Check if target is valid
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    # Check if host is provided
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    # Check if host is valid
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    # Check if scan type is valid
    if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
        raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
    
    # Check if target is unreachable
    if host == "unreachable.example.com":
        raise InputValidationError("Target is not reachable", "host", host)
    
    # Check if target is protected
    if host == "protected.example.com":
        raise AuthorizationError("Target is protected", "host", host)
    
    # Check if scan is already running
    if host == "scanning.example.com":
        raise ValidationError("Scan already in progress", "host", host)
    
    # Check if rate limit exceeded
    if host == "rate_limited.example.com":
        raise SecurityError("Rate limit exceeded", "host", host)
    
    # Check if options are valid
    if options is not None:
        if not isinstance(options, dict):
            raise TypeValidationError("Options must be a dictionary", "options", options, "dict", type(options).__name__)
        
        timeout = options.get('timeout')
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise InputValidationError("Timeout must be a positive number", "timeout", timeout)
    
    # Happy path: Process the scan request
    scan_results = {
        'target': target,
        'scan_type': scan_type,
        'status': 'completed',
        'timestamp': datetime.utcnow(),
        'results': {
            'open_ports': [80, 443, 22],
            'vulnerabilities': [],
            'services': ['http', 'https', 'ssh']
        }
    }
    
    return scan_results


async def process_async_scan_request_happy_path_last(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process async scan request with happy path last - avoiding nested conditionals
    
    Args:
        target: Target to scan
        scan_type: Type of scan
        options: Scan options
        
    Returns:
        Scan results
        
    Raises:
        Various exceptions for error conditions
    """
    # Early returns for error conditions (guard clauses)
    
    # Check if target is provided
    if target is None:
        raise InputValidationError("Scan target is required", "target", target)
    
    # Check if target is valid
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    # Check if host is provided
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    # Check if host is valid
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    # Check if scan type is valid
    if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
        raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
    
    # Async check: Check if target is reachable
    try:
        await asyncio.sleep(0.001)  # Simulate network check
        if host == "unreachable.example.com":
            raise InputValidationError("Target is not reachable", "host", host)
    except Exception as e:
        if isinstance(e, InputValidationError):
            raise
        raise InputValidationError(f"Failed to check target reachability: {str(e)}", "host", host)
    
    # Async check: Check if target is protected
    try:
        await asyncio.sleep(0.001)  # Simulate auth check
        if host == "protected.example.com":
            raise AuthorizationError("Target is protected", "host", host)
    except Exception as e:
        if isinstance(e, AuthorizationError):
            raise
        raise AuthorizationError(f"Failed to check target protection: {str(e)}", "host", host)
    
    # Async check: Check if scan is already running
    try:
        await asyncio.sleep(0.001)  # Simulate status check
        if host == "scanning.example.com":
            raise ValidationError("Scan already in progress", "host", host)
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to check scan status: {str(e)}", "host", host)
    
    # Async check: Check if rate limit exceeded
    try:
        await asyncio.sleep(0.001)  # Simulate rate limit check
        if host == "rate_limited.example.com":
            raise SecurityError("Rate limit exceeded", "host", host)
    except Exception as e:
        if isinstance(e, SecurityError):
            raise
        raise SecurityError(f"Failed to check rate limit: {str(e)}", "host", host)
    
    # Check if options are valid
    if options is not None:
        if not isinstance(options, dict):
            raise TypeValidationError("Options must be a dictionary", "options", options, "dict", type(options).__name__)
        
        timeout = options.get('timeout')
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise InputValidationError("Timeout must be a positive number", "timeout", timeout)
    
    # Happy path: Process the async scan request
    await asyncio.sleep(0.1)  # Simulate scan processing
    
    scan_results = {
        'target': target,
        'scan_type': scan_type,
        'status': 'completed',
        'timestamp': datetime.utcnow(),
        'results': {
            'open_ports': [80, 443, 22],
            'vulnerabilities': [],
            'services': ['http', 'https', 'ssh']
        }
    }
    
    return scan_results


def process_enumeration_request_happy_path_last(
    target: str,
    enum_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process enumeration request with happy path last - avoiding nested conditionals
    
    Args:
        target: Target to enumerate
        enum_type: Type of enumeration
        options: Enumeration options
        
    Returns:
        Enumeration results
        
    Raises:
        Various exceptions for error conditions
    """
    # Early returns for error conditions (guard clauses)
    
    # Check if target is provided
    if target is None:
        raise InputValidationError("Enumeration target is required", "target", target)
    
    # Check if target is valid
    if not isinstance(target, str):
        raise TypeValidationError("Target must be a string", "target", target, "str", type(target).__name__)
    
    # Check if target is empty
    if not target.strip():
        raise InputValidationError("Target cannot be empty", "target", target)
    
    # Check if enum type is valid
    if enum_type not in ['dns', 'smb', 'ssh', 'user', 'service']:
        raise InputValidationError(f"Invalid enumeration type: {enum_type}", "enum_type", enum_type)
    
    # Check if target is unreachable
    if target == "unreachable.example.com":
        raise InputValidationError("Target is not reachable", "target", target)
    
    # Check if target is protected
    if target == "protected.example.com":
        raise AuthorizationError("Target is protected", "target", target)
    
    # Check if enumeration is already running
    if target == "enumerating.example.com":
        raise ValidationError("Enumeration already in progress", "target", target)
    
    # Check if rate limit exceeded
    if target == "rate_limited.example.com":
        raise SecurityError("Rate limit exceeded", "target", target)
    
    # Check if options are valid
    if options is not None:
        if not isinstance(options, dict):
            raise TypeValidationError("Options must be a dictionary", "options", options, "dict", type(options).__name__)
        
        max_records = options.get('max_records')
        if max_records is not None and (not isinstance(max_records, int) or max_records <= 0):
            raise InputValidationError("max_records must be a positive integer", "max_records", max_records)
    
    # Happy path: Process the enumeration request
    enumeration_results = {
        'target': target,
        'enum_type': enum_type,
        'status': 'completed',
        'timestamp': datetime.utcnow(),
        'results': {
            'records': ['record1', 'record2', 'record3'],
            'count': 3,
            'duration': 1.5
        }
    }
    
    return enumeration_results


def process_attack_request_happy_path_last(
    target: str,
    attack_type: str,
    payload: Optional[str] = None,
    credentials: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Process attack request with happy path last - avoiding nested conditionals
    
    Args:
        target: Target to attack
        attack_type: Type of attack
        payload: Attack payload
        credentials: Credentials for attack
        
    Returns:
        Attack results
        
    Raises:
        Various exceptions for error conditions
    """
    # Early returns for error conditions (guard clauses)
    
    # Check if target is provided
    if target is None:
        raise InputValidationError("Attack target is required", "target", target)
    
    # Check if target is valid
    if not isinstance(target, str):
        raise TypeValidationError("Target must be a string", "target", target, "str", type(target).__name__)
    
    # Check if target is empty
    if not target.strip():
        raise InputValidationError("Target cannot be empty", "target", target)
    
    # Check if attack type is valid
    if attack_type not in ['brute_force', 'exploitation', 'social_engineering']:
        raise InputValidationError(f"Invalid attack type: {attack_type}", "attack_type", attack_type)
    
    # Check if target is unreachable
    if target == "unreachable.example.com":
        raise InputValidationError("Target is not reachable", "target", target)
    
    # Check if target is protected
    if target == "protected.example.com":
        raise AuthorizationError("Target is protected", "target", target)
    
    # Check if attack is already running
    if target == "attacking.example.com":
        raise ValidationError("Attack already in progress", "target", target)
    
    # Check if rate limit exceeded
    if target == "rate_limited.example.com":
        raise SecurityError("Rate limit exceeded", "target", target)
    
    # Check if payload is valid (if provided)
    if payload is not None:
        if not isinstance(payload, str):
            raise TypeValidationError("Payload must be a string", "payload", payload, "str", type(payload).__name__)
        
        if not payload.strip():
            raise InputValidationError("Payload cannot be empty", "payload", payload)
        
        # Check for dangerous payload patterns
        dangerous_patterns = ['<script>', 'javascript:', 'data:text/html']
        for pattern in dangerous_patterns:
            if pattern.lower() in payload.lower():
                raise SecurityError(f"Dangerous payload pattern detected: {pattern}", "payload", payload)
    
    # Check if credentials are valid (if provided)
    if credentials is not None:
        if not isinstance(credentials, list):
            raise TypeValidationError("Credentials must be a list", "credentials", credentials, "list", type(credentials).__name__)
        
        for i, cred in enumerate(credentials):
            if not isinstance(cred, dict):
                raise TypeValidationError(f"Credential {i} must be a dictionary", f"credential_{i}", cred, "dict", type(cred).__name__)
            
            username = cred.get('username')
            password = cred.get('password')
            
            if username is not None and not isinstance(username, str):
                raise TypeValidationError(f"Username in credential {i} must be a string", f"username_{i}", username, "str", type(username).__name__)
            
            if password is not None and not isinstance(password, str):
                raise TypeValidationError(f"Password in credential {i} must be a string", f"password_{i}", password, "str", type(password).__name__)
    
    # Happy path: Process the attack request
    attack_results = {
        'target': target,
        'attack_type': attack_type,
        'status': 'completed',
        'timestamp': datetime.utcnow(),
        'results': {
            'attempts': 100,
            'successful': 0,
            'credentials_found': [],
            'vulnerabilities_found': []
        }
    }
    
    return attack_results


def process_authentication_request_happy_path_last(
    username: str,
    password: str,
    auth_method: str = "password"
) -> Dict[str, Any]:
    """
    Process authentication request with happy path last - avoiding nested conditionals
    
    Args:
        username: Username
        password: Password
        auth_method: Authentication method
        
    Returns:
        Authentication results
        
    Raises:
        Various exceptions for error conditions
    """
    # Early returns for error conditions (guard clauses)
    
    # Check if username is provided
    if username is None:
        raise InputValidationError("Username is required", "username", username)
    
    # Check if username is valid
    if not isinstance(username, str):
        raise TypeValidationError("Username must be a string", "username", username, "str", type(username).__name__)
    
    # Check if username is empty
    if not username.strip():
        raise InputValidationError("Username cannot be empty", "username", username)
    
    # Check if username is too short
    if len(username.strip()) < 3:
        raise InputValidationError("Username must be at least 3 characters long", "username", username)
    
    # Check if username is too long
    if len(username) > 50:
        raise InputValidationError("Username must be at most 50 characters long", "username", username)
    
    # Check if password is provided
    if password is None:
        raise InputValidationError("Password is required", "password", password)
    
    # Check if password is valid
    if not isinstance(password, str):
        raise TypeValidationError("Password must be a string", "password", password, "str", type(password).__name__)
    
    # Check if password is too short
    if len(password) < 8:
        raise InputValidationError("Password must be at least 8 characters long", "password", password)
    
    # Check if password is too long
    if len(password) > 128:
        raise InputValidationError("Password must be at most 128 characters long", "password", password)
    
    # Check if auth method is valid
    if auth_method not in ['password', 'token', 'oauth', 'saml']:
        raise InputValidationError(f"Invalid auth method: {auth_method}", "auth_method", auth_method)
    
    # Check if user is blocked
    if username == "blocked_user":
        raise AuthenticationError("User is blocked", "username", username)
    
    # Check if account is locked
    if username == "locked_user":
        raise AuthenticationError("Account is locked", "username", username)
    
    # Check if credentials are invalid
    if username == "invalid_user" or password == "wrong_password":
        raise AuthenticationError("Invalid credentials", "credentials", {"username": username})
    
    # Check if account is expired
    if username == "expired_user":
        raise AuthenticationError("Account has expired", "username", username)
    
    # Check if too many failed attempts
    if username == "rate_limited_user":
        raise SecurityError("Too many failed login attempts", "username", username)
    
    # Happy path: Process the authentication request
    auth_results = {
        'username': username,
        'auth_method': auth_method,
        'status': 'success',
        'timestamp': datetime.utcnow(),
        'session_token': 'session_token_123',
        'expires_at': datetime.utcnow() + timedelta(hours=24),
        'permissions': ['read', 'write', 'execute']
    }
    
    return auth_results


def process_authorization_request_happy_path_last(
    user_id: str,
    action: str,
    resource: str,
    permissions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process authorization request with happy path last - avoiding nested conditionals
    
    Args:
        user_id: User identifier
        action: Action to perform
        resource: Resource to access
        permissions: User permissions
        
    Returns:
        Authorization results
        
    Raises:
        Various exceptions for error conditions
    """
    # Early returns for error conditions (guard clauses)
    
    # Check if user_id is provided
    if user_id is None:
        raise InputValidationError("User ID is required", "user_id", user_id)
    
    # Check if user_id is valid
    if not isinstance(user_id, str):
        raise TypeValidationError("User ID must be a string", "user_id", user_id, "str", type(user_id).__name__)
    
    # Check if user_id is empty
    if not user_id.strip():
        raise InputValidationError("User ID cannot be empty", "user_id", user_id)
    
    # Check if action is provided
    if action is None:
        raise InputValidationError("Action is required", "action", action)
    
    # Check if action is valid
    if not isinstance(action, str):
        raise TypeValidationError("Action must be a string", "action", action, "str", type(action).__name__)
    
    # Check if action is empty
    if not action.strip():
        raise InputValidationError("Action cannot be empty", "action", action)
    
    # Check if action is valid
    if action not in ['read', 'write', 'delete', 'execute']:
        raise InputValidationError(f"Invalid action: {action}", "action", action)
    
    # Check if resource is provided
    if resource is None:
        raise InputValidationError("Resource is required", "resource", resource)
    
    # Check if resource is valid
    if not isinstance(resource, str):
        raise TypeValidationError("Resource must be a string", "resource", resource, "str", type(resource).__name__)
    
    # Check if resource is empty
    if not resource.strip():
        raise InputValidationError("Resource cannot be empty", "resource", resource)
    
    # Check if user is blocked
    if user_id == "blocked_user":
        raise AuthorizationError("User is blocked", "user_id", user_id)
    
    # Check if user is suspended
    if user_id == "suspended_user":
        raise AuthorizationError("User is suspended", "user_id", user_id)
    
    # Check if resource is protected
    protected_resources = ['/admin', '/system', '/config']
    if resource in protected_resources:
        raise AuthorizationError(f"Resource '{resource}' is protected", "resource", resource)
    
    # Check if action is not allowed
    if action == "delete" and resource == "/readonly":
        raise AuthorizationError(f"Action '{action}' is not allowed on '{resource}'", "action", action)
    
    # Check if user lacks permissions
    if permissions is not None:
        required_permission = f"{action}_{resource.replace('/', '_')}"
        if required_permission not in permissions:
            raise AuthorizationError(f"Missing permission: {required_permission}", "permissions", permissions)
    
    # Check if resource doesn't exist
    if resource == "/nonexistent":
        raise InputValidationError("Resource does not exist", "resource", resource)
    
    # Happy path: Process the authorization request
    authz_results = {
        'user_id': user_id,
        'action': action,
        'resource': resource,
        'status': 'authorized',
        'timestamp': datetime.utcnow(),
        'permissions_checked': permissions or [],
        'session_valid': True
    }
    
    return authz_results


# ============================================================================
# COMPLEX HAPPY PATH PATTERNS
# ============================================================================

def process_complex_scan_workflow_happy_path_last(
    target: Dict[str, Any],
    scan_config: Dict[str, Any],
    user_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process complex scan workflow with happy path last - avoiding nested conditionals
    
    Args:
        target: Target to scan
        scan_config: Scan configuration
        user_context: User context
        
    Returns:
        Scan workflow results
        
    Raises:
        Various exceptions for error conditions
    """
    # Early returns for error conditions (guard clauses)
    
    # Validate target
    if target is None:
        raise InputValidationError("Target is required", "target", target)
    
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    # Validate scan configuration
    if scan_config is None:
        raise InputValidationError("Scan configuration is required", "scan_config", scan_config)
    
    if not isinstance(scan_config, dict):
        raise TypeValidationError("Scan configuration must be a dictionary", "scan_config", scan_config, "dict", type(scan_config).__name__)
    
    scan_type = scan_config.get('scan_type')
    if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
        raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
    
    # Validate user context
    if user_context is None:
        raise InputValidationError("User context is required", "user_context", user_context)
    
    if not isinstance(user_context, dict):
        raise TypeValidationError("User context must be a dictionary", "user_context", user_context, "dict", type(user_context).__name__)
    
    user_id = user_context.get('user_id')
    if user_id is None:
        raise InputValidationError("User ID is required", "user_id", user_id)
    
    # Check user permissions
    permissions = user_context.get('permissions', [])
    required_permission = f"scan_{scan_type}"
    if required_permission not in permissions:
        raise AuthorizationError(f"Missing permission: {required_permission}", "permissions", permissions)
    
    # Check user quota
    quota_used = user_context.get('quota_used', 0)
    quota_limit = user_context.get('quota_limit', 100)
    if quota_used >= quota_limit:
        raise SecurityError("User quota exceeded", "quota", {"used": quota_used, "limit": quota_limit})
    
    # Check target accessibility
    if host == "unreachable.example.com":
        raise InputValidationError("Target is not reachable", "host", host)
    
    if host == "protected.example.com":
        raise AuthorizationError("Target is protected", "host", host)
    
    # Check scan limits
    max_scans = scan_config.get('max_scans', 10)
    current_scans = user_context.get('current_scans', 0)
    if current_scans >= max_scans:
        raise ValidationError("Maximum concurrent scans reached", "scans", {"current": current_scans, "max": max_scans})
    
    # Check rate limiting
    if user_context.get('rate_limited', False):
        raise SecurityError("Rate limit exceeded", "rate_limit", True)
    
    # Check maintenance mode
    if user_context.get('maintenance_mode', False):
        raise ValidationError("System is in maintenance mode", "maintenance", True)
    
    # Happy path: Process the complex scan workflow
    workflow_results = {
        'target': target,
        'scan_config': scan_config,
        'user_context': user_context,
        'status': 'completed',
        'timestamp': datetime.utcnow(),
        'workflow_id': 'workflow_123',
        'results': {
            'scan_id': 'scan_456',
            'duration': 45.2,
            'ports_scanned': 1000,
            'vulnerabilities_found': 3,
            'services_detected': ['http', 'https', 'ssh', 'ftp']
        },
        'quota_updated': {
            'quota_used': quota_used + 1,
            'quota_limit': quota_limit
        }
    }
    
    return workflow_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of happy path patterns
    print("🎯 Happy Path Patterns Example")
    
    # Test scan request processing
    try:
        scan_results = process_scan_request_happy_path_last(
            target={'host': '192.168.1.100', 'port': 80},
            scan_type='port_scan',
            options={'timeout': 30}
        )
        print(f"✅ Scan processing successful: {scan_results['status']}")
    except Exception as e:
        print(f"❌ Scan processing failed: {e}")
    
    # Test enumeration request processing
    try:
        enum_results = process_enumeration_request_happy_path_last(
            target='example.com',
            enum_type='dns',
            options={'max_records': 100}
        )
        print(f"✅ Enumeration processing successful: {enum_results['status']}")
    except Exception as e:
        print(f"❌ Enumeration processing failed: {e}")
    
    # Test attack request processing
    try:
        attack_results = process_attack_request_happy_path_last(
            target='192.168.1.100',
            attack_type='brute_force',
            credentials=[{'username': 'admin', 'password': 'password'}]
        )
        print(f"✅ Attack processing successful: {attack_results['status']}")
    except Exception as e:
        print(f"❌ Attack processing failed: {e}")
    
    # Test authentication request processing
    try:
        auth_results = process_authentication_request_happy_path_last(
            username='john_doe',
            password='SecureP@ss123'
        )
        print(f"✅ Authentication processing successful: {auth_results['status']}")
    except Exception as e:
        print(f"❌ Authentication processing failed: {e}")
    
    # Test authorization request processing
    try:
        authz_results = process_authorization_request_happy_path_last(
            user_id='user123',
            action='read',
            resource='/data',
            permissions=['read_data', 'write_data']
        )
        print(f"✅ Authorization processing successful: {authz_results['status']}")
    except Exception as e:
        print(f"❌ Authorization processing failed: {e}")
    
    # Test complex workflow processing
    try:
        workflow_results = process_complex_scan_workflow_happy_path_last(
            target={'host': '192.168.1.100'},
            scan_config={'scan_type': 'port_scan', 'max_scans': 10},
            user_context={
                'user_id': 'user123',
                'permissions': ['scan_port_scan'],
                'quota_used': 5,
                'quota_limit': 100,
                'current_scans': 2
            }
        )
        print(f"✅ Complex workflow processing successful: {workflow_results['status']}")
    except Exception as e:
        print(f"❌ Complex workflow processing failed: {e}")
    
    # Test async scan processing
    async def test_async_scan():
        try:
            scan_results = await process_async_scan_request_happy_path_last(
                target={'host': '192.168.1.100'},
                scan_type='port_scan',
                options={'timeout': 30}
            )
            print(f"✅ Async scan processing successful: {scan_results['status']}")
        except Exception as e:
            print(f"❌ Async scan processing failed: {e}")
    
    asyncio.run(test_async_scan())
    
    print("\n✅ Happy path patterns examples completed!") 