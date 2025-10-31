#!/usr/bin/env python3
"""
Advanced Guard Clauses for Video-OpusClip
Avoid nested conditionals and keep the happy path last
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta

from .custom_exceptions import (
    ValidationError, InputValidationError, TypeValidationError,
    SecurityError, AuthenticationError, AuthorizationError
)


# ============================================================================
# GUARD CLAUSE PATTERNS
# ============================================================================

def validate_scan_request_guard_clauses(
    target: Dict[str, Any],
    scan_type: str,
    timeout: Optional[float] = None,
    credentials: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Validate scan request using guard clauses - avoiding nested conditionals
    
    Args:
        target: Target to scan
        scan_type: Type of scan
        timeout: Timeout value
        credentials: Credentials for scan
        
    Returns:
        Validated scan configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Guard clause 1: Check if target is provided
    if target is None:
        raise InputValidationError("Scan target is required", "target", target)
    
    # Guard clause 2: Check if target is a dictionary
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    # Guard clause 3: Check if target is empty
    if not target:
        raise InputValidationError("Target cannot be empty", "target", target)
    
    # Guard clause 4: Check if host is provided
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    # Guard clause 5: Check if host is empty
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    # Guard clause 6: Check if scan_type is provided
    if scan_type is None:
        raise InputValidationError("Scan type is required", "scan_type", scan_type)
    
    # Guard clause 7: Check if scan_type is empty
    if not isinstance(scan_type, str) or not scan_type.strip():
        raise InputValidationError("Scan type cannot be empty", "scan_type", scan_type)
    
    # Guard clause 8: Check if scan_type is valid
    valid_scan_types = ['port_scan', 'vulnerability_scan', 'web_scan', 'network_scan']
    if scan_type not in valid_scan_types:
        raise InputValidationError(
            f"Invalid scan type. Must be one of: {', '.join(valid_scan_types)}",
            "scan_type",
            scan_type
        )
    
    # Guard clause 9: Check if timeout is valid (if provided)
    if timeout is not None:
        if not isinstance(timeout, (int, float)):
            raise TypeValidationError("Timeout must be a number", "timeout", timeout, "number", type(timeout).__name__)
        if timeout <= 0:
            raise InputValidationError("Timeout must be greater than 0", "timeout", timeout)
    
    # Guard clause 10: Check if credentials are valid (if provided)
    if credentials is not None:
        if not isinstance(credentials, dict):
            raise TypeValidationError("Credentials must be a dictionary", "credentials", credentials, "dict", type(credentials).__name__)
        
        username = credentials.get('username')
        password = credentials.get('password')
        
        if username is not None and not isinstance(username, str):
            raise TypeValidationError("Username must be a string", "username", username, "str", type(username).__name__)
        
        if password is not None and not isinstance(password, str):
            raise TypeValidationError("Password must be a string", "password", password, "str", type(password).__name__)
    
    # Happy path: Return validated configuration
    return {
        'target': {
            'host': host.strip(),
            'port': target.get('port'),
            'protocol': target.get('protocol', 'tcp')
        },
        'scan_type': scan_type,
        'timeout': timeout or 30.0,
        'credentials': credentials,
        'max_retries': 3,
        'user_agent': 'Video-OpusClip-Scanner/1.0'
    }


def validate_enumeration_request_guard_clauses(
    target: str,
    enum_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate enumeration request using guard clauses - avoiding nested conditionals
    
    Args:
        target: Target to enumerate
        enum_type: Type of enumeration
        options: Enumeration options
        
    Returns:
        Validated enumeration configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Guard clause 1: Check if target is provided
    if target is None:
        raise InputValidationError("Enumeration target is required", "target", target)
    
    # Guard clause 2: Check if target is a string
    if not isinstance(target, str):
        raise TypeValidationError("Target must be a string", "target", target, "str", type(target).__name__)
    
    # Guard clause 3: Check if target is empty
    if not target.strip():
        raise InputValidationError("Target cannot be empty", "target", target)
    
    # Guard clause 4: Check if enum_type is provided
    if enum_type is None:
        raise InputValidationError("Enumeration type is required", "enum_type", enum_type)
    
    # Guard clause 5: Check if enum_type is a string
    if not isinstance(enum_type, str):
        raise TypeValidationError("Enumeration type must be a string", "enum_type", enum_type, "str", type(enum_type).__name__)
    
    # Guard clause 6: Check if enum_type is empty
    if not enum_type.strip():
        raise InputValidationError("Enumeration type cannot be empty", "enum_type", enum_type)
    
    # Guard clause 7: Check if enum_type is valid
    valid_enum_types = ['dns', 'smb', 'ssh', 'user', 'service']
    if enum_type not in valid_enum_types:
        raise InputValidationError(
            f"Invalid enumeration type. Must be one of: {', '.join(valid_enum_types)}",
            "enum_type",
            enum_type
        )
    
    # Guard clause 8: Check if options are valid (if provided)
    if options is not None:
        if not isinstance(options, dict):
            raise TypeValidationError("Options must be a dictionary", "options", options, "dict", type(options).__name__)
        
        # Check specific option values
        max_records = options.get('max_records')
        if max_records is not None:
            if not isinstance(max_records, int):
                raise TypeValidationError("max_records must be an integer", "max_records", max_records, "int", type(max_records).__name__)
            if max_records <= 0:
                raise InputValidationError("max_records must be greater than 0", "max_records", max_records)
        
        timeout = options.get('timeout')
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                raise TypeValidationError("timeout must be a number", "timeout", timeout, "number", type(timeout).__name__)
            if timeout <= 0:
                raise InputValidationError("timeout must be greater than 0", "timeout", timeout)
    
    # Happy path: Return validated configuration
    return {
        'target': target.strip(),
        'enum_type': enum_type,
        'options': options or {},
        'timeout': (options or {}).get('timeout', 30.0),
        'max_records': (options or {}).get('max_records', 1000)
    }


def validate_attack_request_guard_clauses(
    target: str,
    attack_type: str,
    payload: Optional[str] = None,
    credentials: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Validate attack request using guard clauses - avoiding nested conditionals
    
    Args:
        target: Target to attack
        attack_type: Type of attack
        payload: Attack payload
        credentials: Credentials for attack
        
    Returns:
        Validated attack configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Guard clause 1: Check if target is provided
    if target is None:
        raise InputValidationError("Attack target is required", "target", target)
    
    # Guard clause 2: Check if target is a string
    if not isinstance(target, str):
        raise TypeValidationError("Target must be a string", "target", target, "str", type(target).__name__)
    
    # Guard clause 3: Check if target is empty
    if not target.strip():
        raise InputValidationError("Target cannot be empty", "target", target)
    
    # Guard clause 4: Check if attack_type is provided
    if attack_type is None:
        raise InputValidationError("Attack type is required", "attack_type", attack_type)
    
    # Guard clause 5: Check if attack_type is a string
    if not isinstance(attack_type, str):
        raise TypeValidationError("Attack type must be a string", "attack_type", attack_type, "str", type(attack_type).__name__)
    
    # Guard clause 6: Check if attack_type is empty
    if not attack_type.strip():
        raise InputValidationError("Attack type cannot be empty", "attack_type", attack_type)
    
    # Guard clause 7: Check if attack_type is valid
    valid_attack_types = ['brute_force', 'exploitation', 'social_engineering']
    if attack_type not in valid_attack_types:
        raise InputValidationError(
            f"Invalid attack type. Must be one of: {', '.join(valid_attack_types)}",
            "attack_type",
            attack_type
        )
    
    # Guard clause 8: Check if payload is valid (if provided)
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
    
    # Guard clause 9: Check if credentials are valid (if provided)
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
    
    # Happy path: Return validated configuration
    return {
        'target': target.strip(),
        'attack_type': attack_type,
        'payload': payload,
        'credentials': credentials or [],
        'timeout': 30.0,
        'max_attempts': 1000,
        'delay': 1.0
    }


# ============================================================================
# ASYNC GUARD CLAUSES
# ============================================================================

async def validate_async_scan_request_guard_clauses(
    target: Dict[str, Any],
    scan_type: str,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate async scan request using guard clauses - avoiding nested conditionals
    
    Args:
        target: Target to scan
        scan_type: Type of scan
        timeout: Timeout value
        
    Returns:
        Validated scan configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Guard clause 1: Check if target is provided
    if target is None:
        raise InputValidationError("Scan target is required", "target", target)
    
    # Guard clause 2: Check if target is a dictionary
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    # Guard clause 3: Check if target is empty
    if not target:
        raise InputValidationError("Target cannot be empty", "target", target)
    
    # Guard clause 4: Check if host is provided
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    # Guard clause 5: Check if host is empty
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    # Guard clause 6: Check if scan_type is provided
    if scan_type is None:
        raise InputValidationError("Scan type is required", "scan_type", scan_type)
    
    # Guard clause 7: Check if scan_type is empty
    if not isinstance(scan_type, str) or not scan_type.strip():
        raise InputValidationError("Scan type cannot be empty", "scan_type", scan_type)
    
    # Guard clause 8: Check if scan_type is valid
    valid_scan_types = ['port_scan', 'vulnerability_scan', 'web_scan']
    if scan_type not in valid_scan_types:
        raise InputValidationError(
            f"Invalid scan type. Must be one of: {', '.join(valid_scan_types)}",
            "scan_type",
            scan_type
        )
    
    # Guard clause 9: Check if timeout is valid (if provided)
    if timeout is not None:
        if not isinstance(timeout, (int, float)):
            raise TypeValidationError("Timeout must be a number", "timeout", timeout, "number", type(timeout).__name__)
        if timeout <= 0:
            raise InputValidationError("Timeout must be greater than 0", "timeout", timeout)
    
    # Guard clause 10: Check if target is reachable (async check)
    try:
        # Simulate async reachability check
        await asyncio.sleep(0.001)
        if host == "unreachable.example.com":
            raise InputValidationError("Target is not reachable", "host", host)
    except Exception as e:
        if isinstance(e, InputValidationError):
            raise
        raise InputValidationError(f"Failed to check target reachability: {str(e)}", "host", host)
    
    # Happy path: Return validated configuration
    return {
        'target': {
            'host': host.strip(),
            'port': target.get('port'),
            'protocol': target.get('protocol', 'tcp')
        },
        'scan_type': scan_type,
        'timeout': timeout or 30.0,
        'max_retries': 3,
        'user_agent': 'Video-OpusClip-Scanner/1.0'
    }


# ============================================================================
# SECURITY GUARD CLAUSES
# ============================================================================

def validate_authentication_guard_clauses(
    username: str,
    password: str,
    auth_method: str = "password"
) -> Dict[str, Any]:
    """
    Validate authentication using guard clauses - avoiding nested conditionals
    
    Args:
        username: Username
        password: Password
        auth_method: Authentication method
        
    Returns:
        Validated authentication configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Guard clause 1: Check if username is provided
    if username is None:
        raise InputValidationError("Username is required", "username", username)
    
    # Guard clause 2: Check if username is a string
    if not isinstance(username, str):
        raise TypeValidationError("Username must be a string", "username", username, "str", type(username).__name__)
    
    # Guard clause 3: Check if username is empty
    if not username.strip():
        raise InputValidationError("Username cannot be empty", "username", username)
    
    # Guard clause 4: Check if username is too short
    if len(username.strip()) < 3:
        raise InputValidationError("Username must be at least 3 characters long", "username", username)
    
    # Guard clause 5: Check if username is too long
    if len(username) > 50:
        raise InputValidationError("Username must be at most 50 characters long", "username", username)
    
    # Guard clause 6: Check username format
    import re
    username_pattern = r'^[a-zA-Z0-9_]+$'
    if not re.match(username_pattern, username):
        raise InputValidationError("Username can only contain letters, numbers, and underscores", "username", username)
    
    # Guard clause 7: Check if password is provided
    if password is None:
        raise InputValidationError("Password is required", "password", password)
    
    # Guard clause 8: Check if password is a string
    if not isinstance(password, str):
        raise TypeValidationError("Password must be a string", "password", password, "str", type(password).__name__)
    
    # Guard clause 9: Check if password is too short
    if len(password) < 8:
        raise InputValidationError("Password must be at least 8 characters long", "password", password)
    
    # Guard clause 10: Check if password is too long
    if len(password) > 128:
        raise InputValidationError("Password must be at most 128 characters long", "password", password)
    
    # Guard clause 11: Check password strength
    if not re.search(r'[A-Z]', password):
        raise InputValidationError("Password must contain at least one uppercase letter", "password", password)
    
    if not re.search(r'[a-z]', password):
        raise InputValidationError("Password must contain at least one lowercase letter", "password", password)
    
    if not re.search(r'\d', password):
        raise InputValidationError("Password must contain at least one digit", "password", password)
    
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password):
        raise InputValidationError("Password must contain at least one special character", "password", password)
    
    # Guard clause 12: Check if auth_method is valid
    valid_methods = ['password', 'token', 'oauth', 'saml']
    if auth_method not in valid_methods:
        raise InputValidationError(f"Invalid auth method. Must be one of: {', '.join(valid_methods)}", "auth_method", auth_method)
    
    # Happy path: Return validated configuration
    return {
        'username': username.strip(),
        'password': password,
        'auth_method': auth_method,
        'timestamp': datetime.utcnow(),
        'max_attempts': 3,
        'lockout_duration': 300  # 5 minutes
    }


def validate_authorization_guard_clauses(
    user_id: str,
    action: str,
    resource: str,
    permissions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate authorization using guard clauses - avoiding nested conditionals
    
    Args:
        user_id: User identifier
        action: Action to perform
        resource: Resource to access
        permissions: User permissions
        
    Returns:
        Validated authorization configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Guard clause 1: Check if user_id is provided
    if user_id is None:
        raise InputValidationError("User ID is required", "user_id", user_id)
    
    # Guard clause 2: Check if user_id is a string
    if not isinstance(user_id, str):
        raise TypeValidationError("User ID must be a string", "user_id", user_id, "str", type(user_id).__name__)
    
    # Guard clause 3: Check if user_id is empty
    if not user_id.strip():
        raise InputValidationError("User ID cannot be empty", "user_id", user_id)
    
    # Guard clause 4: Check if action is provided
    if action is None:
        raise InputValidationError("Action is required", "action", action)
    
    # Guard clause 5: Check if action is a string
    if not isinstance(action, str):
        raise TypeValidationError("Action must be a string", "action", action, "str", type(action).__name__)
    
    # Guard clause 6: Check if action is empty
    if not action.strip():
        raise InputValidationError("Action cannot be empty", "action", action)
    
    # Guard clause 7: Check if action is valid
    valid_actions = ['read', 'write', 'delete', 'execute']
    if action not in valid_actions:
        raise InputValidationError(f"Invalid action. Must be one of: {', '.join(valid_actions)}", "action", action)
    
    # Guard clause 8: Check if resource is provided
    if resource is None:
        raise InputValidationError("Resource is required", "resource", resource)
    
    # Guard clause 9: Check if resource is a string
    if not isinstance(resource, str):
        raise TypeValidationError("Resource must be a string", "resource", resource, "str", type(resource).__name__)
    
    # Guard clause 10: Check if resource is empty
    if not resource.strip():
        raise InputValidationError("Resource cannot be empty", "resource", resource)
    
    # Guard clause 11: Check if permissions are valid (if provided)
    if permissions is not None:
        if not isinstance(permissions, list):
            raise TypeValidationError("Permissions must be a list", "permissions", permissions, "list", type(permissions).__name__)
        
        for i, permission in enumerate(permissions):
            if not isinstance(permission, str):
                raise TypeValidationError(f"Permission {i} must be a string", f"permission_{i}", permission, "str", type(permission).__name__)
            
            if not permission.strip():
                raise InputValidationError(f"Permission {i} cannot be empty", f"permission_{i}", permission)
    
    # Guard clause 12: Check if user is blocked
    if user_id == "blocked_user":
        raise AuthorizationError("User is blocked", "user_id", user_id)
    
    # Guard clause 13: Check if resource is protected
    protected_resources = ['/admin', '/system', '/config']
    if resource in protected_resources:
        raise AuthorizationError(f"Resource '{resource}' is protected", "resource", resource)
    
    # Happy path: Return validated configuration
    return {
        'user_id': user_id.strip(),
        'action': action.strip(),
        'resource': resource.strip(),
        'permissions': permissions or [],
        'timestamp': datetime.utcnow(),
        'session_timeout': 3600  # 1 hour
    }


# ============================================================================
# DATABASE GUARD CLAUSES
# ============================================================================

def validate_database_connection_guard_clauses(
    host: str,
    port: int,
    database: str,
    username: str,
    password: str
) -> Dict[str, Any]:
    """
    Validate database connection using guard clauses - avoiding nested conditionals
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        username: Database username
        password: Database password
        
    Returns:
        Validated connection configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Guard clause 1: Check if host is provided
    if host is None:
        raise InputValidationError("Database host is required", "host", host)
    
    # Guard clause 2: Check if host is a string
    if not isinstance(host, str):
        raise TypeValidationError("Host must be a string", "host", host, "str", type(host).__name__)
    
    # Guard clause 3: Check if host is empty
    if not host.strip():
        raise InputValidationError("Host cannot be empty", "host", host)
    
    # Guard clause 4: Check if host is valid IP or domain
    import re
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
    
    if not (re.match(ip_pattern, host) or re.match(domain_pattern, host)):
        raise InputValidationError("Host must be a valid IP address or domain name", "host", host)
    
    # Guard clause 5: Check if port is provided
    if port is None:
        raise InputValidationError("Database port is required", "port", port)
    
    # Guard clause 6: Check if port is an integer
    if not isinstance(port, int):
        raise TypeValidationError("Port must be an integer", "port", port, "int", type(port).__name__)
    
    # Guard clause 7: Check if port is in valid range
    if port < 1 or port > 65535:
        raise InputValidationError("Port must be between 1 and 65535", "port", port)
    
    # Guard clause 8: Check if database is provided
    if database is None:
        raise InputValidationError("Database name is required", "database", database)
    
    # Guard clause 9: Check if database is a string
    if not isinstance(database, str):
        raise TypeValidationError("Database name must be a string", "database", database, "str", type(database).__name__)
    
    # Guard clause 10: Check if database is empty
    if not database.strip():
        raise InputValidationError("Database name cannot be empty", "database", database)
    
    # Guard clause 11: Check database name format
    db_pattern = r'^[a-zA-Z0-9_]+$'
    if not re.match(db_pattern, database):
        raise InputValidationError("Database name can only contain letters, numbers, and underscores", "database", database)
    
    # Guard clause 12: Check if username is provided
    if username is None:
        raise InputValidationError("Database username is required", "username", username)
    
    # Guard clause 13: Check if username is a string
    if not isinstance(username, str):
        raise TypeValidationError("Username must be a string", "username", username, "str", type(username).__name__)
    
    # Guard clause 14: Check if username is empty
    if not username.strip():
        raise InputValidationError("Username cannot be empty", "username", username)
    
    # Guard clause 15: Check if password is provided
    if password is None:
        raise InputValidationError("Database password is required", "password", password)
    
    # Guard clause 16: Check if password is a string
    if not isinstance(password, str):
        raise TypeValidationError("Password must be a string", "password", password, "str", type(password).__name__)
    
    # Guard clause 17: Check if password is empty
    if not password:
        raise InputValidationError("Password cannot be empty", "password", password)
    
    # Happy path: Return validated configuration
    return {
        'host': host.strip(),
        'port': port,
        'database': database.strip(),
        'username': username.strip(),
        'password': password,
        'ssl_mode': 'prefer',
        'max_connections': 10,
        'connection_timeout': 30.0,
        'query_timeout': 60.0
    }


def validate_database_query_guard_clauses(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    operation: str = "select"
) -> Dict[str, Any]:
    """
    Validate database query using guard clauses - avoiding nested conditionals
    
    Args:
        query: SQL query
        parameters: Query parameters
        operation: Database operation
        
    Returns:
        Validated query configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Guard clause 1: Check if query is provided
    if query is None:
        raise InputValidationError("SQL query is required", "query", query)
    
    # Guard clause 2: Check if query is a string
    if not isinstance(query, str):
        raise TypeValidationError("Query must be a string", "query", query, "str", type(query).__name__)
    
    # Guard clause 3: Check if query is empty
    if not query.strip():
        raise InputValidationError("Query cannot be empty", "query", query)
    
    # Guard clause 4: Check if query is too long
    if len(query) > 10000:
        raise InputValidationError("Query is too long (maximum 10000 characters)", "query", query)
    
    # Guard clause 5: Check if operation is provided
    if operation is None:
        raise InputValidationError("Database operation is required", "operation", operation)
    
    # Guard clause 6: Check if operation is a string
    if not isinstance(operation, str):
        raise TypeValidationError("Operation must be a string", "operation", operation, "str", type(operation).__name__)
    
    # Guard clause 7: Check if operation is empty
    if not operation.strip():
        raise InputValidationError("Operation cannot be empty", "operation", operation)
    
    # Guard clause 8: Check if operation is valid
    valid_operations = ['select', 'insert', 'update', 'delete', 'create', 'drop']
    if operation not in valid_operations:
        raise InputValidationError(f"Invalid operation. Must be one of: {', '.join(valid_operations)}", "operation", operation)
    
    # Guard clause 9: Check for dangerous SQL patterns
    dangerous_patterns = [
        r';\s*$',  # Multiple statements
        r'--',     # SQL comments
        r'/\*.*\*/',  # Block comments
        r'xp_',    # Extended procedures
        r'sp_',    # Stored procedures
        r'exec\s+',  # EXEC statements
        r'execute\s+',  # EXECUTE statements
        r'union\s+select',  # UNION attacks
        r'drop\s+table',  # DROP TABLE
        r'drop\s+database',  # DROP DATABASE
        r'delete\s+from\s+\w+\s*$',  # DELETE without WHERE
        r'truncate\s+table'  # TRUNCATE
    ]
    
    query_lower = query.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, query_lower):
            raise SecurityError(f"Dangerous SQL pattern detected: {pattern}", "query", query)
    
    # Guard clause 10: Check if parameters are valid (if provided)
    if parameters is not None:
        if not isinstance(parameters, dict):
            raise TypeValidationError("Parameters must be a dictionary", "parameters", parameters, "dict", type(parameters).__name__)
        
        for key, value in parameters.items():
            if not isinstance(key, str):
                raise TypeValidationError(f"Parameter key '{key}' must be a string", f"param_key_{key}", key, "str", type(key).__name__)
            
            if not key.strip():
                raise InputValidationError(f"Parameter key '{key}' cannot be empty", f"param_key_{key}", key)
            
            # Check for SQL injection patterns in parameter values
            if isinstance(value, str):
                sql_patterns = ["'", '"', ';', '--', '/*', '*/']
                for pattern in sql_patterns:
                    if pattern in value:
                        raise SecurityError(f"SQL injection pattern detected in parameter '{key}': {pattern}", f"param_value_{key}", value)
    
    # Happy path: Return validated configuration
    return {
        'query': query.strip(),
        'parameters': parameters or {},
        'operation': operation,
        'timeout': 30.0,
        'max_rows': 10000,
        'read_only': operation == 'select'
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of advanced guard clauses
    print("🛡️ Advanced Guard Clauses Example")
    
    # Test scan request validation
    try:
        scan_config = validate_scan_request_guard_clauses(
            target={'host': '192.168.1.100', 'port': 80},
            scan_type='port_scan',
            timeout=30.0
        )
        print(f"✅ Scan config validation successful: {scan_config}")
    except Exception as e:
        print(f"❌ Scan config validation failed: {e}")
    
    # Test enumeration request validation
    try:
        enum_config = validate_enumeration_request_guard_clauses(
            target='example.com',
            enum_type='dns',
            options={'max_records': 100, 'timeout': 30}
        )
        print(f"✅ Enumeration config validation successful: {enum_config}")
    except Exception as e:
        print(f"❌ Enumeration config validation failed: {e}")
    
    # Test attack request validation
    try:
        attack_config = validate_attack_request_guard_clauses(
            target='192.168.1.100',
            attack_type='brute_force',
            credentials=[{'username': 'admin', 'password': 'password'}]
        )
        print(f"✅ Attack config validation successful: {attack_config}")
    except Exception as e:
        print(f"❌ Attack config validation failed: {e}")
    
    # Test authentication validation
    try:
        auth_config = validate_authentication_guard_clauses(
            username='john_doe',
            password='SecureP@ss123'
        )
        print(f"✅ Authentication config validation successful: {auth_config}")
    except Exception as e:
        print(f"❌ Authentication config validation failed: {e}")
    
    # Test authorization validation
    try:
        authz_config = validate_authorization_guard_clauses(
            user_id='user123',
            action='read',
            resource='/data',
            permissions=['read_data']
        )
        print(f"✅ Authorization config validation successful: {authz_config}")
    except Exception as e:
        print(f"❌ Authorization config validation failed: {e}")
    
    # Test database connection validation
    try:
        db_config = validate_database_connection_guard_clauses(
            host='localhost',
            port=5432,
            database='mydb',
            username='user',
            password='secret123'
        )
        print(f"✅ Database connection config validation successful: {db_config}")
    except Exception as e:
        print(f"❌ Database connection config validation failed: {e}")
    
    # Test database query validation
    try:
        query_config = validate_database_query_guard_clauses(
            query='SELECT * FROM users WHERE id = %s',
            parameters={'id': 1},
            operation='select'
        )
        print(f"✅ Database query config validation successful: {query_config}")
    except Exception as e:
        print(f"❌ Database query config validation failed: {e}")
    
    print("\n✅ Advanced guard clauses examples completed!") 