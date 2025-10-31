#!/usr/bin/env python3
"""
Guard Clauses and Early Returns for Video-OpusClip
Implements defensive programming patterns with early validation and returns
"""

import re
import ipaddress
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from urllib.parse import urlparse

from .custom_exceptions import (
    ValidationError, InputValidationError, TypeValidationError,
    SecurityError, AuthenticationError, AuthorizationError
)


# ============================================================================
# GUARD CLAUSE UTILITIES
# ============================================================================

def is_none_or_empty(value: Any) -> bool:
    """Check if value is None or empty"""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict, tuple)) and not value:
        return True
    return False


def is_valid_string(value: Any, min_length: int = 1, max_length: Optional[int] = None) -> bool:
    """Check if value is a valid string"""
    if not isinstance(value, str):
        return False
    if len(value.strip()) < min_length:
        return False
    if max_length and len(value) > max_length:
        return False
    return True


def is_valid_integer(value: Any, min_value: Optional[int] = None, max_value: Optional[int] = None) -> bool:
    """Check if value is a valid integer"""
    try:
        int_val = int(value)
        if min_value is not None and int_val < min_value:
            return False
        if max_value is not None and int_val > max_value:
            return False
        return True
    except (ValueError, TypeError):
        return False


def is_valid_ip_address(value: Any) -> bool:
    """Check if value is a valid IP address"""
    if not isinstance(value, str):
        return False
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def is_valid_domain(value: Any) -> bool:
    """Check if value is a valid domain name"""
    if not isinstance(value, str):
        return False
    pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
    return bool(re.match(pattern, value))


def is_valid_url(value: Any) -> bool:
    """Check if value is a valid URL"""
    if not isinstance(value, str):
        return False
    try:
        parsed = urlparse(value)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def is_valid_port(value: Any) -> bool:
    """Check if value is a valid port number"""
    if not is_valid_integer(value, 1, 65535):
        return False
    return True


# ============================================================================
# SCANNING GUARD CLAUSES
# ============================================================================

def validate_scan_target_early(target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Early validation for scan targets with guard clauses
    
    Args:
        target: Target dictionary to validate
        
    Returns:
        Validated target or None if invalid
    """
    # Guard clause: Check if target is provided
    if is_none_or_empty(target):
        return None
    
    # Guard clause: Check if target is a dictionary
    if not isinstance(target, dict):
        return None
    
    # Guard clause: Check if host is provided
    host = target.get('host')
    if is_none_or_empty(host):
        return None
    
    # Guard clause: Check if host is a valid string
    if not is_valid_string(host, 1, 255):
        return None
    
    # Guard clause: Check if host is valid IP or domain
    if not (is_valid_ip_address(host) or is_valid_domain(host)):
        return None
    
    # Guard clause: Check port if provided
    port = target.get('port')
    if port is not None and not is_valid_port(port):
        return None
    
    # Guard clause: Check protocol if provided
    protocol = target.get('protocol', 'tcp')
    if not isinstance(protocol, str) or protocol.lower() not in ['tcp', 'udp']:
        return None
    
    return {
        'host': host.strip(),
        'port': port,
        'protocol': protocol.lower()
    }


def validate_scan_configuration_early(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Early validation for scan configuration with guard clauses
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated configuration or None if invalid
    """
    # Guard clause: Check if config is provided
    if is_none_or_empty(config):
        return None
    
    # Guard clause: Check if config is a dictionary
    if not isinstance(config, dict):
        return None
    
    # Guard clause: Check scan type
    scan_type = config.get('scan_type')
    if not is_valid_string(scan_type, 1, 50):
        return None
    
    valid_scan_types = ['port_scan', 'vulnerability_scan', 'web_scan', 'network_scan']
    if scan_type not in valid_scan_types:
        return None
    
    # Guard clause: Check targets
    targets = config.get('targets')
    if is_none_or_empty(targets) or not isinstance(targets, list):
        return None
    
    # Guard clause: Check each target
    validated_targets = []
    for target in targets:
        validated_target = validate_scan_target_early(target)
        if validated_target is None:
            return None
        validated_targets.append(validated_target)
    
    # Guard clause: Check timeout
    timeout = config.get('timeout', 30.0)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        return None
    
    # Guard clause: Check max concurrent
    max_concurrent = config.get('max_concurrent', 10)
    if not is_valid_integer(max_concurrent, 1, 100):
        return None
    
    return {
        'scan_type': scan_type,
        'targets': validated_targets,
        'timeout': float(timeout),
        'max_concurrent': int(max_concurrent),
        'retry_count': config.get('retry_count', 3),
        'custom_headers': config.get('custom_headers', {}),
        'user_agent': config.get('user_agent', 'Video-OpusClip-Scanner/1.0'),
        'verify_ssl': config.get('verify_ssl', True)
    }


# ============================================================================
# ENUMERATION GUARD CLAUSES
# ============================================================================

def validate_dns_enumeration_early(domain: str, record_types: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Early validation for DNS enumeration with guard clauses
    
    Args:
        domain: Domain to enumerate
        record_types: Types of DNS records to query
        
    Returns:
        Validated enumeration config or None if invalid
    """
    # Guard clause: Check if domain is provided
    if is_none_or_empty(domain):
        return None
    
    # Guard clause: Check if domain is a valid string
    if not is_valid_string(domain, 1, 253):
        return None
    
    # Guard clause: Check if domain is valid
    if not is_valid_domain(domain):
        return None
    
    # Guard clause: Check record types
    if record_types is not None:
        if not isinstance(record_types, list):
            return None
        
        valid_record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'PTR', 'SOA', 'SRV']
        for record_type in record_types:
            if not isinstance(record_type, str) or record_type.upper() not in valid_record_types:
                return None
    
    return {
        'domain': domain.lower().strip(),
        'record_types': record_types or ['A', 'AAAA', 'MX', 'NS', 'TXT'],
        'timeout': 30.0,
        'max_retries': 3
    }


def validate_smb_enumeration_early(target: str, credentials: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """
    Early validation for SMB enumeration with guard clauses
    
    Args:
        target: Target host
        credentials: SMB credentials
        
    Returns:
        Validated enumeration config or None if invalid
    """
    # Guard clause: Check if target is provided
    if is_none_or_empty(target):
        return None
    
    # Guard clause: Check if target is valid IP or domain
    if not (is_valid_ip_address(target) or is_valid_domain(target)):
        return None
    
    # Guard clause: Check credentials if provided
    if credentials is not None:
        if not isinstance(credentials, dict):
            return None
        
        username = credentials.get('username')
        password = credentials.get('password')
        
        if username is not None and not is_valid_string(username, 1, 50):
            return None
        
        if password is not None and not is_valid_string(password, 0, 128):
            return None
    
    return {
        'target': target.strip(),
        'credentials': credentials,
        'timeout': 30.0,
        'max_retries': 3,
        'anonymous': credentials is None
    }


# ============================================================================
# ATTACK GUARD CLAUSES
# ============================================================================

def validate_brute_force_attack_early(
    target: str,
    service: str,
    credentials: Optional[List[Dict[str, str]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Early validation for brute force attacks with guard clauses
    
    Args:
        target: Target host
        service: Service to attack
        credentials: Credentials to test
        
    Returns:
        Validated attack config or None if invalid
    """
    # Guard clause: Check if target is provided
    if is_none_or_empty(target):
        return None
    
    # Guard clause: Check if target is valid
    if not (is_valid_ip_address(target) or is_valid_domain(target)):
        return None
    
    # Guard clause: Check if service is provided
    if not is_valid_string(service, 1, 20):
        return None
    
    # Guard clause: Check if service is supported
    supported_services = ['ssh', 'ftp', 'http', 'https', 'smb', 'smtp', 'pop3', 'imap']
    if service.lower() not in supported_services:
        return None
    
    # Guard clause: Check credentials if provided
    if credentials is not None:
        if not isinstance(credentials, list):
            return None
        
        for cred in credentials:
            if not isinstance(cred, dict):
                return None
            
            username = cred.get('username')
            password = cred.get('password')
            
            if username is not None and not is_valid_string(username, 1, 50):
                return None
            
            if password is not None and not is_valid_string(password, 0, 128):
                return None
    
    return {
        'target': target.strip(),
        'service': service.lower(),
        'credentials': credentials or [],
        'timeout': 30.0,
        'max_attempts': 1000,
        'delay': 1.0
    }


def validate_exploitation_attack_early(
    target: str,
    exploit_type: str,
    payload: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Early validation for exploitation attacks with guard clauses
    
    Args:
        target: Target host
        exploit_type: Type of exploit
        payload: Exploit payload
        
    Returns:
        Validated attack config or None if invalid
    """
    # Guard clause: Check if target is provided
    if is_none_or_empty(target):
        return None
    
    # Guard clause: Check if target is valid
    if not (is_valid_ip_address(target) or is_valid_domain(target)):
        return None
    
    # Guard clause: Check if exploit type is provided
    if not is_valid_string(exploit_type, 1, 50):
        return None
    
    # Guard clause: Check if exploit type is supported
    supported_exploits = ['sql_injection', 'xss', 'command_injection', 'path_traversal', 'file_inclusion']
    if exploit_type.lower() not in supported_exploits:
        return None
    
    # Guard clause: Check payload if provided
    if payload is not None and not is_valid_string(payload, 1, 1000):
        return None
    
    return {
        'target': target.strip(),
        'exploit_type': exploit_type.lower(),
        'payload': payload,
        'timeout': 30.0,
        'max_retries': 3
    }


# ============================================================================
# SECURITY GUARD CLAUSES
# ============================================================================

def validate_authentication_early(
    username: str,
    password: str,
    auth_method: str = "password"
) -> Optional[Dict[str, Any]]:
    """
    Early validation for authentication with guard clauses
    
    Args:
        username: Username
        password: Password
        auth_method: Authentication method
        
    Returns:
        Validated auth config or None if invalid
    """
    # Guard clause: Check if username is provided
    if is_none_or_empty(username):
        return None
    
    # Guard clause: Check if username is valid
    if not is_valid_string(username, 3, 50):
        return None
    
    # Guard clause: Check username format
    username_pattern = r'^[a-zA-Z0-9_]+$'
    if not re.match(username_pattern, username):
        return None
    
    # Guard clause: Check if password is provided
    if is_none_or_empty(password):
        return None
    
    # Guard clause: Check if password is valid
    if not is_valid_string(password, 8, 128):
        return None
    
    # Guard clause: Check password strength
    if not re.search(r'[A-Z]', password):
        return None
    if not re.search(r'[a-z]', password):
        return None
    if not re.search(r'\d', password):
        return None
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password):
        return None
    
    # Guard clause: Check auth method
    valid_methods = ['password', 'token', 'oauth', 'saml']
    if auth_method not in valid_methods:
        return None
    
    return {
        'username': username.strip(),
        'password': password,
        'auth_method': auth_method,
        'timestamp': datetime.utcnow()
    }


def validate_authorization_early(
    user_id: str,
    resource: str,
    action: str,
    permissions: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Early validation for authorization with guard clauses
    
    Args:
        user_id: User identifier
        resource: Resource to access
        action: Action to perform
        permissions: User permissions
        
    Returns:
        Validated auth config or None if invalid
    """
    # Guard clause: Check if user_id is provided
    if is_none_or_empty(user_id):
        return None
    
    # Guard clause: Check if user_id is valid
    if not is_valid_string(user_id, 1, 100):
        return None
    
    # Guard clause: Check if resource is provided
    if is_none_or_empty(resource):
        return None
    
    # Guard clause: Check if resource is valid
    if not is_valid_string(resource, 1, 200):
        return None
    
    # Guard clause: Check if action is provided
    if is_none_or_empty(action):
        return None
    
    # Guard clause: Check if action is valid
    if not is_valid_string(action, 1, 50):
        return None
    
    # Guard clause: Check permissions if provided
    if permissions is not None:
        if not isinstance(permissions, list):
            return None
        
        for permission in permissions:
            if not is_valid_string(permission, 1, 50):
                return None
    
    return {
        'user_id': user_id.strip(),
        'resource': resource.strip(),
        'action': action.strip(),
        'permissions': permissions or [],
        'timestamp': datetime.utcnow()
    }


# ============================================================================
# DATABASE GUARD CLAUSES
# ============================================================================

def validate_database_connection_early(
    host: str,
    port: int,
    database: str,
    username: str,
    password: str
) -> Optional[Dict[str, Any]]:
    """
    Early validation for database connection with guard clauses
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        username: Database username
        password: Database password
        
    Returns:
        Validated connection config or None if invalid
    """
    # Guard clause: Check if host is provided
    if is_none_or_empty(host):
        return None
    
    # Guard clause: Check if host is valid
    if not (is_valid_ip_address(host) or is_valid_domain(host)):
        return None
    
    # Guard clause: Check if port is valid
    if not is_valid_port(port):
        return None
    
    # Guard clause: Check if database is provided
    if is_none_or_empty(database):
        return None
    
    # Guard clause: Check if database name is valid
    if not is_valid_string(database, 1, 64):
        return None
    
    # Guard clause: Check database name format
    db_pattern = r'^[a-zA-Z0-9_]+$'
    if not re.match(db_pattern, database):
        return None
    
    # Guard clause: Check if username is provided
    if is_none_or_empty(username):
        return None
    
    # Guard clause: Check if username is valid
    if not is_valid_string(username, 1, 32):
        return None
    
    # Guard clause: Check if password is provided
    if is_none_or_empty(password):
        return None
    
    # Guard clause: Check if password is valid
    if not is_valid_string(password, 1, 128):
        return None
    
    return {
        'host': host.strip(),
        'port': int(port),
        'database': database.strip(),
        'username': username.strip(),
        'password': password,
        'ssl_mode': 'prefer',
        'max_connections': 10,
        'connection_timeout': 30.0
    }


def validate_database_query_early(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    operation: str = "select"
) -> Optional[Dict[str, Any]]:
    """
    Early validation for database queries with guard clauses
    
    Args:
        query: SQL query
        parameters: Query parameters
        operation: Database operation type
        
    Returns:
        Validated query config or None if invalid
    """
    # Guard clause: Check if query is provided
    if is_none_or_empty(query):
        return None
    
    # Guard clause: Check if query is valid string
    if not is_valid_string(query, 1, 10000):
        return None
    
    # Guard clause: Check for dangerous SQL patterns
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
            return None
    
    # Guard clause: Check operation type
    valid_operations = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']
    if operation.lower() not in valid_operations:
        return None
    
    # Guard clause: Check parameters if provided
    if parameters is not None:
        if not isinstance(parameters, dict):
            return None
        
        for key, value in parameters.items():
            if not is_valid_string(key, 1, 50):
                return None
    
    return {
        'query': query.strip(),
        'parameters': parameters or {},
        'operation': operation.lower(),
        'timeout': 30.0
    }


# ============================================================================
# FILE SYSTEM GUARD CLAUSES
# ============================================================================

def validate_file_operation_early(
    file_path: str,
    operation: str = "read",
    max_size: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Early validation for file operations with guard clauses
    
    Args:
        file_path: File path
        operation: File operation type
        max_size: Maximum file size
        
    Returns:
        Validated file config or None if invalid
    """
    # Guard clause: Check if file_path is provided
    if is_none_or_empty(file_path):
        return None
    
    # Guard clause: Check if file_path is valid string
    if not is_valid_string(file_path, 1, 500):
        return None
    
    # Guard clause: Check for path traversal attempts
    dangerous_patterns = [
        r'\.\./',  # Directory traversal
        r'\.\.\\',  # Windows directory traversal
        r'//',     # Absolute path
        r'\\',     # Windows absolute path
        r'~',      # Home directory
        r'%',      # Environment variables
        r'<',      # HTML injection
        r'>',      # HTML injection
        r'|',      # Command injection
        r'&',      # Command injection
        r';',      # Command injection
        r'`',      # Command injection
        r'$',      # Variable substitution
    ]
    
    for pattern in dangerous_patterns:
        if pattern in file_path:
            return None
    
    # Guard clause: Check operation type
    valid_operations = ['read', 'write', 'delete', 'create', 'append']
    if operation.lower() not in valid_operations:
        return None
    
    # Guard clause: Check max_size if provided
    if max_size is not None and not is_valid_integer(max_size, 1):
        return None
    
    return {
        'file_path': file_path.strip(),
        'operation': operation.lower(),
        'max_size': max_size,
        'encoding': 'utf-8'
    }


# ============================================================================
# NETWORK GUARD CLAUSES
# ============================================================================

def validate_network_request_early(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Early validation for network requests with guard clauses
    
    Args:
        url: Request URL
        method: HTTP method
        headers: Request headers
        data: Request data
        
    Returns:
        Validated request config or None if invalid
    """
    # Guard clause: Check if URL is provided
    if is_none_or_empty(url):
        return None
    
    # Guard clause: Check if URL is valid
    if not is_valid_url(url):
        return None
    
    # Guard clause: Check URL scheme
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ['http', 'https']:
        return None
    
    # Guard clause: Check HTTP method
    valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
    if method.upper() not in valid_methods:
        return None
    
    # Guard clause: Check headers if provided
    if headers is not None:
        if not isinstance(headers, dict):
            return None
        
        for key, value in headers.items():
            if not is_valid_string(key, 1, 100):
                return None
            if not is_valid_string(value, 0, 1000):
                return None
    
    # Guard clause: Check data if provided
    if data is not None:
        if not isinstance(data, dict):
            return None
        
        for key, value in data.items():
            if not is_valid_string(key, 1, 100):
                return None
    
    return {
        'url': url.strip(),
        'method': method.upper(),
        'headers': headers or {},
        'data': data or {},
        'timeout': 30.0,
        'verify_ssl': True,
        'follow_redirects': True,
        'max_redirects': 5
    }


# ============================================================================
# CONFIGURATION GUARD CLAUSES
# ============================================================================

def validate_configuration_early(
    config: Dict[str, Any],
    required_keys: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Early validation for configuration with guard clauses
    
    Args:
        config: Configuration dictionary
        required_keys: Required configuration keys
        
    Returns:
        Validated configuration or None if invalid
    """
    # Guard clause: Check if config is provided
    if is_none_or_empty(config):
        return None
    
    # Guard clause: Check if config is a dictionary
    if not isinstance(config, dict):
        return None
    
    # Guard clause: Check required keys
    if required_keys:
        for key in required_keys:
            if key not in config or is_none_or_empty(config[key]):
                return None
    
    # Guard clause: Check for sensitive keys and validate them
    sensitive_keys = ['password', 'secret', 'key', 'token', 'api_key']
    for key in sensitive_keys:
        if key in config:
            value = config[key]
            if not is_none_or_empty(value) and not is_valid_string(value, 1, 1000):
                return None
    
    return config


# ============================================================================
# GUARD CLAUSE DECORATORS
# ============================================================================

def with_guard_clauses(
    validation_func: Callable,
    error_handler: Optional[Callable] = None
) -> Callable:
    """
    Decorator to add guard clauses to functions
    
    Args:
        validation_func: Function to validate input
        error_handler: Custom error handler
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Validate input using guard clauses
            validation_result = validation_func(*args, **kwargs)
            
            if validation_result is None:
                if error_handler:
                    return error_handler("Input validation failed")
                else:
                    raise InputValidationError("Input validation failed", "input", args)
            
            # Call original function with validated input
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of guard clauses
    print("🛡️ Guard Clauses Example")
    
    # Test scan target validation
    valid_target = validate_scan_target_early({
        'host': '192.168.1.100',
        'port': 80,
        'protocol': 'tcp'
    })
    print(f"Valid scan target: {valid_target}")
    
    invalid_target = validate_scan_target_early({
        'host': '',  # Invalid empty host
        'port': 99999  # Invalid port
    })
    print(f"Invalid scan target: {invalid_target}")
    
    # Test authentication validation
    valid_auth = validate_authentication_early(
        username="john_doe",
        password="SecureP@ss123"
    )
    print(f"Valid authentication: {valid_auth is not None}")
    
    invalid_auth = validate_authentication_early(
        username="",  # Invalid empty username
        password="weak"  # Invalid weak password
    )
    print(f"Invalid authentication: {invalid_auth is None}")
    
    # Test database query validation
    valid_query = validate_database_query_early(
        query="SELECT * FROM users WHERE id = %s",
        parameters={'id': 1},
        operation="select"
    )
    print(f"Valid database query: {valid_query is not None}")
    
    dangerous_query = validate_database_query_early(
        query="DROP TABLE users;",  # Dangerous query
        operation="drop"
    )
    print(f"Dangerous database query: {dangerous_query is None}")
    
    # Test file operation validation
    valid_file = validate_file_operation_early(
        file_path="data/config.json",
        operation="read"
    )
    print(f"Valid file operation: {valid_file is not None}")
    
    dangerous_file = validate_file_operation_early(
        file_path="../../../etc/passwd",  # Path traversal attempt
        operation="read"
    )
    print(f"Dangerous file operation: {dangerous_file is None}")
    
    # Test network request validation
    valid_request = validate_network_request_early(
        url="https://api.example.com/data",
        method="GET"
    )
    print(f"Valid network request: {valid_request is not None}")
    
    invalid_request = validate_network_request_early(
        url="not-a-url",  # Invalid URL
        method="INVALID"  # Invalid method
    )
    print(f"Invalid network request: {invalid_request is None}")
    
    # Test configuration validation
    valid_config = validate_configuration_early({
        'host': 'localhost',
        'port': 5432,
        'database': 'mydb',
        'username': 'user',
        'password': 'secret123'
    }, required_keys=['host', 'port', 'database'])
    print(f"Valid configuration: {valid_config is not None}")
    
    invalid_config = validate_configuration_early({
        'host': '',  # Empty required field
        'port': 'not-a-number'  # Invalid type
    }, required_keys=['host', 'port'])
    print(f"Invalid configuration: {invalid_config is None}") 