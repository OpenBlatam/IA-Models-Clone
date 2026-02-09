#!/usr/bin/env python3
"""
Early Return Patterns for Video-OpusClip
Implements early return strategies for invalid inputs and edge cases
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta

from .custom_exceptions import (
    ValidationError, InputValidationError, TypeValidationError,
    SecurityError, AuthenticationError, AuthorizationError
)


# ============================================================================
# EARLY RETURN UTILITIES
# ============================================================================

def early_return_if_none(value: Any, error_message: str = "Value is required") -> None:
    """
    Early return if value is None
    
    Args:
        value: Value to check
        error_message: Error message to raise
        
    Raises:
        InputValidationError: If value is None
    """
    if value is None:
        raise InputValidationError(error_message, "value", value)


def early_return_if_empty(value: Any, error_message: str = "Value cannot be empty") -> None:
    """
    Early return if value is empty
    
    Args:
        value: Value to check
        error_message: Error message to raise
        
    Raises:
        InputValidationError: If value is empty
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise InputValidationError(error_message, "value", value)


def early_return_if_invalid_type(value: Any, expected_type: type, error_message: str = None) -> None:
    """
    Early return if value is not of expected type
    
    Args:
        value: Value to check
        expected_type: Expected type
        error_message: Error message to raise
        
    Raises:
        TypeValidationError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        if error_message is None:
            error_message = f"Expected {expected_type.__name__}, got {type(value).__name__}"
        raise TypeValidationError(error_message, "value", value, expected_type.__name__, type(value).__name__)


def early_return_if_condition(condition: bool, error_message: str, field: str = "value", value: Any = None) -> None:
    """
    Early return if condition is True
    
    Args:
        condition: Condition to check
        error_message: Error message to raise
        field: Field name for error
        value: Value for error
        
    Raises:
        InputValidationError: If condition is True
    """
    if condition:
        raise InputValidationError(error_message, field, value)


# ============================================================================
# SCANNING EARLY RETURNS
# ============================================================================

def validate_scan_inputs_early_return(
    target: Dict[str, Any],
    scan_type: str,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate scan inputs with early returns
    
    Args:
        target: Target to scan
        scan_type: Type of scan
        timeout: Timeout value
        
    Returns:
        Validated scan configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Early return if target is None
    early_return_if_none(target, "Scan target is required")
    
    # Early return if target is not a dictionary
    early_return_if_invalid_type(target, dict, "Target must be a dictionary")
    
    # Early return if target is empty
    early_return_if_empty(target, "Target cannot be empty")
    
    # Early return if host is missing
    host = target.get('host')
    early_return_if_none(host, "Target host is required")
    early_return_if_empty(host, "Target host cannot be empty")
    
    # Early return if scan_type is invalid
    early_return_if_none(scan_type, "Scan type is required")
    early_return_if_empty(scan_type, "Scan type cannot be empty")
    
    valid_scan_types = ['port_scan', 'vulnerability_scan', 'web_scan', 'network_scan']
    early_return_if_condition(
        scan_type not in valid_scan_types,
        f"Invalid scan type. Must be one of: {', '.join(valid_scan_types)}",
        "scan_type",
        scan_type
    )
    
    # Early return if timeout is invalid
    if timeout is not None:
        early_return_if_invalid_type(timeout, (int, float), "Timeout must be a number")
        early_return_if_condition(
            timeout <= 0,
            "Timeout must be greater than 0",
            "timeout",
            timeout
        )
    
    return {
        'target': target,
        'scan_type': scan_type,
        'timeout': timeout or 30.0
    }


async def execute_scan_with_early_returns(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute scan with early returns for edge cases
    
    Args:
        target: Target to scan
        scan_type: Type of scan
        options: Scan options
        
    Returns:
        Scan results
    """
    # Early return if target is None
    early_return_if_none(target, "Scan target is required")
    
    # Early return if scan_type is None
    early_return_if_none(scan_type, "Scan type is required")
    
    # Early return if target host is unreachable
    host = target.get('host')
    if host:
        try:
            # Simulate host reachability check
            await asyncio.sleep(0.1)  # Simulate network check
            if host == "unreachable.example.com":
                return {
                    "success": False,
                    "error": "Target host is unreachable",
                    "target": host,
                    "scan_type": scan_type
                }
        except Exception:
            return {
                "success": False,
                "error": "Failed to reach target host",
                "target": host,
                "scan_type": scan_type
            }
    
    # Early return if scan type is not supported
    supported_scans = ['port_scan', 'vulnerability_scan', 'web_scan']
    if scan_type not in supported_scans:
        return {
            "success": False,
            "error": f"Unsupported scan type: {scan_type}",
            "supported_types": supported_scans
        }
    
    # Early return if options are invalid
    if options:
        early_return_if_invalid_type(options, dict, "Options must be a dictionary")
        
        # Check for invalid option values
        max_ports = options.get('max_ports')
        if max_ports is not None:
            early_return_if_invalid_type(max_ports, int, "max_ports must be an integer")
            early_return_if_condition(
                max_ports <= 0 or max_ports > 65535,
                "max_ports must be between 1 and 65535",
                "max_ports",
                max_ports
            )
    
    # Proceed with scan
    return {
        "success": True,
        "target": target,
        "scan_type": scan_type,
        "results": {"ports": [80, 443, 22], "status": "completed"}
    }


# ============================================================================
# ENUMERATION EARLY RETURNS
# ============================================================================

def validate_enumeration_inputs_early_return(
    target: str,
    enum_type: str,
    credentials: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Validate enumeration inputs with early returns
    
    Args:
        target: Target to enumerate
        enum_type: Type of enumeration
        credentials: Credentials for enumeration
        
    Returns:
        Validated enumeration configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Early return if target is None
    early_return_if_none(target, "Enumeration target is required")
    early_return_if_empty(target, "Enumeration target cannot be empty")
    
    # Early return if target is not a string
    early_return_if_invalid_type(target, str, "Target must be a string")
    
    # Early return if enum_type is invalid
    early_return_if_none(enum_type, "Enumeration type is required")
    early_return_if_empty(enum_type, "Enumeration type cannot be empty")
    
    valid_enum_types = ['dns', 'smb', 'ssh', 'user', 'service']
    early_return_if_condition(
        enum_type not in valid_enum_types,
        f"Invalid enumeration type. Must be one of: {', '.join(valid_enum_types)}",
        "enum_type",
        enum_type
    )
    
    # Early return if credentials are invalid
    if credentials is not None:
        early_return_if_invalid_type(credentials, dict, "Credentials must be a dictionary")
        
        username = credentials.get('username')
        password = credentials.get('password')
        
        if username is not None:
            early_return_if_invalid_type(username, str, "Username must be a string")
            early_return_if_empty(username, "Username cannot be empty")
        
        if password is not None:
            early_return_if_invalid_type(password, str, "Password must be a string")
    
    return {
        'target': target.strip(),
        'enum_type': enum_type,
        'credentials': credentials,
        'timeout': 30.0
    }


async def execute_enumeration_with_early_returns(
    target: str,
    enum_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute enumeration with early returns for edge cases
    
    Args:
        target: Target to enumerate
        enum_type: Type of enumeration
        options: Enumeration options
        
    Returns:
        Enumeration results
    """
    # Early return if target is None
    early_return_if_none(target, "Enumeration target is required")
    
    # Early return if target is empty
    early_return_if_empty(target, "Enumeration target cannot be empty")
    
    # Early return if target is not reachable
    if target == "unreachable.example.com":
        return {
            "success": False,
            "error": "Target is not reachable",
            "target": target,
            "enum_type": enum_type
        }
    
    # Early return if enumeration type is not supported
    supported_enums = ['dns', 'smb', 'ssh']
    if enum_type not in supported_enums:
        return {
            "success": False,
            "error": f"Unsupported enumeration type: {enum_type}",
            "supported_types": supported_enums
        }
    
    # Early return if options are invalid
    if options:
        early_return_if_invalid_type(options, dict, "Options must be a dictionary")
        
        # Check for invalid option values
        max_records = options.get('max_records')
        if max_records is not None:
            early_return_if_invalid_type(max_records, int, "max_records must be an integer")
            early_return_if_condition(
                max_records <= 0,
                "max_records must be greater than 0",
                "max_records",
                max_records
            )
    
    # Proceed with enumeration
    return {
        "success": True,
        "target": target,
        "enum_type": enum_type,
        "results": {"records": ["record1", "record2"], "status": "completed"}
    }


# ============================================================================
# ATTACK EARLY RETURNS
# ============================================================================

def validate_attack_inputs_early_return(
    target: str,
    attack_type: str,
    payload: Optional[str] = None,
    credentials: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Validate attack inputs with early returns
    
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
    # Early return if target is None
    early_return_if_none(target, "Attack target is required")
    early_return_if_empty(target, "Attack target cannot be empty")
    
    # Early return if target is not a string
    early_return_if_invalid_type(target, str, "Target must be a string")
    
    # Early return if attack_type is invalid
    early_return_if_none(attack_type, "Attack type is required")
    early_return_if_empty(attack_type, "Attack type cannot be empty")
    
    valid_attack_types = ['brute_force', 'exploitation', 'social_engineering']
    early_return_if_condition(
        attack_type not in valid_attack_types,
        f"Invalid attack type. Must be one of: {', '.join(valid_attack_types)}",
        "attack_type",
        attack_type
    )
    
    # Early return if payload is invalid
    if payload is not None:
        early_return_if_invalid_type(payload, str, "Payload must be a string")
        early_return_if_empty(payload, "Payload cannot be empty")
        
        # Check for dangerous payload patterns
        dangerous_patterns = ['<script>', 'javascript:', 'data:text/html']
        for pattern in dangerous_patterns:
            early_return_if_condition(
                pattern.lower() in payload.lower(),
                f"Dangerous payload pattern detected: {pattern}",
                "payload",
                payload
            )
    
    # Early return if credentials are invalid
    if credentials is not None:
        early_return_if_invalid_type(credentials, list, "Credentials must be a list")
        
        for i, cred in enumerate(credentials):
            early_return_if_invalid_type(cred, dict, f"Credential {i} must be a dictionary")
            
            username = cred.get('username')
            password = cred.get('password')
            
            if username is not None:
                early_return_if_invalid_type(username, str, f"Username in credential {i} must be a string")
            
            if password is not None:
                early_return_if_invalid_type(password, str, f"Password in credential {i} must be a string")
    
    return {
        'target': target.strip(),
        'attack_type': attack_type,
        'payload': payload,
        'credentials': credentials or [],
        'timeout': 30.0
    }


async def execute_attack_with_early_returns(
    target: str,
    attack_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute attack with early returns for edge cases
    
    Args:
        target: Target to attack
        attack_type: Type of attack
        options: Attack options
        
    Returns:
        Attack results
    """
    # Early return if target is None
    early_return_if_none(target, "Attack target is required")
    
    # Early return if target is empty
    early_return_if_empty(target, "Attack target cannot be empty")
    
    # Early return if target is protected
    if target == "protected.example.com":
        return {
            "success": False,
            "error": "Target is protected and cannot be attacked",
            "target": target,
            "attack_type": attack_type
        }
    
    # Early return if attack type is not supported
    supported_attacks = ['brute_force', 'exploitation']
    if attack_type not in supported_attacks:
        return {
            "success": False,
            "error": f"Unsupported attack type: {attack_type}",
            "supported_types": supported_attacks
        }
    
    # Early return if options are invalid
    if options:
        early_return_if_invalid_type(options, dict, "Options must be a dictionary")
        
        # Check for invalid option values
        max_attempts = options.get('max_attempts')
        if max_attempts is not None:
            early_return_if_invalid_type(max_attempts, int, "max_attempts must be an integer")
            early_return_if_condition(
                max_attempts <= 0,
                "max_attempts must be greater than 0",
                "max_attempts",
                max_attempts
            )
    
    # Proceed with attack
    return {
        "success": True,
        "target": target,
        "attack_type": attack_type,
        "results": {"attempts": 100, "successful": 0, "status": "completed"}
    }


# ============================================================================
# SECURITY EARLY RETURNS
# ============================================================================

def validate_security_inputs_early_return(
    user_id: str,
    action: str,
    resource: str,
    permissions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate security inputs with early returns
    
    Args:
        user_id: User identifier
        action: Action to perform
        resource: Resource to access
        permissions: User permissions
        
    Returns:
        Validated security configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Early return if user_id is None
    early_return_if_none(user_id, "User ID is required")
    early_return_if_empty(user_id, "User ID cannot be empty")
    
    # Early return if user_id is not a string
    early_return_if_invalid_type(user_id, str, "User ID must be a string")
    
    # Early return if action is None
    early_return_if_none(action, "Action is required")
    early_return_if_empty(action, "Action cannot be empty")
    
    # Early return if action is not a string
    early_return_if_invalid_type(action, str, "Action must be a string")
    
    # Early return if resource is None
    early_return_if_none(resource, "Resource is required")
    early_return_if_empty(resource, "Resource cannot be empty")
    
    # Early return if resource is not a string
    early_return_if_invalid_type(resource, str, "Resource must be a string")
    
    # Early return if permissions are invalid
    if permissions is not None:
        early_return_if_invalid_type(permissions, list, "Permissions must be a list")
        
        for i, permission in enumerate(permissions):
            early_return_if_invalid_type(permission, str, f"Permission {i} must be a string")
            early_return_if_empty(permission, f"Permission {i} cannot be empty")
    
    return {
        'user_id': user_id.strip(),
        'action': action.strip(),
        'resource': resource.strip(),
        'permissions': permissions or [],
        'timestamp': datetime.utcnow()
    }


async def check_authorization_with_early_returns(
    user_id: str,
    action: str,
    resource: str,
    user_permissions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Check authorization with early returns for edge cases
    
    Args:
        user_id: User identifier
        action: Action to perform
        resource: Resource to access
        user_permissions: User permissions
        
    Returns:
        Authorization result
    """
    # Early return if user_id is None
    early_return_if_none(user_id, "User ID is required")
    
    # Early return if user is blocked
    if user_id == "blocked_user":
        return {
            "authorized": False,
            "reason": "User is blocked",
            "user_id": user_id,
            "action": action,
            "resource": resource
        }
    
    # Early return if action is None
    early_return_if_none(action, "Action is required")
    
    # Early return if action is not allowed
    allowed_actions = ['read', 'write', 'delete', 'execute']
    if action not in allowed_actions:
        return {
            "authorized": False,
            "reason": f"Action '{action}' is not allowed",
            "allowed_actions": allowed_actions
        }
    
    # Early return if resource is None
    early_return_if_none(resource, "Resource is required")
    
    # Early return if resource is protected
    protected_resources = ['/admin', '/system', '/config']
    if resource in protected_resources:
        return {
            "authorized": False,
            "reason": f"Resource '{resource}' is protected",
            "user_id": user_id,
            "action": action,
            "resource": resource
        }
    
    # Early return if permissions are invalid
    if user_permissions is not None:
        early_return_if_invalid_type(user_permissions, list, "User permissions must be a list")
        
        # Check if user has required permissions
        required_permissions = [f"{action}_{resource}"]
        for permission in required_permissions:
            if permission not in user_permissions:
                return {
                    "authorized": False,
                    "reason": f"Missing permission: {permission}",
                    "user_id": user_id,
                    "action": action,
                    "resource": resource,
                    "required_permissions": required_permissions,
                    "user_permissions": user_permissions
                }
    
    # Authorization successful
    return {
        "authorized": True,
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "timestamp": datetime.utcnow()
    }


# ============================================================================
# DATABASE EARLY RETURNS
# ============================================================================

def validate_database_inputs_early_return(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    operation: str = "select"
) -> Dict[str, Any]:
    """
    Validate database inputs with early returns
    
    Args:
        query: SQL query
        parameters: Query parameters
        operation: Database operation
        
    Returns:
        Validated database configuration
        
    Raises:
        InputValidationError: If validation fails
    """
    # Early return if query is None
    early_return_if_none(query, "SQL query is required")
    early_return_if_empty(query, "SQL query cannot be empty")
    
    # Early return if query is not a string
    early_return_if_invalid_type(query, str, "Query must be a string")
    
    # Early return if query is too long
    early_return_if_condition(
        len(query) > 10000,
        "Query is too long (maximum 10000 characters)",
        "query",
        query
    )
    
    # Early return if operation is invalid
    early_return_if_none(operation, "Database operation is required")
    early_return_if_empty(operation, "Database operation cannot be empty")
    
    valid_operations = ['select', 'insert', 'update', 'delete', 'create', 'drop']
    early_return_if_condition(
        operation not in valid_operations,
        f"Invalid operation. Must be one of: {', '.join(valid_operations)}",
        "operation",
        operation
    )
    
    # Early return if parameters are invalid
    if parameters is not None:
        early_return_if_invalid_type(parameters, dict, "Parameters must be a dictionary")
        
        for key, value in parameters.items():
            early_return_if_invalid_type(key, str, f"Parameter key '{key}' must be a string")
            early_return_if_empty(key, f"Parameter key '{key}' cannot be empty")
    
    return {
        'query': query.strip(),
        'parameters': parameters or {},
        'operation': operation,
        'timeout': 30.0
    }


async def execute_database_operation_with_early_returns(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    operation: str = "select"
) -> Dict[str, Any]:
    """
    Execute database operation with early returns for edge cases
    
    Args:
        query: SQL query
        parameters: Query parameters
        operation: Database operation
        
    Returns:
        Database operation result
    """
    # Early return if query is None
    early_return_if_none(query, "SQL query is required")
    
    # Early return if query is empty
    early_return_if_empty(query, "SQL query cannot be empty")
    
    # Early return if query contains dangerous patterns
    dangerous_patterns = ['DROP TABLE', 'DELETE FROM', 'TRUNCATE', 'EXEC', 'EXECUTE']
    query_upper = query.upper()
    for pattern in dangerous_patterns:
        if pattern in query_upper:
            return {
                "success": False,
                "error": f"Dangerous SQL pattern detected: {pattern}",
                "query": query,
                "operation": operation
            }
    
    # Early return if operation is not supported
    supported_operations = ['select', 'insert', 'update']
    if operation not in supported_operations:
        return {
            "success": False,
            "error": f"Unsupported operation: {operation}",
            "supported_operations": supported_operations
        }
    
    # Early return if parameters are invalid
    if parameters:
        early_return_if_invalid_type(parameters, dict, "Parameters must be a dictionary")
        
        # Check for SQL injection patterns in parameters
        for key, value in parameters.items():
            if isinstance(value, str):
                sql_patterns = ["'", '"', ';', '--', '/*', '*/']
                for pattern in sql_patterns:
                    if pattern in value:
                        return {
                            "success": False,
                            "error": f"SQL injection pattern detected in parameter '{key}': {pattern}",
                            "parameter": key,
                            "value": value
                        }
    
    # Proceed with database operation
    return {
        "success": True,
        "query": query,
        "parameters": parameters or {},
        "operation": operation,
        "results": {"rows_affected": 1, "status": "completed"}
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of early returns
    print("🔄 Early Returns Example")
    
    # Test scan validation with early returns
    try:
        valid_scan = validate_scan_inputs_early_return(
            target={'host': '192.168.1.100', 'port': 80},
            scan_type='port_scan',
            timeout=30.0
        )
        print(f"Valid scan config: {valid_scan}")
    except InputValidationError as e:
        print(f"Scan validation error: {e}")
    
    # Test enumeration validation with early returns
    try:
        valid_enum = validate_enumeration_inputs_early_return(
            target='example.com',
            enum_type='dns',
            credentials={'username': 'user', 'password': 'pass'}
        )
        print(f"Valid enumeration config: {valid_enum}")
    except InputValidationError as e:
        print(f"Enumeration validation error: {e}")
    
    # Test attack validation with early returns
    try:
        valid_attack = validate_attack_inputs_early_return(
            target='192.168.1.100',
            attack_type='brute_force',
            credentials=[{'username': 'admin', 'password': 'password'}]
        )
        print(f"Valid attack config: {valid_attack}")
    except InputValidationError as e:
        print(f"Attack validation error: {e}")
    
    # Test security validation with early returns
    try:
        valid_security = validate_security_inputs_early_return(
            user_id='user123',
            action='read',
            resource='/data',
            permissions=['read_data', 'write_data']
        )
        print(f"Valid security config: {valid_security}")
    except InputValidationError as e:
        print(f"Security validation error: {e}")
    
    # Test database validation with early returns
    try:
        valid_db = validate_database_inputs_early_return(
            query='SELECT * FROM users WHERE id = %s',
            parameters={'id': 1},
            operation='select'
        )
        print(f"Valid database config: {valid_db}")
    except InputValidationError as e:
        print(f"Database validation error: {e}")
    
    # Test early return utilities
    try:
        early_return_if_none("valid_value")  # Should not raise
        print("Early return utility test passed")
    except InputValidationError as e:
        print(f"Early return utility error: {e}")
    
    try:
        early_return_if_none(None, "Custom error message")  # Should raise
    except InputValidationError as e:
        print(f"Expected early return error: {e}") 