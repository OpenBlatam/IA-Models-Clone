#!/usr/bin/env python3
"""
Anti-Patterns Guide for Video-OpusClip
Avoiding nested conditionals and keeping happy path last
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta

from .custom_exceptions import (
    ValidationError, InputValidationError, TypeValidationError,
    SecurityError, AuthenticationError, AuthorizationError
)


# ============================================================================
# ANTI-PATTERNS TO AVOID
# ============================================================================

def anti_pattern_nested_conditionals(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    ANTI-PATTERN: Nested conditionals - DON'T DO THIS
    
    This function demonstrates what NOT to do:
    - Deeply nested if-else statements
    - Happy path buried in the middle
    - Hard to read and maintain
    - Difficult to test individual conditions
    """
    if target is not None:
        if isinstance(target, dict):
            host = target.get('host')
            if host is not None:
                if isinstance(host, str) and host.strip():
                    if scan_type in ['port_scan', 'vulnerability_scan', 'web_scan']:
                        if options is None or isinstance(options, dict):
                            if options is None or 'timeout' not in options or (
                                isinstance(options.get('timeout'), (int, float)) and 
                                options.get('timeout') > 0
                            ):
                                # Happy path buried deep in nested conditionals
                                return {
                                    'target': target,
                                    'scan_type': scan_type,
                                    'status': 'completed',
                                    'timestamp': datetime.utcnow(),
                                    'results': {'open_ports': [80, 443]}
                                }
                            else:
                                raise InputValidationError("Invalid timeout value", "timeout", options.get('timeout'))
                        else:
                            raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
                    else:
                        raise InputValidationError("Target host cannot be empty", "host", host)
                else:
                    raise InputValidationError("Target host is required", "host", host)
            else:
                raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
        else:
            raise InputValidationError("Target is required", "target", target)
    else:
        raise InputValidationError("Target is required", "target", target)


def anti_pattern_happy_path_first(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    ANTI-PATTERN: Happy path first - DON'T DO THIS
    
    This function demonstrates what NOT to do:
    - Happy path at the beginning
    - Error handling scattered throughout
    - Inconsistent flow
    - Hard to follow logic
    """
    # Happy path first (BAD!)
    scan_results = {
        'target': target,
        'scan_type': scan_type,
        'status': 'completed',
        'timestamp': datetime.utcnow(),
        'results': {'open_ports': [80, 443]}
    }
    
    # Error handling scattered throughout (BAD!)
    if target is None:
        raise InputValidationError("Target is required", "target", target)
    
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
        raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
    
    if options is not None:
        if not isinstance(options, dict):
            raise TypeValidationError("Options must be a dictionary", "options", options, "dict", type(options).__name__)
        
        timeout = options.get('timeout')
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise InputValidationError("Timeout must be a positive number", "timeout", timeout)
    
    return scan_results


def anti_pattern_mixed_responsibilities(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    ANTI-PATTERN: Mixed responsibilities - DON'T DO THIS
    
    This function demonstrates what NOT to do:
    - Validation mixed with business logic
    - Multiple responsibilities in one function
    - Hard to test individual parts
    - Difficult to reuse
    """
    # Validation mixed with business logic (BAD!)
    if target is None:
        raise InputValidationError("Target is required", "target", target)
    
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
        raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
    
    # Business logic mixed with validation (BAD!)
    scan_results = {
        'target': target,
        'scan_type': scan_type,
        'status': 'completed',
        'timestamp': datetime.utcnow()
    }
    
    # More validation mixed with business logic (BAD!)
    if scan_type == 'port_scan':
        scan_results['results'] = {'open_ports': [80, 443, 22]}
        if options and options.get('detailed'):
            scan_results['results']['services'] = ['http', 'https', 'ssh']
    elif scan_type == 'vulnerability_scan':
        scan_results['results'] = {'vulnerabilities': ['CVE-2021-1234']}
        if options and options.get('severity'):
            scan_results['results']['severity'] = 'high'
    else:  # web_scan
        scan_results['results'] = {'web_technologies': ['nginx', 'php']}
        if options and options.get('headers'):
            scan_results['results']['headers'] = {'Server': 'nginx/1.18.0'}
    
    # More validation (BAD!)
    if options is not None:
        if not isinstance(options, dict):
            raise TypeValidationError("Options must be a dictionary", "options", options, "dict", type(options).__name__)
        
        timeout = options.get('timeout')
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise InputValidationError("Timeout must be a positive number", "timeout", timeout)
    
    return scan_results


def anti_pattern_early_returns_scattered(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    ANTI-PATTERN: Early returns scattered - DON'T DO THIS
    
    This function demonstrates what NOT to do:
    - Early returns mixed with business logic
    - Inconsistent error handling
    - Hard to follow the flow
    """
    # Some validation at the beginning
    if target is None:
        raise InputValidationError("Target is required", "target", target)
    
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    # Business logic
    scan_results = {
        'target': target,
        'scan_type': scan_type,
        'status': 'processing',
        'timestamp': datetime.utcnow()
    }
    
    # More validation scattered throughout (BAD!)
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    # More business logic
    if scan_type == 'port_scan':
        scan_results['results'] = {'open_ports': [80, 443]}
    
    # More validation (BAD!)
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    # More business logic
    if scan_type == 'vulnerability_scan':
        scan_results['results'] = {'vulnerabilities': ['CVE-2021-1234']}
    
    # More validation (BAD!)
    if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
        raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
    
    # Final business logic
    scan_results['status'] = 'completed'
    
    return scan_results


# ============================================================================
# REFACTORED PATTERNS (GOOD PRACTICES)
# ============================================================================

def refactored_guard_clauses_pattern(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    REFACTORED: Guard clauses pattern - DO THIS
    
    This function demonstrates good practices:
    - All validation at the beginning (guard clauses)
    - Early returns for error conditions
    - Happy path at the end
    - Clear separation of concerns
    """
    # Guard clause 1: Check if target is provided
    if target is None:
        raise InputValidationError("Target is required", "target", target)
    
    # Guard clause 2: Check if target is a dictionary
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    # Guard clause 3: Check if host is provided
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    # Guard clause 4: Check if host is valid
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    # Guard clause 5: Check if scan type is valid
    if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
        raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
    
    # Guard clause 6: Check if options are valid (if provided)
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
        'results': _generate_scan_results(scan_type, options)
    }
    
    return scan_results


def _generate_scan_results(scan_type: str, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Separate function for business logic - good practice
    """
    if scan_type == 'port_scan':
        results = {'open_ports': [80, 443, 22]}
        if options and options.get('detailed'):
            results['services'] = ['http', 'https', 'ssh']
        return results
    elif scan_type == 'vulnerability_scan':
        results = {'vulnerabilities': ['CVE-2021-1234']}
        if options and options.get('severity'):
            results['severity'] = 'high'
        return results
    else:  # web_scan
        results = {'web_technologies': ['nginx', 'php']}
        if options and options.get('headers'):
            results['headers'] = {'Server': 'nginx/1.18.0'}
        return results


def refactored_separate_validation_pattern(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    REFACTORED: Separate validation pattern - DO THIS
    
    This function demonstrates good practices:
    - Separate validation function
    - Clear separation of concerns
    - Easy to test validation independently
    """
    # Validate inputs first
    _validate_scan_inputs(target, scan_type, options)
    
    # Happy path: Process the scan request
    scan_results = {
        'target': target,
        'scan_type': scan_type,
        'status': 'completed',
        'timestamp': datetime.utcnow(),
        'results': _generate_scan_results(scan_type, options)
    }
    
    return scan_results


def _validate_scan_inputs(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]]
) -> None:
    """
    Separate validation function - good practice
    """
    # Guard clause 1: Check if target is provided
    if target is None:
        raise InputValidationError("Target is required", "target", target)
    
    # Guard clause 2: Check if target is a dictionary
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
    
    # Guard clause 3: Check if host is provided
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    # Guard clause 4: Check if host is valid
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    # Guard clause 5: Check if scan type is valid
    if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
        raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
    
    # Guard clause 6: Check if options are valid (if provided)
    if options is not None:
        if not isinstance(options, dict):
            raise TypeValidationError("Options must be a dictionary", "options", options, "dict", type(options).__name__)
        
        timeout = options.get('timeout')
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise InputValidationError("Timeout must be a positive number", "timeout", timeout)


def refactored_decorator_pattern(
    target: Dict[str, Any],
    scan_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    REFACTORED: Decorator pattern - DO THIS
    
    This function demonstrates good practices:
    - Validation handled by decorator
    - Clean business logic
    - Reusable validation
    """
    # Happy path: Process the scan request (validation handled by decorator)
    scan_results = {
        'target': target,
        'scan_type': scan_type,
        'status': 'completed',
        'timestamp': datetime.utcnow(),
        'results': _generate_scan_results(scan_type, options)
    }
    
    return scan_results


def validate_scan_decorator(func: Callable) -> Callable:
    """
    Validation decorator - good practice
    """
    def wrapper(target: Dict[str, Any], scan_type: str, options: Optional[Dict[str, Any]] = None):
        # Guard clause 1: Check if target is provided
        if target is None:
            raise InputValidationError("Target is required", "target", target)
        
        # Guard clause 2: Check if target is a dictionary
        if not isinstance(target, dict):
            raise TypeValidationError("Target must be a dictionary", "target", target, "dict", type(target).__name__)
        
        # Guard clause 3: Check if host is provided
        host = target.get('host')
        if host is None:
            raise InputValidationError("Target host is required", "host", host)
        
        # Guard clause 4: Check if host is valid
        if not isinstance(host, str) or not host.strip():
            raise InputValidationError("Target host cannot be empty", "host", host)
        
        # Guard clause 5: Check if scan type is valid
        if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
            raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
        
        # Guard clause 6: Check if options are valid (if provided)
        if options is not None:
            if not isinstance(options, dict):
                raise TypeValidationError("Options must be a dictionary", "options", options, "dict", type(options).__name__)
            
            timeout = options.get('timeout')
            if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
                raise InputValidationError("Timeout must be a positive number", "timeout", timeout)
        
        # Call the original function
        return func(target, scan_type, options)
    
    return wrapper


# ============================================================================
# COMPARISON EXAMPLES
# ============================================================================

def compare_patterns():
    """Compare anti-patterns with good practices"""
    print("🔄 Pattern Comparison Examples")
    
    target = {'host': '192.168.1.100', 'port': 80}
    scan_type = 'port_scan'
    options = {'timeout': 30}
    
    print("\n" + "="*60)
    print("ANTI-PATTERNS (DON'T DO THIS)")
    print("="*60)
    
    print("\n1. Nested Conditionals:")
    print("   - Deeply nested if-else statements")
    print("   - Happy path buried in the middle")
    print("   - Hard to read and maintain")
    print("   - Difficult to test individual conditions")
    
    print("\n2. Happy Path First:")
    print("   - Happy path at the beginning")
    print("   - Error handling scattered throughout")
    print("   - Inconsistent flow")
    print("   - Hard to follow logic")
    
    print("\n3. Mixed Responsibilities:")
    print("   - Validation mixed with business logic")
    print("   - Multiple responsibilities in one function")
    print("   - Hard to test individual parts")
    print("   - Difficult to reuse")
    
    print("\n4. Early Returns Scattered:")
    print("   - Early returns mixed with business logic")
    print("   - Inconsistent error handling")
    print("   - Hard to follow the flow")
    
    print("\n" + "="*60)
    print("GOOD PRACTICES (DO THIS)")
    print("="*60)
    
    print("\n1. Guard Clauses Pattern:")
    print("   - All validation at the beginning")
    print("   - Early returns for error conditions")
    print("   - Happy path at the end")
    print("   - Clear separation of concerns")
    
    print("\n2. Separate Validation Pattern:")
    print("   - Separate validation function")
    print("   - Clear separation of concerns")
    print("   - Easy to test validation independently")
    
    print("\n3. Decorator Pattern:")
    print("   - Validation handled by decorator")
    print("   - Clean business logic")
    print("   - Reusable validation")
    
    print("\n" + "="*60)
    print("CODE EXAMPLES")
    print("="*60)
    
    print("\nAnti-Pattern (Nested Conditionals):")
    print("""
def anti_pattern_nested_conditionals(target, scan_type, options):
    if target is not None:
        if isinstance(target, dict):
            host = target.get('host')
            if host is not None:
                if isinstance(host, str) and host.strip():
                    if scan_type in ['port_scan', 'vulnerability_scan', 'web_scan']:
                        if options is None or isinstance(options, dict):
                            # Happy path buried deep in nested conditionals
                            return {'status': 'completed'}
                        else:
                            raise ValidationError("Invalid options")
                    else:
                        raise ValidationError("Invalid scan type")
                else:
                    raise ValidationError("Invalid host")
            else:
                raise ValidationError("Host required")
        else:
            raise ValidationError("Target must be dict")
    else:
        raise ValidationError("Target required")
    """)
    
    print("\nGood Practice (Guard Clauses):")
    print("""
def refactored_guard_clauses_pattern(target, scan_type, options):
    # Guard clause 1: Check if target is provided
    if target is None:
        raise InputValidationError("Target is required", "target", target)
    
    # Guard clause 2: Check if target is a dictionary
    if not isinstance(target, dict):
        raise TypeValidationError("Target must be a dictionary", "target", target)
    
    # Guard clause 3: Check if host is provided
    host = target.get('host')
    if host is None:
        raise InputValidationError("Target host is required", "host", host)
    
    # Guard clause 4: Check if host is valid
    if not isinstance(host, str) or not host.strip():
        raise InputValidationError("Target host cannot be empty", "host", host)
    
    # Guard clause 5: Check if scan type is valid
    if scan_type not in ['port_scan', 'vulnerability_scan', 'web_scan']:
        raise InputValidationError(f"Invalid scan type: {scan_type}", "scan_type", scan_type)
    
    # Happy path: Process the scan request
    return {
        'target': target,
        'scan_type': scan_type,
        'status': 'completed',
        'timestamp': datetime.utcnow()
    }
    """)


# ============================================================================
# BEST PRACTICES SUMMARY
# ============================================================================

def best_practices_summary():
    """Summary of best practices for avoiding nested conditionals"""
    print("📋 Best Practices Summary")
    
    print("\n" + "="*60)
    print("DO'S")
    print("="*60)
    
    print("\n✅ Use Guard Clauses:")
    print("   - Put all validation at the beginning of the function")
    print("   - Use early returns for error conditions")
    print("   - Keep the happy path at the end")
    
    print("\n✅ Separate Concerns:")
    print("   - Separate validation from business logic")
    print("   - Create dedicated validation functions")
    print("   - Use decorators for reusable validation")
    
    print("\n✅ Keep Functions Focused:")
    print("   - One function, one responsibility")
    print("   - Extract complex logic into separate functions")
    print("   - Make functions easy to test")
    
    print("\n✅ Use Clear Error Messages:")
    print("   - Provide specific error messages")
    print("   - Include context information")
    print("   - Use consistent error types")
    
    print("\n✅ Follow Consistent Patterns:")
    print("   - Use the same validation pattern throughout")
    print("   - Maintain consistent function structure")
    print("   - Follow established conventions")
    
    print("\n" + "="*60)
    print("DON'TS")
    print("="*60)
    
    print("\n❌ Avoid Nested Conditionals:")
    print("   - Don't use deeply nested if-else statements")
    print("   - Don't bury the happy path in the middle")
    print("   - Don't mix validation with business logic")
    
    print("\n❌ Avoid Happy Path First:")
    print("   - Don't put the happy path at the beginning")
    print("   - Don't scatter error handling throughout")
    print("   - Don't create inconsistent flow")
    
    print("\n❌ Avoid Mixed Responsibilities:")
    print("   - Don't mix validation with business logic")
    print("   - Don't create functions with multiple purposes")
    print("   - Don't make functions hard to test")
    
    print("\n❌ Avoid Scattered Early Returns:")
    print("   - Don't mix early returns with business logic")
    print("   - Don't create inconsistent error handling")
    print("   - Don't make the flow hard to follow")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of anti-patterns guide
    print("🚫 Anti-Patterns Guide Example")
    
    # Show pattern comparison
    compare_patterns()
    
    # Show best practices summary
    best_practices_summary()
    
    # Test refactored patterns
    print("\n" + "="*60)
    print("TESTING REFACTORED PATTERNS")
    print("="*60)
    
    target = {'host': '192.168.1.100', 'port': 80}
    scan_type = 'port_scan'
    options = {'timeout': 30}
    
    # Test guard clauses pattern
    try:
        results = refactored_guard_clauses_pattern(target, scan_type, options)
        print(f"✅ Guard clauses pattern successful: {results['status']}")
    except Exception as e:
        print(f"❌ Guard clauses pattern failed: {e}")
    
    # Test separate validation pattern
    try:
        results = refactored_separate_validation_pattern(target, scan_type, options)
        print(f"✅ Separate validation pattern successful: {results['status']}")
    except Exception as e:
        print(f"❌ Separate validation pattern failed: {e}")
    
    # Test decorator pattern
    try:
        decorated_func = validate_scan_decorator(refactored_decorator_pattern)
        results = decorated_func(target, scan_type, options)
        print(f"✅ Decorator pattern successful: {results['status']}")
    except Exception as e:
        print(f"❌ Decorator pattern failed: {e}")
    
    print("\n✅ Anti-patterns guide examples completed!") 