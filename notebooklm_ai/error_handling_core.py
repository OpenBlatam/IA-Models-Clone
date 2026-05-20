from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import traceback
import json
from typing import Any, List, Dict, Optional
"""
Error Handling and Validation Core - Comprehensive error management
Uses guard clauses, early returns, and structured logging
"""


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class ValidationError:
    """Immutable validation error with context"""
    field_name: str
    error_type: str
    error_message: str
    value: Any = None
    expected_format: Optional[str] = None

@dataclass(frozen=True)
class ErrorContext:
    """Immutable error context for structured logging"""
    module_name: str
    function_name: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_id: str = field(default_factory=lambda: f"err_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}")

@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class OperationResult:
    """Immutable operation result with error handling"""
    is_successful: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_context: Optional[ErrorContext] = None
    validation_result: Optional[ValidationResult] = None

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_structured_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup structured logging with JSON format"""
    logger = logging.getLogger("security_core")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create JSON formatter
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "function": "%(funcName)s", "message": %(message)s}'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler("security_operations.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_structured_logging()

# ============================================================================
# VALIDATION FUNCTIONS (CPU-bound)
# ============================================================================

def validate_ip_address(ip_address: str) -> ValidationResult:
    """Validate IP address format with guard clauses"""
    # Guard clause: Check if input is string
    if not isinstance(ip_address, str):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="ip_address",
                error_type="type_error",
                error_message="IP address must be a string",
                value=ip_address,
                expected_format="string"
            )]
        )
    
    # Guard clause: Check if input is empty
    if not ip_address.strip():
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="ip_address",
                error_type="empty_value",
                error_message="IP address cannot be empty",
                value=ip_address
            )]
        )
    
    # Guard clause: Check IP format
    ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    if not re.match(ip_pattern, ip_address):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="ip_address",
                error_type="format_error",
                error_message="Invalid IP address format",
                value=ip_address,
                expected_format="xxx.xxx.xxx.xxx"
            )]
        )
    
    return ValidationResult(is_valid=True)

def validate_port_number(port: Union[int, str]) -> ValidationResult:
    """Validate port number with guard clauses"""
    # Guard clause: Check if input is valid type
    if not isinstance(port, (int, str)):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="port",
                error_type="type_error",
                error_message="Port must be an integer or string",
                value=port,
                expected_format="int or str"
            )]
        )
    
    # Convert string to int if needed
    try:
        port_int = int(port) if isinstance(port, str) else port
    except ValueError:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="port",
                error_type="conversion_error",
                error_message="Port must be a valid number",
                value=port,
                expected_format="integer"
            )]
        )
    
    # Guard clause: Check port range
    if port_int < 1 or port_int > 65535:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="port",
                error_type="range_error",
                error_message="Port must be between 1 and 65535",
                value=port_int,
                expected_format="1-65535"
            )]
        )
    
    return ValidationResult(is_valid=True)

def validate_domain_name(domain: str) -> ValidationResult:
    """Validate domain name with guard clauses"""
    # Guard clause: Check if input is string
    if not isinstance(domain, str):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="domain",
                error_type="type_error",
                error_message="Domain must be a string",
                value=domain,
                expected_format="string"
            )]
        )
    
    # Guard clause: Check if input is empty
    if not domain.strip():
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="domain",
                error_type="empty_value",
                error_message="Domain cannot be empty",
                value=domain
            )]
        )
    
    # Guard clause: Check domain format
    domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
    if not re.match(domain_pattern, domain):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="domain",
                error_type="format_error",
                error_message="Invalid domain name format",
                value=domain,
                expected_format="example.com"
            )]
        )
    
    return ValidationResult(is_valid=True)

def validate_password_strength(password: str) -> ValidationResult:
    """Validate password strength with guard clauses"""
    errors = []
    warnings = []
    
    # Guard clause: Check if input is string
    if not isinstance(password, str):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="password",
                error_type="type_error",
                error_message="Password must be a string",
                value=password,
                expected_format="string"
            )]
        )
    
    # Guard clause: Check minimum length
    if len(password) < 12:
        errors.append(ValidationError(
            field_name="password",
            error_type="length_error",
            error_message="Password must be at least 12 characters long",
            value=len(password),
            expected_format=">=12 characters"
        ))
    
    # Check character requirements
    if not re.search(r'[A-Z]', password):
        errors.append(ValidationError(
            field_name="password",
            error_type="character_error",
            error_message="Password must contain at least one uppercase letter",
            value="no uppercase letters"
        ))
    
    if not re.search(r'[a-z]', password):
        errors.append(ValidationError(
            field_name="password",
            error_type="character_error",
            error_message="Password must contain at least one lowercase letter",
            value="no lowercase letters"
        ))
    
    if not re.search(r'\d', password):
        errors.append(ValidationError(
            field_name="password",
            error_type="character_error",
            error_message="Password must contain at least one digit",
            value="no digits"
        ))
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append(ValidationError(
            field_name="password",
            error_type="character_error",
            error_message="Password must contain at least one special character",
            value="no special characters"
        ))
    
    # Add warnings for additional security
    if len(password) < 16:
        warnings.append("Consider using a password longer than 16 characters")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]{2,}', password):
        warnings.append("Consider using multiple special characters")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_scan_parameters(parameters: Dict[str, Any]) -> ValidationResult:
    """Validate scan parameters with guard clauses"""
    errors = []
    
    # Guard clause: Check if input is dictionary
    if not isinstance(parameters, dict):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="parameters",
                error_type="type_error",
                error_message="Parameters must be a dictionary",
                value=parameters,
                expected_format="dict"
            )]
        )
    
    # Validate required fields
    required_fields = ["timeout", "max_threads"]
    for field_name in required_fields:
        if field_name not in parameters:
            errors.append(ValidationError(
                field_name=field_name,
                error_type="missing_field",
                error_message=f"Required field '{field_name}' is missing",
                expected_format="required"
            ))
    
    # Validate timeout if present
    if "timeout" in parameters:
        timeout = parameters["timeout"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append(ValidationError(
                field_name="timeout",
                error_type="range_error",
                error_message="Timeout must be a positive number",
                value=timeout,
                expected_format="positive number"
            ))
    
    # Validate max_threads if present
    if "max_threads" in parameters:
        max_threads = parameters["max_threads"]
        if not isinstance(max_threads, int) or max_threads <= 0:
            errors.append(ValidationError(
                field_name="max_threads",
                error_type="range_error",
                error_message="Max threads must be a positive integer",
                value=max_threads,
                expected_format="positive integer"
            ))
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)

# ============================================================================
# ERROR HANDLING FUNCTIONS
# ============================================================================

def log_error_with_context(
    error: Exception,
    context: ErrorContext,
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """Log error with structured context"""
    error_data = {
        "error_id": context.error_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "module_name": context.module_name,
        "function_name": context.function_name,
        "parameters": context.parameters,
        "timestamp": context.timestamp.isoformat(),
        "traceback": traceback.format_exc()
    }
    
    if additional_data:
        error_data.update(additional_data)
    
    logger.error(json.dumps(error_data))

def create_error_context(
    module_name: str,
    function_name: str,
    parameters: Dict[str, Any]
) -> ErrorContext:
    """Create error context for structured logging"""
    return ErrorContext(
        module_name=module_name,
        function_name=function_name,
        parameters=parameters
    )

def handle_validation_errors(
    validation_result: ValidationResult,
    context: ErrorContext
) -> OperationResult:
    """Handle validation errors and return operation result"""
    if not validation_result.is_valid:
        error_messages = [f"{error.field_name}: {error.error_message}" 
                         for error in validation_result.errors]
        
        log_error_with_context(
            ValueError("Validation failed"),
            context,
            {"validation_errors": [error.__dict__ for error in validation_result.errors]}
        )
        
        return OperationResult(
            is_successful=False,
            error_message="; ".join(error_messages),
            error_context=context,
            validation_result=validation_result
        )
    
    return OperationResult(
        is_successful=True,
        validation_result=validation_result
    )

# ============================================================================
# DECORATORS FOR ERROR HANDLING
# ============================================================================

def with_error_handling[T](func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add comprehensive error handling"""
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = create_error_context(
                module_name=func.__module__,
                function_name=func.__name__,
                parameters={"args": str(args), "kwargs": kwargs}
            )
            log_error_with_context(e, context)
            raise
    return wrapper

def with_async_error_handling[T](func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add comprehensive error handling for async functions"""
    async def wrapper(*args, **kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            context = create_error_context(
                module_name=func.__module__,
                function_name=func.__name__,
                parameters={"args": str(args), "kwargs": kwargs}
            )
            log_error_with_context(e, context)
            raise
    return wrapper

def with_validation(validation_func: Callable[[Any], ValidationResult]):
    """Decorator to add validation to functions"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Validate first argument (assuming it's the main input)
            if args:
                validation_result = validation_func(args[0])
                if not validation_result.is_valid:
                    context = create_error_context(
                        module_name=func.__module__,
                        function_name=func.__name__,
                        parameters={"args": str(args), "kwargs": kwargs}
                    )
                    return handle_validation_errors(validation_result, context)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ============================================================================
# EXAMPLE USAGE WITH GUARD CLAUSES
# ============================================================================

@with_validation(validate_ip_address)
def scan_single_port(ip_address: str, port: int) -> OperationResult:
    """Scan single port with guard clauses and validation"""
    # Guard clause: Validate port
    port_validation = validate_port_number(port)
    if not port_validation.is_valid:
        context = create_error_context(
            module_name=__name__,
            function_name="scan_single_port",
            parameters={"ip_address": ip_address, "port": port}
        )
        return handle_validation_errors(port_validation, context)
    
    # Guard clause: Check if port is in common range
    if port < 1 or port > 65535:
        context = create_error_context(
            module_name=__name__,
            function_name="scan_single_port",
            parameters={"ip_address": ip_address, "port": port}
        )
        return OperationResult(
            is_successful=False,
            error_message="Port must be between 1 and 65535",
            error_context=context
        )
    
    # Main operation logic would go here
    try:
        # Simulate port scanning
        is_open = port % 2 == 0  # Simulated result
        
        return OperationResult(
            is_successful=True,
            data={
                "ip_address": ip_address,
                "port": port,
                "is_open": is_open
            }
        )
        
    except Exception as e:
        context = create_error_context(
            module_name=__name__,
            function_name="scan_single_port",
            parameters={"ip_address": ip_address, "port": port}
        )
        log_error_with_context(e, context)
        
        return OperationResult(
            is_successful=False,
            error_message=str(e),
            error_context=context
        )

@with_async_error_handling
async def scan_multiple_ports_async(ip_address: str, ports: List[int]) -> OperationResult:
    """Scan multiple ports asynchronously with guard clauses"""
    # Guard clause: Validate IP address
    ip_validation = validate_ip_address(ip_address)
    if not ip_validation.is_valid:
        context = create_error_context(
            module_name=__name__,
            function_name="scan_multiple_ports_async",
            parameters={"ip_address": ip_address, "ports": ports}
        )
        return handle_validation_errors(ip_validation, context)
    
    # Guard clause: Validate ports list
    if not isinstance(ports, list) or len(ports) == 0:
        context = create_error_context(
            module_name=__name__,
            function_name="scan_multiple_ports_async",
            parameters={"ip_address": ip_address, "ports": ports}
        )
        return OperationResult(
            is_successful=False,
            error_message="Ports must be a non-empty list",
            error_context=context
        )
    
    # Guard clause: Validate each port
    for port in ports:
        port_validation = validate_port_number(port)
        if not port_validation.is_valid:
            context = create_error_context(
                module_name=__name__,
                function_name="scan_multiple_ports_async",
                parameters={"ip_address": ip_address, "ports": ports}
            )
            return handle_validation_errors(port_validation, context)
    
    # Main operation logic
    try:
        results = []
        for port in ports:
            result = scan_single_port(ip_address, port)
            results.append(result.data)
        
        return OperationResult(
            is_successful=True,
            data={
                "ip_address": ip_address,
                "scan_results": results,
                "total_ports": len(ports)
            }
        )
        
    except Exception as e:
        context = create_error_context(
            module_name=__name__,
            function_name="scan_multiple_ports_async",
            parameters={"ip_address": ip_address, "ports": ports}
        )
        log_error_with_context(e, context)
        
        return OperationResult(
            is_successful=False,
            error_message=str(e),
            error_context=context
        )

# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Data structures
    "ValidationError",
    "ErrorContext", 
    "ValidationResult",
    "OperationResult",
    
    # Validation functions
    "validate_ip_address",
    "validate_port_number",
    "validate_domain_name",
    "validate_password_strength",
    "validate_scan_parameters",
    
    # Error handling functions
    "log_error_with_context",
    "create_error_context",
    "handle_validation_errors",
    
    # Decorators
    "with_error_handling",
    "with_async_error_handling",
    "with_validation",
    
    # Example functions
    "scan_single_port",
    "scan_multiple_ports_async",
    
    # Logging
    "logger",
    "setup_structured_logging"
] 