"""
Error Handling Module for Video-OpusClip
Comprehensive error handling, validation, guard clauses, structured logging, exception mapping, and happy path patterns
"""

from .custom_exceptions import (
    VideoOpusClipException, ValidationError, InputValidationError, SchemaValidationError, TypeValidationError,
    SecurityError, AuthenticationError, AuthorizationError, RateLimitError, IntrusionDetectionError,
    ScanningError, PortScanError, VulnerabilityScanError, WebScanError,
    EnumerationError, DNSEnumerationError, SMBEnumerationError, SSHEnumerationError,
    AttackError, BruteForceError, ExploitationError,
    DatabaseError, ConnectionError, QueryError, TransactionError,
    NetworkError, ConnectionTimeoutError, DNSResolutionError, HTTPError,
    FileSystemError, FileNotFoundError, FilePermissionError, FileSizeError,
    ConfigurationError, MissingConfigurationError, InvalidConfigurationError
)

from .error_handlers import (
    setup_error_logging, create_error_response, create_user_friendly_error_response,
    handle_errors, handle_specific_errors, error_context, retry_context,
    handle_validation_error, handle_security_error, handle_database_error,
    handle_network_error, handle_file_system_error,
    ErrorRecoveryStrategy, RetryStrategy, FallbackStrategy, CircuitBreakerStrategy,
    ErrorMonitor, ErrorHandlerRegistry
)

from .validation import (
    validate_string, validate_integer, validate_float, validate_boolean,
    validate_email_address, validate_ip_address, validate_domain_name,
    validate_url, validate_port_number, validate_password_strength,
    validate_host_port, validate_network_range, validate_file_info,
    validate_multiple_fields, validate_input,
    validate_api_key, validate_jwt_token, validate_uuid,
    create_error_response, create_success_response, create_paginated_response
)

from .guard_clauses import (
    is_none_or_empty, is_valid_string, is_valid_integer, is_valid_ip_address,
    is_valid_domain, is_valid_url, is_valid_port,
    validate_scan_target_early, validate_scan_configuration_early,
    validate_dns_enumeration_early, validate_smb_enumeration_early,
    validate_brute_force_attack_early, validate_exploitation_attack_early,
    validate_authentication_early, validate_authorization_early,
    validate_database_connection_early, validate_database_query_early,
    validate_file_operation_early, validate_network_request_early,
    validate_configuration_early, with_guard_clauses
)

from .guard_clauses_advanced import (
    validate_scan_request_guard_clauses, validate_enumeration_request_guard_clauses,
    validate_attack_request_guard_clauses, validate_async_scan_request_guard_clauses,
    validate_authentication_guard_clauses, validate_authorization_guard_clauses,
    validate_database_connection_guard_clauses, validate_database_query_guard_clauses
)

from .early_returns import (
    early_return_if_none, early_return_if_empty, early_return_if_invalid_type,
    early_return_if_condition,
    validate_scan_inputs_early_return, execute_scan_with_early_returns,
    validate_enumeration_inputs_early_return, execute_enumeration_with_early_returns,
    validate_attack_inputs_early_return, execute_attack_with_early_returns,
    validate_security_inputs_early_return, check_authorization_with_early_returns,
    validate_database_inputs_early_return, execute_database_operation_with_early_returns
)

from .happy_path_patterns import (
    process_scan_request_happy_path_last, process_async_scan_request_happy_path_last,
    process_enumeration_request_happy_path_last, process_attack_request_happy_path_last,
    process_authentication_request_happy_path_last, process_authorization_request_happy_path_last,
    process_complex_scan_workflow_happy_path_last
)

from .anti_patterns_guide import (
    anti_pattern_nested_conditionals, anti_pattern_happy_path_first,
    anti_pattern_mixed_responsibilities, anti_pattern_early_returns_scattered,
    refactored_guard_clauses_pattern, refactored_separate_validation_pattern,
    refactored_decorator_pattern, validate_scan_decorator,
    compare_patterns, best_practices_summary
)

from .structured_logging import (
    StructuredLogger, get_logger, log_function_call, log_errors,
    LogContext, LogParameters, ErrorDetails, PerformanceMetrics, StructuredLogEntry,
    StructuredJSONFormatter, StructuredTextFormatter,
    ErrorTracker, PerformanceTracker,
    logging_context, performance_logging_context
)

from .logging_integration import (
    comprehensive_logging,
    ScanningLogger, EnumerationLogger, AttackLogger, SecurityLogger,
    _validate_function_inputs, _apply_guard_clauses, _check_early_returns,
    _check_async_early_returns
)

from .cli_exceptions import (
    CLIMessage, CLISeverity, CLICategory,
    CLITimeoutError, CLIInvalidTargetError, CLIInvalidPortError,
    CLIInvalidScanTypeError, CLINetworkConnectionError,
    CLIAuthenticationError, CLIAuthorizationError,
    CLIConfigurationError, CLIResourceError, CLISystemError,
    CLIValidationError,
    CLIMessageRenderer, CLIExceptionHandler, CLIExceptionContext
)

from .api_exceptions import (
    APIMessage, APISeverity, APICategory,
    APITimeoutError, APIInvalidTargetError, APIInvalidPortError,
    APIInvalidScanTypeError, APINetworkConnectionError,
    APIAuthenticationError, APIAuthorizationError,
    APIConfigurationError, APIResourceError, APISystemError,
    APIValidationError, APIRateLimitError, APIResourceNotFoundError,
    APIInvalidRequestError,
    APIMessageSerializer, APIExceptionHandler,
    create_fastapi_exception_handler
)

from .exception_mapper import (
    OutputFormat, ExceptionMapping, ExceptionMapper,
    ExceptionMapperContext, map_exceptions, map_async_exceptions,
    create_exception_mapper, map_video_opusclip_exception
)

__all__ = [
    # Custom Exceptions
    'VideoOpusClipException', 'ValidationError', 'InputValidationError', 'SchemaValidationError', 'TypeValidationError',
    'SecurityError', 'AuthenticationError', 'AuthorizationError', 'RateLimitError', 'IntrusionDetectionError',
    'ScanningError', 'PortScanError', 'VulnerabilityScanError', 'WebScanError',
    'EnumerationError', 'DNSEnumerationError', 'SMBEnumerationError', 'SSHEnumerationError',
    'AttackError', 'BruteForceError', 'ExploitationError',
    'DatabaseError', 'ConnectionError', 'QueryError', 'TransactionError',
    'NetworkError', 'ConnectionTimeoutError', 'DNSResolutionError', 'HTTPError',
    'FileSystemError', 'FileNotFoundError', 'FilePermissionError', 'FileSizeError',
    'ConfigurationError', 'MissingConfigurationError', 'InvalidConfigurationError',
    
    # Error Handlers
    'setup_error_logging', 'create_error_response', 'create_user_friendly_error_response',
    'handle_errors', 'handle_specific_errors', 'error_context', 'retry_context',
    'handle_validation_error', 'handle_security_error', 'handle_database_error',
    'handle_network_error', 'handle_file_system_error',
    'ErrorRecoveryStrategy', 'RetryStrategy', 'FallbackStrategy', 'CircuitBreakerStrategy',
    'ErrorMonitor', 'ErrorHandlerRegistry',
    
    # Validation
    'validate_string', 'validate_integer', 'validate_float', 'validate_boolean',
    'validate_email_address', 'validate_ip_address', 'validate_domain_name',
    'validate_url', 'validate_port_number', 'validate_password_strength',
    'validate_host_port', 'validate_network_range', 'validate_file_info',
    'validate_multiple_fields', 'validate_input',
    'validate_api_key', 'validate_jwt_token', 'validate_uuid',
    'create_error_response', 'create_success_response', 'create_paginated_response',
    
    # Guard Clauses
    'is_none_or_empty', 'is_valid_string', 'is_valid_integer', 'is_valid_ip_address',
    'is_valid_domain', 'is_valid_url', 'is_valid_port',
    'validate_scan_target_early', 'validate_scan_configuration_early',
    'validate_dns_enumeration_early', 'validate_smb_enumeration_early',
    'validate_brute_force_attack_early', 'validate_exploitation_attack_early',
    'validate_authentication_early', 'validate_authorization_early',
    'validate_database_connection_early', 'validate_database_query_early',
    'validate_file_operation_early', 'validate_network_request_early',
    'validate_configuration_early', 'with_guard_clauses',
    
    # Advanced Guard Clauses
    'validate_scan_request_guard_clauses', 'validate_enumeration_request_guard_clauses',
    'validate_attack_request_guard_clauses', 'validate_async_scan_request_guard_clauses',
    'validate_authentication_guard_clauses', 'validate_authorization_guard_clauses',
    'validate_database_connection_guard_clauses', 'validate_database_query_guard_clauses',
    
    # Early Returns
    'early_return_if_none', 'early_return_if_empty', 'early_return_if_invalid_type',
    'early_return_if_condition',
    'validate_scan_inputs_early_return', 'execute_scan_with_early_returns',
    'validate_enumeration_inputs_early_return', 'execute_enumeration_with_early_returns',
    'validate_attack_inputs_early_return', 'execute_attack_with_early_returns',
    'validate_security_inputs_early_return', 'check_authorization_with_early_returns',
    'validate_database_inputs_early_return', 'execute_database_operation_with_early_returns',
    
    # Happy Path Patterns
    'process_scan_request_happy_path_last', 'process_async_scan_request_happy_path_last',
    'process_enumeration_request_happy_path_last', 'process_attack_request_happy_path_last',
    'process_authentication_request_happy_path_last', 'process_authorization_request_happy_path_last',
    'process_complex_scan_workflow_happy_path_last',
    
    # Anti-Patterns Guide
    'anti_pattern_nested_conditionals', 'anti_pattern_happy_path_first',
    'anti_pattern_mixed_responsibilities', 'anti_pattern_early_returns_scattered',
    'refactored_guard_clauses_pattern', 'refactored_separate_validation_pattern',
    'refactored_decorator_pattern', 'validate_scan_decorator',
    'compare_patterns', 'best_practices_summary',
    
    # Structured Logging
    'StructuredLogger', 'get_logger', 'log_function_call', 'log_errors',
    'LogContext', 'LogParameters', 'ErrorDetails', 'PerformanceMetrics', 'StructuredLogEntry',
    'StructuredJSONFormatter', 'StructuredTextFormatter',
    'ErrorTracker', 'PerformanceTracker',
    'logging_context', 'performance_logging_context',
    
    # Logging Integration
    'comprehensive_logging',
    'ScanningLogger', 'EnumerationLogger', 'AttackLogger', 'SecurityLogger',
    '_validate_function_inputs', '_apply_guard_clauses', '_check_early_returns',
    '_check_async_early_returns',
    
    # CLI Exceptions
    'CLIMessage', 'CLISeverity', 'CLICategory',
    'CLITimeoutError', 'CLIInvalidTargetError', 'CLIInvalidPortError',
    'CLIInvalidScanTypeError', 'CLINetworkConnectionError',
    'CLIAuthenticationError', 'CLIAuthorizationError',
    'CLIConfigurationError', 'CLIResourceError', 'CLISystemError',
    'CLIValidationError',
    'CLIMessageRenderer', 'CLIExceptionHandler', 'CLIExceptionContext',
    
    # API Exceptions
    'APIMessage', 'APISeverity', 'APICategory',
    'APITimeoutError', 'APIInvalidTargetError', 'APIInvalidPortError',
    'APIInvalidScanTypeError', 'APINetworkConnectionError',
    'APIAuthenticationError', 'APIAuthorizationError',
    'APIConfigurationError', 'APIResourceError', 'APISystemError',
    'APIValidationError', 'APIRateLimitError', 'APIResourceNotFoundError',
    'APIInvalidRequestError',
    'APIMessageSerializer', 'APIExceptionHandler',
    'create_fastapi_exception_handler',
    
    # Exception Mapper
    'OutputFormat', 'ExceptionMapping', 'ExceptionMapper',
    'ExceptionMapperContext', 'map_exceptions', 'map_async_exceptions',
    'create_exception_mapper', 'map_video_opusclip_exception'
]

# Error handling utilities
def create_comprehensive_error_handler(
    logger: Optional[logging.Logger] = None,
    include_traceback: bool = False,
    user_level: str = "user"
) -> Callable:
    """
    Create a comprehensive error handler that combines all error handling strategies
    
    Args:
        logger: Logger instance for error logging
        include_traceback: Whether to include stack trace
        user_level: User level for error messages
        
    Returns:
        Comprehensive error handler function
    """
    def handle_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive error handler"""
        context = context or {}
        
        # Log error if logger is provided
        if logger:
            logger.error(f"Error in {context.get('operation', 'unknown')}: {error}", exc_info=include_traceback)
        
        # Create appropriate error response based on user level
        if user_level == "developer":
            return create_error_response(error, include_traceback=True, include_details=True)
        elif user_level == "admin":
            return create_error_response(error, include_traceback=False, include_details=True)
        else:
            return create_user_friendly_error_response(error, user_level="user")
    
    return handle_error


def setup_defensive_programming(
    enable_guard_clauses: bool = True,
    enable_early_returns: bool = True,
    enable_validation: bool = True,
    enable_error_handling: bool = True,
    enable_structured_logging: bool = True,
    enable_exception_mapping: bool = True,
    enable_happy_path_patterns: bool = True,
    logger: Optional[StructuredLogger] = None
) -> Dict[str, Any]:
    """
    Setup comprehensive defensive programming for the application
    
    Args:
        enable_guard_clauses: Whether to enable guard clauses
        enable_early_returns: Whether to enable early returns
        enable_validation: Whether to enable validation
        enable_error_handling: Whether to enable error handling
        enable_structured_logging: Whether to enable structured logging
        enable_exception_mapping: Whether to enable exception mapping
        enable_happy_path_patterns: Whether to enable happy path patterns
        logger: Structured logger instance
        
    Returns:
        Configuration dictionary
    """
    config = {
        'guard_clauses_enabled': enable_guard_clauses,
        'early_returns_enabled': enable_early_returns,
        'validation_enabled': enable_validation,
        'error_handling_enabled': enable_error_handling,
        'structured_logging_enabled': enable_structured_logging,
        'exception_mapping_enabled': enable_exception_mapping,
        'happy_path_patterns_enabled': enable_happy_path_patterns,
        'logger': logger
    }
    
    # Setup structured logging
    if enable_structured_logging and logger is None:
        config['logger'] = get_logger(
            name="video_opusclip_defensive",
            log_file="logs/defensive.log",
            error_file="logs/defensive_errors.log",
            performance_file="logs/defensive_performance.log"
        )
    
    # Setup exception mapper
    if enable_exception_mapping:
        config['exception_mapper'] = create_exception_mapper(logger=config['logger'])
    
    # Setup error handler registry
    if enable_error_handling:
        registry = ErrorHandlerRegistry()
        
        # Register default handlers
        registry.register_handler(ValidationError, handle_validation_error)
        registry.register_handler(SecurityError, handle_security_error)
        registry.register_handler(DatabaseError, handle_database_error)
        registry.register_handler(NetworkError, handle_network_error)
        registry.register_handler(FileSystemError, handle_file_system_error)
        
        # Register recovery strategies
        registry.register_recovery_strategy(RetryStrategy(logger=config['logger']))
        registry.register_recovery_strategy(FallbackStrategy({
            'config_load': {'timeout': 30, 'retries': 3},
            'database_connection': {'timeout': 10, 'retries': 2}
        }, logger=config['logger']))
        
        config['error_registry'] = registry
    
    return config


def validate_input_with_defensive_programming(
    data: Dict[str, Any],
    validation_rules: Dict[str, Dict[str, Any]],
    config: Dict[str, Any]
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate input using defensive programming patterns
    
    Args:
        data: Data to validate
        validation_rules: Validation rules
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, validated_data, error_message)
    """
    logger = config.get('logger')
    
    # Early return if validation is disabled
    if not config.get('validation_enabled', True):
        if logger:
            logger.debug("Input validation disabled", data=data, tags=['validation', 'disabled'])
        return True, data, None
    
    # Early return if data is None
    if data is None:
        if logger:
            logger.warning("Data is None", tags=['validation', 'error', 'null_data'])
        return False, None, "Data is required"
    
    # Early return if data is not a dictionary
    if not isinstance(data, dict):
        if logger:
            logger.warning("Data is not a dictionary", data_type=type(data).__name__, tags=['validation', 'error', 'invalid_type'])
        return False, None, "Data must be a dictionary"
    
    try:
        # Use guard clauses if enabled
        if config.get('guard_clauses_enabled', True):
            if logger:
                logger.debug("Applying guard clauses", data=data, tags=['guard_clauses', 'start'])
            
            # Apply guard clause validation
            for field, rules in validation_rules.items():
                value = data.get(field)
                
                # Check if field is required
                if rules.get('required', False) and is_none_or_empty(value):
                    if logger:
                        logger.warning(f"Required field '{field}' is missing or empty", field=field, rules=rules, tags=['guard_clauses', 'error', 'missing_field'])
                    return False, None, f"Field '{field}' is required"
                
                # Check field type
                expected_type = rules.get('type', 'string')
                if value is not None:
                    if expected_type == 'string' and not is_valid_string(value, rules.get('min_length', 1), rules.get('max_length')):
                        if logger:
                            logger.warning(f"Field '{field}' is not a valid string", field=field, value=value, rules=rules, tags=['guard_clauses', 'error', 'invalid_string'])
                        return False, None, f"Field '{field}' must be a valid string"
                    elif expected_type == 'integer' and not is_valid_integer(value, rules.get('min_value'), rules.get('max_value')):
                        if logger:
                            logger.warning(f"Field '{field}' is not a valid integer", field=field, value=value, rules=rules, tags=['guard_clauses', 'error', 'invalid_integer'])
                        return False, None, f"Field '{field}' must be a valid integer"
                    elif expected_type == 'email' and not validate_email_address(value):
                        if logger:
                            logger.warning(f"Field '{field}' is not a valid email", field=field, value=value, tags=['guard_clauses', 'error', 'invalid_email'])
                        return False, None, f"Field '{field}' must be a valid email address"
            
            if logger:
                logger.debug("Guard clauses passed", data=data, tags=['guard_clauses', 'success'])
        
        # Use comprehensive validation if enabled
        if config.get('validation_enabled', True):
            if logger:
                logger.debug("Applying comprehensive validation", data=data, tags=['validation', 'start'])
            
            validated_data = validate_multiple_fields(data, validation_rules)
            
            if logger:
                logger.debug("Comprehensive validation passed", validated_data=validated_data, tags=['validation', 'success'])
            
            return True, validated_data, None
        
        return True, data, None
        
    except Exception as e:
        # Use error handling if enabled
        if config.get('error_handling_enabled', True) and logger:
            logger.error("Validation error", error=e, data=data, validation_rules=validation_rules, tags=['validation', 'error', 'exception'])
        
        return False, None, str(e)


def create_integrated_logging_decorator(
    logger: Optional[StructuredLogger] = None,
    enable_all_features: bool = True,
    **kwargs
) -> Callable:
    """
    Create an integrated logging decorator with all defensive programming features
    
    Args:
        logger: Structured logger instance
        enable_all_features: Whether to enable all features
        **kwargs: Additional configuration options
        
    Returns:
        Decorator function
    """
    if logger is None:
        logger = get_logger(
            name="video_opusclip_integrated",
            log_file="logs/integrated.log",
            error_file="logs/integrated_errors.log",
            performance_file="logs/integrated_performance.log"
        )
    
    if enable_all_features:
        return comprehensive_logging(
            logger=logger,
            log_parameters=True,
            log_performance=True,
            log_errors=True,
            validate_inputs=True,
            use_guard_clauses=True,
            use_early_returns=True,
            **kwargs
        )
    else:
        return comprehensive_logging(
            logger=logger,
            **kwargs
        )


def create_unified_exception_handler(
    output_format: OutputFormat = OutputFormat.CLI,
    logger: Optional[StructuredLogger] = None
) -> Callable:
    """
    Create a unified exception handler that works for both CLI and API
    
    Args:
        output_format: Desired output format
        logger: Structured logger instance
        
    Returns:
        Exception handler function
    """
    if logger is None:
        logger = get_logger(
            name="video_opusclip_unified",
            log_file="logs/unified.log",
            error_file="logs/unified_errors.log"
        )
    
    mapper = create_exception_mapper(logger=logger)
    
    def handle_exception(exception: Exception, context: Dict[str, Any] = None) -> Union[CLIMessage, APIMessage, Dict[str, Any]]:
        """Unified exception handler"""
        return mapper.handle_exception(exception, output_format, context, exit_on_error=False)
    
    return handle_exception


def create_happy_path_function_template(
    validation_function: Optional[Callable] = None,
    logger: Optional[StructuredLogger] = None
) -> Callable:
    """
    Create a template for functions that follow the happy path pattern
    
    Args:
        validation_function: Function to validate inputs
        logger: Structured logger instance
        
    Returns:
        Template function
    """
    def template_decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Guard clauses for validation
            if validation_function:
                try:
                    validation_function(*args, **kwargs)
                except Exception as e:
                    if logger:
                        logger.error(f"Validation failed for {func.__name__}", error=e, args=args, kwargs=kwargs, tags=['validation', 'error'])
                    raise
            
            # Happy path: Execute the function
            try:
                result = func(*args, **kwargs)
                if logger:
                    logger.info(f"Function {func.__name__} completed successfully", result=result, tags=['happy_path', 'success'])
                return result
            except Exception as e:
                if logger:
                    logger.error(f"Function {func.__name__} failed", error=e, args=args, kwargs=kwargs, tags=['happy_path', 'error'])
                raise
        
        return wrapper
    return template_decorator


# Example usage
def example_comprehensive_error_handling():
    """Example of comprehensive error handling with all features"""
    print("🛡️ Comprehensive Error Handling Example")
    
    # Setup defensive programming with all features
    config = setup_defensive_programming(
        enable_guard_clauses=True,
        enable_early_returns=True,
        enable_validation=True,
        enable_error_handling=True,
        enable_structured_logging=True,
        enable_exception_mapping=True,
        enable_happy_path_patterns=True
    )
    
    logger = config['logger']
    mapper = config['exception_mapper']
    
    # Define validation rules
    validation_rules = {
        'username': {'type': 'string', 'min_length': 3, 'max_length': 50, 'required': True},
        'email': {'type': 'email', 'required': True},
        'age': {'type': 'integer', 'min_value': 18, 'max_value': 100, 'required': True}
    }
    
    # Test data
    test_data = {
        'username': 'john_doe',
        'email': 'john@example.com',
        'age': 25
    }
    
    # Validate with defensive programming and structured logging
    with logger.context(request_id="test_123", user_id="test_user"):
        is_valid, validated_data, error = validate_input_with_defensive_programming(
            test_data, validation_rules, config
        )
        
        if is_valid:
            logger.info("Validation successful", validated_data=validated_data, tags=['validation', 'success'])
        else:
            logger.error("Validation failed", error=error, tags=['validation', 'error'])
    
    # Test exception mapping
    exceptions_to_test = [
        TimeoutError("Operation timed out"),
        FileNotFoundError("config.json"),
        ValueError("Invalid input"),
        VideoOpusClipException("Custom error", error_code="CUSTOM_ERROR")
    ]
    
    print("\n" + "="*60)
    print("EXCEPTION MAPPING EXAMPLES")
    print("="*60)
    
    for exception in exceptions_to_test:
        print(f"\n{'-'*40}")
        print(f"Exception: {type(exception).__name__}")
        
        # Test CLI mapping
        cli_message = mapper.map_exception(exception, OutputFormat.CLI)
        print(f"CLI Title: {cli_message.title}")
        print(f"CLI Exit Code: {cli_message.exit_code}")
        
        # Test API mapping
        api_message = mapper.map_exception(exception, OutputFormat.API)
        print(f"API Title: {api_message.title}")
        print(f"API HTTP Status: {api_message.http_status}")
    
    # Test happy path patterns
    print(f"\n{'-'*40}")
    print("HAPPY PATH PATTERNS")
    print(f"{'-'*40}")
    
    try:
        scan_results = process_scan_request_happy_path_last(
            target={'host': '192.168.1.100', 'port': 80},
            scan_type='port_scan',
            options={'timeout': 30}
        )
        print(f"✅ Happy path pattern successful: {scan_results['status']}")
    except Exception as e:
        print(f"❌ Happy path pattern failed: {e}")
    
    # Test unified exception handler
    print(f"\n{'-'*40}")
    print("UNIFIED EXCEPTION HANDLER")
    print(f"{'-'*40}")
    
    cli_handler = create_unified_exception_handler(OutputFormat.CLI, logger)
    api_handler = create_unified_exception_handler(OutputFormat.API, logger)
    
    test_exception = TimeoutError("Test timeout")
    
    cli_result = cli_handler(test_exception, {"operation": "test"})
    api_result = api_handler(test_exception, {"operation": "test"})
    
    print(f"CLI Handler Result: {cli_result.title}")
    print(f"API Handler Result: {api_result.title}")
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"\n📊 Comprehensive Statistics:")
    print(f"Total logs: {stats['total_logs']}")
    print(f"Error logs: {stats['error_logs']}")
    print(f"Performance logs: {stats['performance_logs']}")
    
    if stats['error_analysis']['total_errors'] > 0:
        print(f"Total errors: {stats['error_analysis']['total_errors']}")
    
    if stats['performance_analysis']['total_operations'] > 0:
        perf = stats['performance_analysis']
        print(f"Average execution time: {perf['execution_time']['avg']:.3f}s")


if __name__ == "__main__":
    import logging
    example_comprehensive_error_handling() 