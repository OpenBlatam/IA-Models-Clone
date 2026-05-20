"""
Utilities Module
================

Consolidated utilities exports.
"""

# Error handling
from .error_handlers import (
    ValidationError,
    FileValidationError,
    EnhancementError,
    handle_error
)

from .error_context import (
    ErrorContext,
    get_error_context,
    set_error_context
)

from .error_reporter import (
    ErrorReporter,
    get_error_reporter
)

# Validation
from .validators import (
    ParameterValidator,
    FileValidator
)

from .validation_helpers import (
    ValidationRule,
    ValidationChain,
    is_positive,
    is_valid_email,
    is_valid_url
)

from .advanced_validators import (
    validate_file_integrity,
    validate_url,
    validate_email,
    validate_config_structure
)

from .schema_validator import (
    SchemaValidator
)

# File operations
from .file_helpers import (
    get_mime_type,
    generate_unique_filename,
    save_uploaded_file
)

from .file_operations import (
    save_file,
    move_file,
    delete_file,
    ensure_file_exists,
    get_file_size
)

# Data transformation
from .transformers import (
    transform_dict,
    transform_list,
    transform_datetime
)

from .serialization import (
    serialize_json,
    deserialize_json,
    serialize_pickle,
    deserialize_pickle,
    encode_base64,
    decode_base64
)

# Formatting
from .formatters import (
    DataFormatter,
    StringFormatter
)

# Response building
from .response_builder import (
    ResponseBuilder,
    create_success_response,
    create_error_response
)

# Configuration
from .config_loader import (
    ConfigLoader,
    load_config
)

from .config_validator import (
    ConfigValidator,
    validate_config
)

# Development
from .dev_helpers import (
    time_function,
    log_function_call,
    debug_info
)

from .profiler import (
    PerformanceProfiler,
    profile_function,
    profile_async_function,
    TimingContext
)

# Testing
from .test_helpers import (
    AsyncTestMixin,
    temp_directory,
    temp_file,
    create_mock_client,
    create_mock_task,
    AssertionHelpers
)

# Logging
from .logger_utils import (
    StructuredLogger,
    PerformanceLogger,
    setup_structured_logging
)

# Documentation
from .doc_helpers import (
    DocGenerator,
    generate_module_docs
)

# Type checking
from .type_checker import (
    TypeChecker,
    check_type
)

# Optimization
from .optimization import (
    cache_result,
    throttle
)

# Performance
from .performance import (
    PerformanceMonitor
)

# Memory
from .memory_optimizer import (
    MemoryOptimizer
)

# Export
from .export import (
    export_to_json,
    export_to_markdown,
    export_to_csv,
    export_to_html
)

# Compression
from .compression import (
    compress_gzip,
    decompress_gzip,
    compress_file
)

# Backup
from .backup import (
    BackupManager
)

# Integration
from .integration import (
    format_webhook_payload,
    format_api_response,
    transform_result,
    validate_webhook_signature
)

# Image utilities
from .image_utils import (
    get_image_info,
    validate_image,
    estimate_processing_time
)

# Helpers
from .helpers import (
    generate_id,
    hash_file,
    format_size,
    parse_json,
    save_json,
    load_json,
    ensure_directory_exists,
    chunk_list,
    retry_decorator
)

__all__ = [
    # Error handling
    "ValidationError",
    "FileValidationError",
    "EnhancementError",
    "handle_error",
    "ErrorContext",
    "get_error_context",
    "set_error_context",
    "ErrorReporter",
    "get_error_reporter",
    # Validation
    "ParameterValidator",
    "FileValidator",
    "ValidationRule",
    "ValidationChain",
    "is_positive",
    "is_valid_email",
    "is_valid_url",
    "validate_file_integrity",
    "validate_url",
    "validate_email",
    "validate_config_structure",
    "SchemaValidator",
    # File operations
    "get_mime_type",
    "generate_unique_filename",
    "save_uploaded_file",
    "save_file",
    "move_file",
    "delete_file",
    "ensure_file_exists",
    "get_file_size",
    # Data transformation
    "transform_dict",
    "transform_list",
    "transform_datetime",
    "serialize_json",
    "deserialize_json",
    "serialize_pickle",
    "deserialize_pickle",
    "encode_base64",
    "decode_base64",
    # Formatting
    "DataFormatter",
    "StringFormatter",
    # Response building
    "ResponseBuilder",
    "create_success_response",
    "create_error_response",
    # Configuration
    "ConfigLoader",
    "load_config",
    "ConfigValidator",
    "validate_config",
    # Development
    "time_function",
    "log_function_call",
    "debug_info",
    "PerformanceProfiler",
    "profile_function",
    "profile_async_function",
    "TimingContext",
    # Testing
    "AsyncTestMixin",
    "temp_directory",
    "temp_file",
    "create_mock_client",
    "create_mock_task",
    "AssertionHelpers",
    # Logging
    "StructuredLogger",
    "PerformanceLogger",
    "setup_structured_logging",
    # Documentation
    "DocGenerator",
    "generate_module_docs",
    # Type checking
    "TypeChecker",
    "check_type",
    # Optimization
    "cache_result",
    "throttle",
    # Performance
    "PerformanceMonitor",
    # Memory
    "MemoryOptimizer",
    # Export
    "export_to_json",
    "export_to_markdown",
    "export_to_csv",
    "export_to_html",
    # Compression
    "compress_gzip",
    "decompress_gzip",
    "compress_file",
    # Backup
    "BackupManager",
    # Integration
    "format_webhook_payload",
    "format_api_response",
    "transform_result",
    "validate_webhook_signature",
    # Image utilities
    "get_image_info",
    "validate_image",
    "estimate_processing_time",
    # Helpers
    "generate_id",
    "hash_file",
    "format_size",
    "parse_json",
    "save_json",
    "load_json",
    "ensure_directory_exists",
    "chunk_list",
    "retry_decorator",
    # Advanced utilities
    "DataTransformer",
    # Advanced logging
    "StructuredLogger",
    "PerformanceLogger",
    "setup_logging",
    "LogLevel",
    # Advanced monitoring
    "SystemMonitor",
    "HealthMonitor",
    "SystemMetrics",
    "HealthCheck",
    "CacheUtils",
    "CompressionUtils",
    "CompressionType",
    "EncryptionUtils",
    "FileUtilsAdvanced",
    "NetworkUtils",
]

# Advanced utilities
from .data_transformer import DataTransformer

# Advanced logging
from .logging_advanced import StructuredLogger, PerformanceLogger, setup_logging, LogLevel

# Advanced monitoring
from .monitoring_advanced import SystemMonitor, HealthMonitor, SystemMetrics, HealthCheck
from .cache_utils import CacheUtils
from .compression_utils import CompressionUtils, CompressionType
from .encryption_utils import EncryptionUtils
from .file_utils_advanced import FileUtilsAdvanced
from .network_utils import NetworkUtils
