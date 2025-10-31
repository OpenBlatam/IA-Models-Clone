"""
BUL Utils Module
================

Modern utilities for the BUL system.
"""

from .cache_manager import (
    get_cache_manager,
    cached,
    CacheManager
)

from .modern_logging import (
    get_modern_logger,
    get_logger,
    log_performance,
    log_api_call,
    log_document_generation,
    log_security_event,
    LogContext,
    log_function_calls,
    log_async_function_calls
)

from .data_processor import (
    get_data_processor,
    get_async_data_processor,
    ModernDataProcessor,
    AsyncDataProcessor,
    DocumentMetrics
)

from .performance_optimizer import (
    get_performance_monitor,
    monitor_performance,
    OptimizedHTTPClient,
    batch_process,
    optimize_json_serialization,
    create_performance_report
)

__all__ = [
    # Cache
    "get_cache_manager",
    "cached",
    "CacheManager",
    
    # Logging
    "get_modern_logger",
    "get_logger",
    "log_performance",
    "log_api_call",
    "log_document_generation",
    "log_security_event",
    "LogContext",
    "log_function_calls",
    "log_async_function_calls",
    
    # Data Processing
    "get_data_processor",
    "get_async_data_processor",
    "ModernDataProcessor",
    "AsyncDataProcessor",
    "DocumentMetrics",
    
    # Performance
    "get_performance_monitor",
    "monitor_performance",
    "OptimizedHTTPClient",
    "batch_process",
    "optimize_json_serialization",
    "create_performance_report"
]