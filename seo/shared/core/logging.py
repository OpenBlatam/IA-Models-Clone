from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import sys
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager
import structlog
from structlog.stdlib import LoggerFactory
import logging
import orjson
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Ultra-Optimized Logging v10
Production-ready logging with maximum performance
"""


# Ultra-fast imports


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
    enable_console: bool = True
):
    """Setup ultra-optimized logging"""
    
    # Configure structlog for maximum performance
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(serializer=orjson.dumps)
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if enable_console else None,
        level=getattr(logging, level.upper()),
        handlers=[
            logging.StreamHandler(sys.stdout) if enable_console else logging.NullHandler()
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get ultra-optimized logger"""
    return structlog.get_logger(name)


def log_startup(
    version: str,
    environment: str,
    host: str,
    port: int,
    workers: int,
    **kwargs
):
    """Log application startup"""
    logger = get_logger("startup")
    logger.info(
        "Application starting",
        version=version,
        environment=environment,
        host=host,
        port=port,
        workers=workers,
        **kwargs
    )


def log_shutdown(reason: str = "unknown", **kwargs):
    """Log application shutdown"""
    logger = get_logger("shutdown")
    logger.info(
        "Application shutting down",
        reason=reason,
        **kwargs
    )


@contextmanager
def log_performance(operation: str, **context):
    """Context manager for performance logging"""
    start_time = time.time()
    logger = get_logger("performance")
    
    try:
        yield
        elapsed = time.time() - start_time
        logger.info(
            f"{operation} completed",
            operation=operation,
            elapsed=elapsed,
            **context
        )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"{operation} failed",
            operation=operation,
            elapsed=elapsed,
            error=str(e),
            **context
        )
        raise


def log_request(
    method: str,
    url: str,
    status_code: int,
    elapsed: float,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None,
    **kwargs
):
    """Log HTTP request with performance metrics"""
    logger = get_logger("http")
    logger.info(
        "HTTP request",
        method=method,
        url=url,
        status_code=status_code,
        elapsed=elapsed,
        user_agent=user_agent,
        ip_address=ip_address,
        **kwargs
    )


def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """Log error with context"""
    logger = get_logger("error")
    logger.error(
        "Application error",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {},
        **kwargs
    )


def log_metrics(
    metrics: Dict[str, Any],
    **kwargs
):
    """Log performance metrics"""
    logger = get_logger("metrics")
    logger.info(
        "Performance metrics",
        metrics=metrics,
        **kwargs
    )


def log_cache(
    operation: str,
    key: str,
    hit: bool,
    elapsed: float,
    **kwargs
):
    """Log cache operations"""
    logger = get_logger("cache")
    logger.info(
        "Cache operation",
        operation=operation,
        key=key,
        hit=hit,
        elapsed=elapsed,
        **kwargs
    )


def log_seo_analysis(
    url: str,
    status_code: int,
    title_length: int,
    description_length: int,
    h1_count: int,
    h2_count: int,
    image_count: int,
    link_count: int,
    word_count: int,
    total_time: float,
    cached: bool = False,
    **kwargs
):
    """Log SEO analysis results"""
    logger = get_logger("seo")
    logger.info(
        "SEO analysis completed",
        url=url,
        status_code=status_code,
        title_length=title_length,
        description_length=description_length,
        h1_count=h1_count,
        h2_count=h2_count,
        image_count=image_count,
        link_count=link_count,
        word_count=word_count,
        total_time=total_time,
        cached=cached,
        **kwargs
    )


def log_batch_analysis(
    url_count: int,
    successful_count: int,
    failed_count: int,
    total_time: float,
    avg_time_per_url: float,
    **kwargs
):
    """Log batch analysis results"""
    logger = get_logger("batch")
    logger.info(
        "Batch analysis completed",
        url_count=url_count,
        successful_count=successful_count,
        failed_count=failed_count,
        total_time=total_time,
        avg_time_per_url=avg_time_per_url,
        success_rate=(successful_count / url_count * 100) if url_count > 0 else 0,
        **kwargs
    )


def log_health_check(
    component: str,
    status: str,
    elapsed: float,
    **kwargs
):
    """Log health check results"""
    logger = get_logger("health")
    logger.info(
        "Health check",
        component=component,
        status=status,
        elapsed=elapsed,
        **kwargs
    )


def log_system_metrics(
    cpu_percent: float,
    memory_percent: float,
    disk_usage_percent: float,
    active_connections: int,
    **kwargs
):
    """Log system metrics"""
    logger = get_logger("system")
    logger.info(
        "System metrics",
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        disk_usage_percent=disk_usage_percent,
        active_connections=active_connections,
        **kwargs
    )


# Performance logging decorator
def log_performance_decorator(operation_name: str):
    """Decorator for automatic performance logging"""
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            with log_performance(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Async performance logging decorator
def log_async_performance_decorator(operation_name: str):
    """Async decorator for automatic performance logging"""
    def decorator(func) -> Any:
        async def wrapper(*args, **kwargs) -> Any:
            with log_performance(operation_name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator 