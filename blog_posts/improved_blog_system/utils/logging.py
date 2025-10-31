"""
Logging configuration and utilities
"""

import structlog
import logging
from typing import Any, Dict


def configure_logging(log_level: str = "INFO", log_format: str = "json"):
    """Configure structured logging."""
    
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))


def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger."""
    return structlog.get_logger(name)






























