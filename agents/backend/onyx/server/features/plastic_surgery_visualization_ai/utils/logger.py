"""Logging utilities with structured logging support."""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

from config.settings import settings


def setup_logging(log_level: str = "INFO", use_json: bool = False) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_json: Whether to use JSON formatting
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    if STRUCTLOG_AVAILABLE and use_json:
        _setup_structlog(level)
    else:
        _setup_standard_logging(level, use_json)


def _setup_structlog(level: int) -> None:
    """Setup structured logging with structlog."""
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
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level
    )


def _setup_standard_logging(level: int, use_json: bool) -> None:
    """Setup standard logging."""
    if use_json:
        format_string = "%(message)s"
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
    else:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler = logging.StreamHandler(sys.stdout)
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[handler]
    )


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)


def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration: float,
    **kwargs
) -> None:
    """Log HTTP request."""
    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration": round(duration, 3),
        **kwargs
    }
    
    if status_code >= 500:
        logger.error("request_error", **log_data)
    elif status_code >= 400:
        logger.warning("request_warning", **log_data)
    else:
        logger.info("request_completed", **log_data)

