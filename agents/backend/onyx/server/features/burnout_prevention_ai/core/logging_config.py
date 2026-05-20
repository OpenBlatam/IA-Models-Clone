"""
Logging Configuration
=====================
Structured logging configuration using structlog.

Provides JSON-formatted logs with context for better observability
and integration with log aggregation systems.
"""

import logging
import sys
from typing import Optional

try:
    import structlog
    from structlog.stdlib import LoggerFactory
    from pythonjsonlogger import jsonlogger
    
    def setup_structlog(level: str = "INFO"):
        """Setup structured logging with JSON output."""
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
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(jsonlogger.JsonFormatter())
        
        root_logger = logging.getLogger()
        root_logger.handlers = [handler]
        root_logger.setLevel(getattr(logging, level.upper()))
        
        return structlog.get_logger()
    
    def get_logger(name: Optional[str] = None):
        """Get structured logger."""
        return structlog.get_logger(name)
        
except ImportError:
    # Fallback to standard logging
    def setup_structlog(level: str = "INFO"):
        """Fallback to standard logging."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        return logging.getLogger()
    
    def get_logger(name: Optional[str] = None):
        """Get standard logger."""
        return logging.getLogger(name)

