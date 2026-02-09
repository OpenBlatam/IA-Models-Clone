"""Structured logging utilities"""
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime
from pathlib import Path
import sys


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger manager"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize structured logger
        
        Args:
            log_file: Optional log file path
        """
        self.log_file = log_file
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging"""
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Console handler with structured formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(console_handler)
        
        # File handler if log file specified
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(file_handler)
    
    def log_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None
    ):
        """
        Log HTTP request
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
            user_id: Optional user ID
        """
        logger = logging.getLogger("http")
        logger.info(
            "HTTP request",
            extra={
                "extra_fields": {
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "duration": duration,
                    "user_id": user_id
                }
            }
        )
    
    def log_conversion(
        self,
        format: str,
        status: str,
        duration: float,
        file_size: Optional[int] = None
    ):
        """
        Log conversion
        
        Args:
            format: Output format
            status: Conversion status
            duration: Conversion duration
            file_size: Optional output file size
        """
        logger = logging.getLogger("conversion")
        logger.info(
            "Document conversion",
            extra={
                "extra_fields": {
                    "format": format,
                    "status": status,
                    "duration": duration,
                    "file_size": file_size
                }
            }
        )
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log error
        
        Args:
            error_type: Error type
            error_message: Error message
            context: Optional context
        """
        logger = logging.getLogger("error")
        extra_fields = {
            "error_type": error_type,
            "error_message": error_message
        }
        if context:
            extra_fields.update(context)
        
        logger.error(
            "Error occurred",
            extra={"extra_fields": extra_fields}
        )


# Global structured logger
_structured_logger: Optional[StructuredLogger] = None


def get_structured_logger() -> StructuredLogger:
    """Get global structured logger"""
    global _structured_logger
    if _structured_logger is None:
        from config import settings
        log_file = getattr(settings, 'log_file', None)
        _structured_logger = StructuredLogger(log_file)
    return _structured_logger

