"""
Structured Logger
Advanced structured logging
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class StructuredLogger:
    """Structured logger with JSON output"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.log_file = Path(log_file) if log_file else None
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
    
    def _log(self, level: str, message: str, **kwargs):
        """Log structured message"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        
        log_json = json.dumps(log_entry)
        
        if level == "DEBUG":
            self.logger.debug(log_json)
        elif level == "INFO":
            self.logger.info(log_json)
        elif level == "WARNING":
            self.logger.warning(log_json)
        elif level == "ERROR":
            self.logger.error(log_json)
        elif level == "CRITICAL":
            self.logger.critical(log_json)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log("CRITICAL", message, **kwargs)
    
    def log_video_generation(
        self,
        video_id: str,
        status: str,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        **kwargs
    ):
        """Log video generation event"""
        self.info(
            f"Video generation {status}",
            event_type="video_generation",
            video_id=video_id,
            status=status,
            duration=duration,
            error=error,
            **kwargs
        )
    
    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Log API request"""
        self.info(
            f"API {method} {path}",
            event_type="api_request",
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            user_id=user_id,
            **kwargs
        )


_structured_logger: Optional[StructuredLogger] = None


def get_structured_logger(name: str = "faceless_video_ai", log_file: Optional[str] = None) -> StructuredLogger:
    """Get structured logger instance"""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger(name, log_file)
    return _structured_logger

