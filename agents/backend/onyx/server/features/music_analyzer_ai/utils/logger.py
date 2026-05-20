"""
Advanced logging utilities for Music Analyzer AI
Enhanced with performance tracking, structured logging, and ML metrics
"""

import logging
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
import traceback

try:
    from config.settings import settings
except ImportError:
    class Settings:
        LOG_LEVEL = "INFO"
        LOG_FILE = None
    settings = Settings()


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for better log analysis"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
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
        
        return json.dumps(log_data, ensure_ascii=False)


class MusicAnalyzerLogger:
    """
    Advanced logger for Music Analyzer AI with:
    - Performance tracking
    - Structured logging
    - ML metrics logging
    - Request/response logging
    """
    
    def __init__(self, name: str = "music_analyzer_ai"):
        self.logger = logging.getLogger(name)
        self._setup_logger()
        self.metrics: Dict[str, Any] = {
            "requests": 0,
            "errors": 0,
            "total_time": 0.0,
            "avg_time": 0.0
        }
    
    def _setup_logger(self):
        """Setup logger with handlers"""
        if self.logger.handlers:
            return
        
        self.logger.setLevel(getattr(logging, settings.LOG_LEVEL, "INFO"))
        
        # Console handler with standard format
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with structured format (if configured)
        if settings.LOG_FILE:
            log_path = Path(settings.LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            structured_formatter = StructuredFormatter()
            file_handler.setFormatter(structured_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional extra fields"""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.info(message, extra=extra)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message"""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.error(message, exc_info=exc_info, extra=extra)
        self.metrics["errors"] += 1
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.warning(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.debug(message, extra=extra)
    
    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        **kwargs
    ):
        """Log API request with metrics"""
        self.metrics["requests"] += 1
        self.metrics["total_time"] += duration
        self.metrics["avg_time"] = (
            self.metrics["total_time"] / self.metrics["requests"]
        )
        
        self.info(
            f"API Request: {method} {endpoint} - {status_code}",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration=duration,
            **kwargs
        )
    
    def log_ml_inference(
        self,
        model_name: str,
        input_shape: tuple,
        output_shape: tuple,
        duration: float,
        **kwargs
    ):
        """Log ML model inference"""
        self.info(
            f"ML Inference: {model_name}",
            model_name=model_name,
            input_shape=input_shape,
            output_shape=output_shape,
            duration=duration,
            inference_type="ml",
            **kwargs
        )
    
    def log_audio_analysis(
        self,
        track_id: str,
        analysis_type: str,
        duration: float,
        **kwargs
    ):
        """Log audio analysis operation"""
        self.info(
            f"Audio Analysis: {analysis_type} for {track_id}",
            track_id=track_id,
            analysis_type=analysis_type,
            duration=duration,
            **kwargs
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logger metrics"""
        return self.metrics.copy()


def timing_decorator(logger: Optional[MusicAnalyzerLogger] = None):
    """Decorator to measure function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if logger:
                    logger.debug(
                        f"{func.__name__} executed in {duration:.4f}s",
                        function=func.__name__,
                        duration=duration
                    )
                return result
            except Exception as e:
                duration = time.time() - start_time
                if logger:
                    logger.error(
                        f"{func.__name__} failed after {duration:.4f}s: {str(e)}",
                        function=func.__name__,
                        duration=duration,
                        error=str(e)
                    )
                raise
        return wrapper
    return decorator


# Global logger instance
_logger_instance: Optional[MusicAnalyzerLogger] = None


def get_logger(name: str = "music_analyzer_ai") -> MusicAnalyzerLogger:
    """Get or create global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = MusicAnalyzerLogger(name)
    return _logger_instance


# Convenience function for backward compatibility
def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup standard logger (backward compatibility)"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL, "INFO"))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if settings.LOG_FILE:
            log_path = Path(settings.LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


# Export main logger
logger = get_logger()
