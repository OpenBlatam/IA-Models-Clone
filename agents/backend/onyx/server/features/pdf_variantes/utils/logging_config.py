"""
PDF Variantes Logging Configuration
Structured logging setup
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup structured logging"""
    
    # Create logs directory if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}',
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "detailed",
                "filename": log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console"] + (["file"] if log_file else []),
                "propagate": False
            },
            "pdf_variantes": {
                "level": log_level,
                "handlers": ["console"] + (["file"] if log_file else []),
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"] + (["file"] if log_file else []),
                "propagate": False
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console"] + (["file"] if log_file else []),
                "propagate": False
            }
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Set specific loggers
    logging.getLogger("pdf_variantes").setLevel(log_level)
    logging.getLogger("uvicorn").setLevel("INFO")
    logging.getLogger("fastapi").setLevel("INFO")
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(f"pdf_variantes.{name}")

class LoggerMixin:
    """Mixin class to add logging to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(self.__class__.__name__)
