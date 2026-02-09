"""
Logger Utilities for Piel Mejorador AI SAM3
==========================================

Unified logger initialization and utilities.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


class LoggerUtils:
    """Unified logger utilities."""
    
    _loggers: dict[str, logging.Logger] = {}
    _default_level = logging.INFO
    
    @staticmethod
    def get_logger(
        name: str,
        level: Optional[int] = None,
        propagate: bool = False
    ) -> logging.Logger:
        """
        Get or create logger with consistent configuration.
        
        Args:
            name: Logger name (typically __name__)
            level: Optional logging level
            propagate: Whether to propagate to parent loggers
            
        Returns:
            Configured logger
        """
        if name in LoggerUtils._loggers:
            return LoggerUtils._loggers[name]
        
        logger = logging.getLogger(name)
        
        if level is None:
            level = LoggerUtils._default_level
        
        logger.setLevel(level)
        logger.propagate = propagate
        
        # Only add handler if logger doesn't have one
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        LoggerUtils._loggers[name] = logger
        return logger
    
    @staticmethod
    def set_default_level(level: int) -> None:
        """
        Set default logging level for new loggers.
        
        Args:
            level: Logging level
        """
        LoggerUtils._default_level = level
    
    @staticmethod
    def configure_logger(
        logger: logging.Logger,
        level: int = logging.INFO,
        format_string: Optional[str] = None,
        file_path: Optional[Path] = None
    ) -> None:
        """
        Configure logger with custom settings.
        
        Args:
            logger: Logger to configure
            level: Logging level
            format_string: Optional custom format string
            file_path: Optional file path for file handler
        """
        logger.setLevel(level)
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if format_string:
            formatter = logging.Formatter(format_string)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if file_path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.propagate = False


# Convenience function
def get_logger(name: str, **kwargs) -> logging.Logger:
    """Get or create logger."""
    return LoggerUtils.get_logger(name, **kwargs)




