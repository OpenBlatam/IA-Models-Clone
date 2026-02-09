"""
Logger factory for consistent logging across the application.

Provides a centralized way to create loggers with consistent configuration.
"""

import logging
from typing import Optional, Dict, Any


class LoggerFactory:
    """Factory for creating configured loggers."""
    
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        level: Optional[int] = None,
        format_string: Optional[str] = None
    ) -> logging.Logger:
        """
        Get or create a logger with consistent configuration.
        
        Args:
            name: Logger name (typically __name__)
            level: Logging level (default: INFO)
            format_string: Custom format string
        
        Returns:
            Configured logger
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        
        if level is not None:
            logger.setLevel(level)
        
        # Only add handler if not already configured
        if not logger.handlers:
            handler = logging.StreamHandler()
            
            if format_string:
                formatter = logging.Formatter(format_string)
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def configure_logging(
        cls,
        level: int = logging.INFO,
        format_string: Optional[str] = None
    ) -> None:
        """
        Configure root logger.
        
        Args:
            level: Logging level
            format_string: Format string
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            
            if format_string:
                formatter = logging.Formatter(format_string)
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Convenience function to get a logger."""
    return LoggerFactory.get_logger(name, level)

