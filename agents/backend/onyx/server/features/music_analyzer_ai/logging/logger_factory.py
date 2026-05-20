"""
Logger Factory
Create loggers with different configurations
"""

from typing import Optional, Dict, Any
import logging
import sys
from pathlib import Path


class LoggerFactory:
    """Factory for creating loggers"""
    
    @staticmethod
    def create(
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
        console: bool = True
    ) -> logging.Logger:
        """
        Create logger
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            format_string: Optional format string
            console: Whether to log to console
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Default format
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


def create_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """Convenience function for creating loggers"""
    return LoggerFactory.create(name, level, log_file, format_string, console)



