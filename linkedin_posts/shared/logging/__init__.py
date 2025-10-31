from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
import sys
from typing import Optional
from loguru import logger
from typing import Any, List, Dict, Optional
import asyncio
"""
Shared Logging - LinkedIn Posts
==============================

This module provides logging functionality for the LinkedIn Posts system.
"""




def get_logger(name: str) -> logger:
    """Get a logger instance for the given name."""
    return logger.bind(module=name)


def setup_logging(
    level: str = "INFO",
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    file_path: Optional[str] = None,
) -> None:
    """Setup logging configuration."""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=format,
        level=level,
        colorize=True,
    )
    
    # Add file handler if specified
    if file_path:
        logger.add(
            file_path,
            format=format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )
    
    # Intercept standard logging
    class InterceptHandler(logging.Handler):
        def emit(self, record) -> Any:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True) 