"""
Logging configuration for audio separator.
Refactored to use constants and improve organization.
"""

import logging
import sys
from typing import Optional

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_LOGGER_NAME = "audio_separator"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(message)s"
)

# ════════════════════════════════════════════════════════════════════════════
# LOGGER SETUP
# ════════════════════════════════════════════════════════════════════════════

def setup_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: int = DEFAULT_LOG_LEVEL,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger for audio separator.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = DEFAULT_LOG_FORMAT
    
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# Default logger instance
logger = setup_logger()
