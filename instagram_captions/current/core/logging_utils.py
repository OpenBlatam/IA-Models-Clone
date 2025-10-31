"""
Logging Utilities for Instagram Captions API v10.0

Core logging functionality.
"""

import logging
from typing import Optional

def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """Setup logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("instagram_captions.log")
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)






