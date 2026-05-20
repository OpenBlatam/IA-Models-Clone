"""Logging utilities for training scripts."""
import logging
import warnings

from rich.logging import RichHandler

DEFAULT_LOG_FILE = 'training.log'
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logging(log_file: str = DEFAULT_LOG_FILE, level: int = DEFAULT_LOG_LEVEL) -> None:
    """Setup logging configuration for training scripts."""
    warnings.filterwarnings("ignore")
    
    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(log_file)
        ]
    )


