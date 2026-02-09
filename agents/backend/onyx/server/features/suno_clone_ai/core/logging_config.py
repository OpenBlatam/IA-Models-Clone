"""
Logging configuration utility
"""

from config.settings import settings
from utils.structured_logging import setup_structured_logging


def configure_logging() -> None:
    """
    Configure structured logging based on environment.
    
    Uses CloudWatch format for Lambda, JSON format for other environments.
    """
    if settings.is_lambda:
        setup_structured_logging(
            level="INFO" if not settings.debug else "DEBUG",
            format_type="cloudwatch",
            output="stdout"
        )
    else:
        setup_structured_logging(
            level="INFO" if not settings.debug else "DEBUG",
            format_type="json",
            output="stdout"
        )







