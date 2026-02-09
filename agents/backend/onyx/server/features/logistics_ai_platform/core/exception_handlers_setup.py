"""
Exception handlers configuration

This module provides functions to configure exception handlers.
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from utils.exceptions import LogisticsException
from utils.http import (
    logistics_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure all exception handlers"""
    app.add_exception_handler(LogisticsException, logistics_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

