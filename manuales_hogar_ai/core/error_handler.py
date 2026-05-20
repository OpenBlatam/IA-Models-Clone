"""
Global Error Handler (Legacy)
=============================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar core.error_handlers.*
"""

from .error_handlers import (
    global_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    custom_exception_handler,
    database_exception_handler,
    httpx_exception_handler,
)

__all__ = [
    "global_exception_handler",
    "validation_exception_handler",
    "http_exception_handler",
    "custom_exception_handler",
    "database_exception_handler",
    "httpx_exception_handler",
]




