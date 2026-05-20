"""
Error Handlers Module
====================

Módulo especializado para manejo de errores.
"""

from .global_handler import global_exception_handler
from .validation_handler import validation_exception_handler
from .http_handler import http_exception_handler
from .custom_handler import custom_exception_handler
from .database_handler import database_exception_handler
from .httpx_handler import httpx_exception_handler

__all__ = [
    "global_exception_handler",
    "validation_exception_handler",
    "http_exception_handler",
    "custom_exception_handler",
    "database_exception_handler",
    "httpx_exception_handler",
]

