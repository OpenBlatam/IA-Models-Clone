"""
BUL Core Module
===============

Core functionality for the BUL (Business Universal Language) system.
"""

from .bul_engine import (
    BULEngine,
    DocumentRequest,
    DocumentResponse,
    BusinessArea,
    DocumentType,
    get_global_bul_engine
)

__all__ = [
    "BULEngine",
    "DocumentRequest", 
    "DocumentResponse",
    "BusinessArea",
    "DocumentType",
    "get_global_bul_engine"
]
























