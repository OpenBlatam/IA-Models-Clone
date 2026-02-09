"""
Document Set Feature Module
============================

AI-powered document set management and processing system.
"""

from typing import Any, List, Dict, Optional, Union, Tuple
from typing_extensions import Literal, TypedDict
import logging
import asyncio

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered document set management and processing system"

# Try to import main components with error handling
try:
    from .api import router as document_set_router
except ImportError:
    document_set_router = None

__all__ = [
    "document_set_router",
]