"""
OS Content Feature Module
==========================

AI-powered content generation and management system.
"""

from typing import Any, List, Dict, Optional, Union, Tuple
from typing_extensions import Literal, TypedDict
import logging
import asyncio

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered content generation and management system"

# Try to import router with error handling
try:
    from .api import router as os_content_router
except ImportError:
    os_content_router = None

__all__ = [
    'os_content_router'
]