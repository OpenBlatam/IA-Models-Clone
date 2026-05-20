"""
Key Messages Feature Module
============================

AI-powered key message generation and management system.
"""

from typing import Any, List, Dict, Optional, Union, Tuple
from typing_extensions import Literal, TypedDict
import logging
import asyncio

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered key message generation and management system"

# Try to import router with error handling
try:
    from .api import router as key_messages_router
except ImportError:
    key_messages_router = None

__all__ = [
    'key_messages_router'
] 