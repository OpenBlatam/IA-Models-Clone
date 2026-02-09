"""
Persona Feature Module
======================

AI-powered persona creation and management system.
"""

from typing import Any, List, Dict, Optional, Union, Tuple
from typing_extensions import Literal, TypedDict
import logging
import asyncio

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered persona creation and management system"

# Try to import main components with error handling
try:
    from .api import router as persona_router
except ImportError:
    persona_router = None

__all__ = [
    "persona_router",
]