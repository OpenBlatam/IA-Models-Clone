"""
SEO Feature Module
==================

AI-powered SEO optimization and analysis system.
"""

from typing import Any, List, Dict, Optional, Union, Tuple
from typing_extensions import Literal, TypedDict
import logging
import asyncio

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered SEO optimization and analysis system"

# Try to import router with error handling
try:
    from .api import router as seo_router
except ImportError:
    seo_router = None

__all__ = [
    'seo_router'
] 