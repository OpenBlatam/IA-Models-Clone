from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .app import create_app
from .core.config import get_settings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Refactored Product API - Clean Architecture
==========================================

Professional enterprise API with proper separation of concerns,
clean code principles, and maintainable structure.
"""

__version__ = "2.0.0"
__author__ = "Enterprise Development Team"


__all__ = ["create_app", "get_settings"] 