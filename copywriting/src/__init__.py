from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .core.engine import CopywritingEngine
from .api.app import create_app
from .config.settings import get_settings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Copywriting System - Main Package
================================

A high-performance, production-ready copywriting generation system
with advanced AI capabilities and optimization features.
"""

__version__ = "3.0.0"
__author__ = "Copywriting System Team"
__description__ = "Ultra-optimized copywriting generation system"


__all__ = [
    "CopywritingEngine",
    "create_app", 
    "get_settings"
] 