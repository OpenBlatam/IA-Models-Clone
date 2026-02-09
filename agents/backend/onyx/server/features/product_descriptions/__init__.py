"""
Product Descriptions Generator Module
====================================

AI-powered product descriptions generator using advanced transformer models
and deep learning techniques for e-commerce and marketing applications.

Features:
- Multi-language product description generation
- SEO-optimized content creation
- Custom tone and style adaptation
- Batch processing capabilities
- Performance optimization with mixed precision
- Gradio interface for easy testing
"""

from typing import Any, List, Dict, Optional, Union, Tuple
from typing_extensions import Literal, TypedDict
import logging
import asyncio

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI Product Descriptions Generator with Transformers"

# Try to import components with error handling
try:
    from .core.model import ProductDescriptionModel
except ImportError:
    ProductDescriptionModel = None

try:
    from .core.generator import ProductDescriptionGenerator
except ImportError:
    ProductDescriptionGenerator = None

try:
    from .api.service import ProductDescriptionService
except ImportError:
    ProductDescriptionService = None

try:
    from .interfaces.api_interface import ProductDescriptionAPI
except ImportError:
    ProductDescriptionAPI = None

__all__ = [
    "ProductDescriptionModel",
    "ProductDescriptionGenerator", 
    "ProductDescriptionService",
    "ProductDescriptionAPI"
] 