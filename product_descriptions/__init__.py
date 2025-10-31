from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .core.model import ProductDescriptionModel
from .core.generator import ProductDescriptionGenerator
from .api.service import ProductDescriptionService
from .interfaces.api_interface import ProductDescriptionAPI
from typing import Any, List, Dict, Optional
import logging
import asyncio
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

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__description__ = "AI Product Descriptions Generator with Transformers"


__all__ = [
    "ProductDescriptionModel",
    "ProductDescriptionGenerator", 
    "ProductDescriptionService",
    "ProductDescriptionAPI"
] 