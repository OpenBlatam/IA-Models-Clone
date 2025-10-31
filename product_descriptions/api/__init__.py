from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .service import ProductDescriptionService
from .gradio_interface import ProductDescriptionGradioApp
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API module for Product Descriptions Generator
=============================================

Contains web services, REST API endpoints, and external interfaces.
"""


__all__ = ["ProductDescriptionService", "ProductDescriptionGradioApp"] 