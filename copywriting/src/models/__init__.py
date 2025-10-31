from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .requests import CopywritingRequest, BatchRequest, OptimizationRequest
from .responses import CopywritingResponse, BatchResponse, SystemMetrics
from .entities import CopywritingVariant, PerformanceMetrics
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Models Package
=============

Data models and schemas for the copywriting system.
"""


__all__ = [
    "CopywritingRequest",
    "BatchRequest", 
    "OptimizationRequest",
    "CopywritingResponse",
    "BatchResponse",
    "SystemMetrics",
    "CopywritingVariant",
    "PerformanceMetrics"
] 