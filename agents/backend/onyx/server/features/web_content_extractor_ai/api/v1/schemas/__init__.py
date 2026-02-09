"""
Schemas de la API v1
"""

from .requests import ExtractContentRequest, BatchExtractRequest, BaseExtractionRequest
from .responses import ExtractContentResponse, ErrorResponse, BatchExtractResponse
from . import constants

__all__ = [
    "ExtractContentRequest",
    "BatchExtractRequest",
    "BaseExtractionRequest",
    "ExtractContentResponse",
    "ErrorResponse",
    "BatchExtractResponse",
    "constants",
]

