"""
Schemas de responses para la API
"""

from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class ExtractContentResponse(BaseModel):
    """Response con información extraída de la página web"""
    success: bool
    url: str
    raw_data: Dict[str, Any]
    extracted_info: str
    processing_metadata: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    structured_data: Optional[Dict[str, Any]] = None
    links: Optional[list] = None
    images: Optional[list] = None
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Response de error"""
    success: bool = False
    error: str
    message: str


class BatchExtractResponse(BaseModel):
    """Response para extracción en batch"""
    success: bool
    total_urls: int
    successful: int
    failed: int
    results: Dict[str, Dict[str, Any]]
    message: Optional[str] = None

