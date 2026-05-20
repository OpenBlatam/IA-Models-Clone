"""
API Request/Response Models
===========================

Pydantic models for API requests and responses.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class EnhanceImageRequest(BaseModel):
    """Request model for image enhancement."""
    file_path: str
    enhancement_type: str = "general"
    options: Optional[Dict[str, Any]] = None
    priority: int = 0


class EnhanceVideoRequest(BaseModel):
    """Request model for video enhancement."""
    file_path: str
    enhancement_type: str = "general"
    options: Optional[Dict[str, Any]] = None
    priority: int = 0


class UpscaleRequest(BaseModel):
    """Request model for upscaling."""
    file_path: str
    scale_factor: int = 2
    options: Optional[Dict[str, Any]] = None
    priority: int = 0


class DenoiseRequest(BaseModel):
    """Request model for denoising."""
    file_path: str
    noise_level: str = "medium"
    options: Optional[Dict[str, Any]] = None
    priority: int = 0


class RestoreRequest(BaseModel):
    """Request model for restoration."""
    file_path: str
    damage_type: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    priority: int = 0


class ColorCorrectionRequest(BaseModel):
    """Request model for color correction."""
    file_path: str
    correction_type: str = "auto"
    options: Optional[Dict[str, Any]] = None
    priority: int = 0


class BatchProcessRequest(BaseModel):
    """Request model for batch processing."""
    items: List[Dict[str, Any]]
    progress_callback_url: Optional[str] = None


class WebhookRegisterRequest(BaseModel):
    """Request model for webhook registration."""
    url: str
    events: List[str]
    secret: Optional[str] = None
    timeout: float = 10.0
    retries: int = 3
    enabled: bool = True


class AnalyzeRequest(BaseModel):
    """Request model for file analysis."""
    file_path: str
    file_type: Optional[str] = None


class ExportResultsRequest(BaseModel):
    """Request model for exporting results."""
    task_ids: Optional[List[str]] = None
    format: str = "json"
    output_path: Optional[str] = None
    compress: bool = False




