"""
Export Models
Export related Pydantic models
"""

from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from .enums import OutputFormat
from .base import BaseResponse

class ExportRequest(BaseModel):
    """Export request model"""
    content: Dict[str, Any]
    output_format: OutputFormat
    theme: Optional[str] = None
    template: Optional[str] = None
    style: Optional[str] = None
    document_type: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)

class ExportResponse(BaseResponse):
    """Export response model"""
    export_id: str
    download_url: str
    file_size: int
    expires_at: datetime







