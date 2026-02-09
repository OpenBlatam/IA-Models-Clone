"""
Base Models
Base response models and common structures
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: str = "Success"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error: str
    status_code: int
    details: Optional[Dict[str, Any]] = None







