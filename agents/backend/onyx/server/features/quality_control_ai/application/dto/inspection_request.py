"""
Inspection Request DTO

Data transfer object for inspection requests.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class InspectionRequest:
    """
    Request DTO for image inspection.
    
    Attributes:
        image_data: Image data (numpy array, bytes, or file path)
        image_format: Format of the image ('numpy', 'bytes', 'file_path')
        config_overrides: Optional configuration overrides
        include_visualization: Whether to include visualization in response
        timeout_seconds: Optional timeout for inspection
    """
    
    image_data: any  # Can be numpy array, bytes, or file path
    image_format: str = "numpy"  # 'numpy', 'bytes', 'file_path'
    config_overrides: Optional[dict] = None
    include_visualization: bool = False
    timeout_seconds: Optional[float] = None
    
    def __post_init__(self):
        """Validate request."""
        if self.image_format not in ['numpy', 'bytes', 'file_path']:
            raise ValueError(
                f"Invalid image_format: {self.image_format}. "
                "Must be 'numpy', 'bytes', or 'file_path'"
            )
        
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")



