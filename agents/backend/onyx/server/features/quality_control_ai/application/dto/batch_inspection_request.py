"""
Batch Inspection Request DTO

Data transfer object for batch inspection requests.
"""

from dataclasses import dataclass
from typing import List, Optional
from .inspection_request import InspectionRequest


@dataclass
class BatchInspectionRequest:
    """
    Request DTO for batch image inspection.
    
    Attributes:
        images: List of InspectionRequest objects
        batch_size: Optional batch size for processing
        parallel: Whether to process in parallel
        max_workers: Maximum number of worker threads/processes
    """
    
    images: List[InspectionRequest]
    batch_size: Optional[int] = None
    parallel: bool = True
    max_workers: Optional[int] = None
    
    def __post_init__(self):
        """Validate request."""
        if not self.images:
            raise ValueError("images list cannot be empty")
        
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_workers is not None and self.max_workers <= 0:
            raise ValueError("max_workers must be positive")



