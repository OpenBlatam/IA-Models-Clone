"""
Batch Inspection Response DTO

Data transfer object for batch inspection responses.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from .inspection_response import InspectionResponse
from .quality_metrics_dto import QualityMetricsDTO


@dataclass
class BatchInspectionResponse:
    """
    Response DTO for batch image inspection.
    
    Attributes:
        inspections: List of inspection responses
        total_processed: Total number of images processed
        total_succeeded: Number of successful inspections
        total_failed: Number of failed inspections
        average_quality_score: Average quality score across all inspections
        total_processing_time_ms: Total processing time in milliseconds
        quality_metrics: Aggregated quality metrics
        created_at: Timestamp when batch was processed
    """
    
    inspections: List[InspectionResponse]
    total_processed: int
    total_succeeded: int
    total_failed: int
    average_quality_score: float
    total_processing_time_ms: Optional[float] = None
    quality_metrics: Optional[QualityMetricsDTO] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """
        Convert batch response to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {
            "inspections": [inspection.to_dict() for inspection in self.inspections],
            "total_processed": self.total_processed,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "average_quality_score": self.average_quality_score,
            "total_processing_time_ms": self.total_processing_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        
        if self.quality_metrics:
            result["quality_metrics"] = self.quality_metrics.to_dict()
        
        return result



