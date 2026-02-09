"""
Inspection Response DTO

Data transfer object for inspection responses.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
from datetime import datetime

from .defect_dto import DefectDTO
from .anomaly_dto import AnomalyDTO
from .quality_metrics_dto import QualityMetricsDTO


@dataclass
class InspectionResponse:
    """
    Response DTO for image inspection.
    
    Attributes:
        inspection_id: Unique identifier for the inspection
        quality_score: Quality score (0.0 to 100.0)
        quality_status: Quality status (excellent, good, acceptable, poor, rejected)
        defects: List of detected defects
        anomalies: List of detected anomalies
        is_acceptable: Whether quality is acceptable
        recommendation: Recommendation based on quality
        inference_time_ms: Time taken for inference in milliseconds
        visualization: Optional visualization image (base64 encoded or bytes)
        metadata: Additional metadata
        created_at: Timestamp when inspection was created
    """
    
    inspection_id: str
    quality_score: float
    quality_status: str
    defects: List[DefectDTO]
    anomalies: List[AnomalyDTO]
    is_acceptable: bool
    recommendation: str
    inference_time_ms: Optional[float] = None
    visualization: Optional[Any] = None
    metadata: Optional[dict] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """
        Convert response to dictionary.
        
        Returns:
            Dictionary representation of the response
        """
        result = {
            "inspection_id": self.inspection_id,
            "quality_score": self.quality_score,
            "quality_status": self.quality_status,
            "defects": [defect.to_dict() for defect in self.defects],
            "anomalies": [anomaly.to_dict() for anomaly in self.anomalies],
            "is_acceptable": self.is_acceptable,
            "recommendation": self.recommendation,
            "inference_time_ms": self.inference_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        
        if self.visualization is not None:
            result["visualization"] = "base64_image_data"  # Placeholder
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result



