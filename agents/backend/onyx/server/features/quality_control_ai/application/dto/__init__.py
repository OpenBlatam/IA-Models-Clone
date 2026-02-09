"""
Data Transfer Objects (DTOs)

DTOs are used to transfer data between layers without exposing domain entities.
"""

from .inspection_request import InspectionRequest
from .inspection_response import InspectionResponse
from .defect_dto import DefectDTO
from .anomaly_dto import AnomalyDTO
from .quality_metrics_dto import QualityMetricsDTO
from .batch_inspection_request import BatchInspectionRequest
from .batch_inspection_response import BatchInspectionResponse

__all__ = [
    "InspectionRequest",
    "InspectionResponse",
    "DefectDTO",
    "AnomalyDTO",
    "QualityMetricsDTO",
    "BatchInspectionRequest",
    "BatchInspectionResponse",
]



