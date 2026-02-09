"""
Domain Entities

Core business entities that represent the main concepts in the Quality Control AI system.
"""

from .inspection import Inspection
from .defect import Defect, DefectType, DefectSeverity
from .anomaly import Anomaly, AnomalyType, AnomalySeverity
from .quality_score import QualityScore, QualityStatus
from .camera import Camera, CameraStatus

__all__ = [
    "Inspection",
    "Defect",
    "DefectType",
    "DefectSeverity",
    "Anomaly",
    "AnomalyType",
    "AnomalySeverity",
    "QualityScore",
    "QualityStatus",
    "Camera",
    "CameraStatus",
]



