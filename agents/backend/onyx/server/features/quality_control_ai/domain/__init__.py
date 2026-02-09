"""
Domain Layer - Core Business Logic

This module contains the core business entities, value objects, and domain services
that represent the business logic of the Quality Control AI system.
"""

from .entities import (
    Inspection,
    Defect,
    Anomaly,
    QualityScore,
    Camera,
)
from .value_objects import (
    ImageMetadata,
    DetectionResult,
    QualityMetrics,
)
from .services import (
    InspectionService,
    QualityAssessmentService,
    DefectClassificationService,
)
from .exceptions import (
    QualityControlException,
    InspectionException,
    ModelException,
    CameraException,
    ConfigurationException,
)

__all__ = [
    # Entities
    "Inspection",
    "Defect",
    "Anomaly",
    "QualityScore",
    "Camera",
    # Value Objects
    "ImageMetadata",
    "DetectionResult",
    "QualityMetrics",
    # Services
    "InspectionService",
    "QualityAssessmentService",
    "DefectClassificationService",
    # Exceptions
    "QualityControlException",
    "InspectionException",
    "ModelException",
    "CameraException",
    "ConfigurationException",
]



