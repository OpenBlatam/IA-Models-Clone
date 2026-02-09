"""
Infrastructure Layer - External Concerns

This layer contains implementations of repositories, adapters, and external services
that interact with databases, file systems, ML models, and hardware.
"""

from .repositories import (
    InspectionRepository,
    ModelRepository,
    ConfigurationRepository,
)
from .adapters import (
    CameraAdapter,
    MLModelLoader,
    StorageAdapter,
)
from .ml_services import (
    AnomalyDetectionService,
    ObjectDetectionService,
    DefectClassificationService,
)

__all__ = [
    # Repositories
    "InspectionRepository",
    "ModelRepository",
    "ConfigurationRepository",
    # Adapters
    "CameraAdapter",
    "MLModelLoader",
    "StorageAdapter",
    # ML Services
    "AnomalyDetectionService",
    "ObjectDetectionService",
    "DefectClassificationService",
]



