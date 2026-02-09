"""
Application Layer - Use Cases and Application Services

This layer contains use cases (interactors) that orchestrate domain logic
and application services that coordinate multiple use cases.
"""

from .use_cases import (
    InspectImageUseCase,
    InspectBatchUseCase,
    StartInspectionStreamUseCase,
    StopInspectionStreamUseCase,
    TrainModelUseCase,
    GenerateReportUseCase,
)
from .dto import (
    InspectionRequest,
    InspectionResponse,
    DefectDTO,
    AnomalyDTO,
    QualityMetricsDTO,
    BatchInspectionRequest,
    BatchInspectionResponse,
)
from .services import (
    InspectionApplicationService,
    ModelTrainingApplicationService,
)

__all__ = [
    # Use Cases
    "InspectImageUseCase",
    "InspectBatchUseCase",
    "StartInspectionStreamUseCase",
    "StopInspectionStreamUseCase",
    "TrainModelUseCase",
    "GenerateReportUseCase",
    # DTOs
    "InspectionRequest",
    "InspectionResponse",
    "DefectDTO",
    "AnomalyDTO",
    "QualityMetricsDTO",
    "BatchInspectionRequest",
    "BatchInspectionResponse",
    # Application Services
    "InspectionApplicationService",
    "ModelTrainingApplicationService",
]



