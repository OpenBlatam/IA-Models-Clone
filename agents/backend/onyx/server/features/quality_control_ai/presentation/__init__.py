"""
Presentation Layer - API and WebSocket

This layer contains REST API endpoints, WebSocket handlers, and request/response models.
"""

from .api import (
    create_app,
    router as api_router,
)
from .schemas import (
    InspectionRequestSchema,
    InspectionResponseSchema,
    BatchInspectionRequestSchema,
    BatchInspectionResponseSchema,
    DefectSchema,
    AnomalySchema,
    QualityMetricsSchema,
)

__all__ = [
    "create_app",
    "api_router",
    "InspectionRequestSchema",
    "InspectionResponseSchema",
    "BatchInspectionRequestSchema",
    "BatchInspectionResponseSchema",
    "DefectSchema",
    "AnomalySchema",
    "QualityMetricsSchema",
]



