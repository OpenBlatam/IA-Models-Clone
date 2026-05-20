"""
Application Layer
================

This module contains the application layer components that orchestrate
business logic and coordinate between the domain and infrastructure layers.

The application layer includes:
- Use cases that implement business workflows
- Application services that coordinate domain services
- DTOs (Data Transfer Objects) for API communication
- Command and query handlers
- Application event handlers
"""

from .use_cases import (
    AnalyzeContentUseCase,
    CompareModelsUseCase,
    GenerateReportUseCase,
    TrackTrendsUseCase,
    ManageAnalysisJobUseCase
)

from .dto import (
    AnalyzeContentRequest,
    AnalyzeContentResponse,
    CompareModelsRequest,
    CompareModelsResponse,
    GenerateReportRequest,
    GenerateReportResponse,
    TrackTrendsRequest,
    TrackTrendsResponse
)

from .services import (
    ApplicationService,
    EventHandler,
    NotificationService
)

__all__ = [
    # Use Cases
    "AnalyzeContentUseCase",
    "CompareModelsUseCase",
    "GenerateReportUseCase",
    "TrackTrendsUseCase",
    "ManageAnalysisJobUseCase",
    
    # DTOs
    "AnalyzeContentRequest",
    "AnalyzeContentResponse",
    "CompareModelsRequest",
    "CompareModelsResponse",
    "GenerateReportRequest",
    "GenerateReportResponse",
    "TrackTrendsRequest",
    "TrackTrendsResponse",
    
    # Services
    "ApplicationService",
    "EventHandler",
    "NotificationService"
]




