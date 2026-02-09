"""
🎯 ADS Application Layer - Use Cases and Application Services

This module contains the application layer logic, including use cases
that orchestrate domain services and coordinate business operations.
"""

from .use_cases import (
    CreateAdUseCase,
    ApproveAdUseCase,
    ActivateAdUseCase,
    PauseAdUseCase,
    ArchiveAdUseCase,
    CreateCampaignUseCase,
    ActivateCampaignUseCase,
    PauseCampaignUseCase,
    OptimizeAdUseCase,
    PredictPerformanceUseCase
)
from .dto import (
    CreateAdRequest,
    CreateAdResponse,
    ApproveAdRequest,
    ApproveAdResponse,
    ActivateAdRequest,
    ActivateAdResponse,
    PauseAdRequest,
    PauseAdResponse,
    ArchiveAdRequest,
    ArchiveAdResponse,
    CreateCampaignRequest,
    CreateCampaignResponse,
    ActivateCampaignResponse,
    PauseCampaignResponse,
    ActivateCampaignRequest,
    PauseCampaignRequest,
    OptimizationRequest,
    OptimizationResponse,
    PerformancePredictionRequest,
    PerformancePredictionResponse
)

__all__ = [
    # Use Cases
    'CreateAdUseCase',
    'ApproveAdUseCase',
    'ActivateAdUseCase',
    'PauseAdUseCase',
    'ArchiveAdUseCase',
    'CreateCampaignUseCase',
    'ActivateCampaignUseCase',
    'PauseCampaignUseCase',
    'OptimizeAdUseCase',
    'PredictPerformanceUseCase',
    
    # DTOs
    'CreateAdRequest',
    'CreateAdResponse',
    'ApproveAdRequest',
    'ApproveAdResponse',
    'ActivateAdRequest',
    'ActivateAdResponse',
    'PauseAdRequest',
    'PauseAdResponse',
    'ArchiveAdRequest',
    'ArchiveAdResponse',
    'CreateCampaignRequest',
    'CreateCampaignResponse',
    'ActivateCampaignResponse',
    'PauseCampaignResponse',
    'ActivateCampaignRequest',
    'PauseCampaignRequest',
    'OptimizationRequest',
    'OptimizationResponse',
    'PerformancePredictionRequest',
    'PerformancePredictionResponse'
]
