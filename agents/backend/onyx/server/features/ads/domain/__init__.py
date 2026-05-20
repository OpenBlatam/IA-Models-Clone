"""
🎯 ADS Domain Layer - Core Business Logic

This module contains the core business entities, repositories, and services
for the advertising system, following Clean Architecture principles.
"""

from .entities import Ad, AdCampaign, AdGroup, AdPerformance
from .repositories import AdRepository, CampaignRepository, PerformanceRepository
from .services import AdService, CampaignService, OptimizationService
from .value_objects import AdStatus, AdType, TargetingCriteria, Budget

__all__ = [
    # Entities
    'Ad', 'AdCampaign', 'AdGroup', 'AdPerformance',
    
    # Repositories
    'AdRepository', 'CampaignRepository', 'PerformanceRepository',
    
    # Services
    'AdService', 'CampaignService', 'OptimizationService',
    
    # Value Objects
    'AdStatus', 'AdType', 'TargetingCriteria', 'Budget'
]
