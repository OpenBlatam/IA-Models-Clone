"""
Use case factory for creating use cases with resolved dependencies
Extracted from composition_root.py to reduce duplication
"""

from typing import Dict, Any, Callable, List, Optional
import logging
import asyncio

from .service_factory import ServiceFactory
from .application import (
    AnalyzeImageUseCase,
    GetRecommendationsUseCase,
    GetAnalysisHistoryUseCase,
)

logger = logging.getLogger(__name__)


class UseCaseFactory:
    """Factory for creating use cases with dependency resolution"""
    
    @staticmethod
    async def create_analyze_image_use_case(
        service_factory: ServiceFactory
    ) -> AnalyzeImageUseCase:
        """Create analyze image use case"""
        dependencies = await asyncio.gather(
            service_factory.create("analysis_repository"),
            service_factory.create("image_processor"),
            service_factory.create("event_publisher")
        )
        
        analysis_repo, image_processor, event_publisher = dependencies
        
        analysis_service = None
        if service_factory.is_registered("analysis_service"):
            analysis_service = await service_factory.create("analysis_service")
        
        return AnalyzeImageUseCase(
            analysis_repository=analysis_repo,
            image_processor=image_processor,
            analysis_service=analysis_service,
            event_publisher=event_publisher
        )
    
    @staticmethod
    async def create_recommendations_use_case(
        service_factory: ServiceFactory
    ) -> GetRecommendationsUseCase:
        """Create recommendations use case"""
        dependencies = await asyncio.gather(
            service_factory.create("analysis_repository"),
            service_factory.create("user_repository")
        )
        
        analysis_repo, user_repo = dependencies
        
        recommendation_service = None
        if service_factory.is_registered("recommendation_service"):
            recommendation_service = await service_factory.create("recommendation_service")
        
        return GetRecommendationsUseCase(
            analysis_repository=analysis_repo,
            recommendation_service=recommendation_service,
            user_repository=user_repo
        )
    
    @staticmethod
    async def create_history_use_case(
        service_factory: ServiceFactory
    ) -> GetAnalysisHistoryUseCase:
        """Create history use case"""
        analysis_repo = await service_factory.create("analysis_repository")
        return GetAnalysisHistoryUseCase(analysis_repository=analysis_repo)




