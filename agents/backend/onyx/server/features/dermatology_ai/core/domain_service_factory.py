"""
Domain service factory for creating domain services.

Extracted from composition_root.py for better organization and separation
of concerns. Handles creation of domain services with proper dependency
injection.
"""

import logging
from typing import Optional

from .service_factory import ServiceFactory
from .domain.interfaces import IAnalysisService, IRecommendationService

logger = logging.getLogger(__name__)


class DomainServiceFactory:
    """
    Factory for creating domain services.
    
    Provides static methods to create domain services with their
    dependencies resolved from the service factory.
    """
    
    @staticmethod
    async def create_analysis_service(
        service_factory: ServiceFactory
    ) -> Optional[IAnalysisService]:
        """
        Create analysis service with dependencies resolved.
        
        Args:
            service_factory: Service factory for dependency resolution
            
        Returns:
            Configured AnalysisService instance or None if creation fails
        """
        try:
            from .domain.services import AnalysisService
            image_processor = await service_factory.create("image_processor")
            
            ml_model_manager = None
            try:
                from .ml_model_manager import MLModelManager
                ml_model_manager = MLModelManager()
            except Exception:
                logger.debug("ML model manager not available, continuing without it")
            
            return AnalysisService(
                image_processor=image_processor,
                ml_model_manager=ml_model_manager
            )
        except Exception as e:
            logger.debug(f"Analysis service not available: {e}")
            return None
    
    @staticmethod
    async def create_recommendation_service(
        service_factory: ServiceFactory
    ) -> Optional[IRecommendationService]:
        """
        Create recommendation service with dependencies resolved.
        
        Args:
            service_factory: Service factory for dependency resolution
            
        Returns:
            Configured RecommendationService instance or None if creation fails
        """
        try:
            from .domain.services import RecommendationService
            product_repo = await service_factory.create("product_repository")
            return RecommendationService(product_repository=product_repo)
        except Exception as e:
            logger.debug(f"Recommendation service not available: {e}")
            return None



