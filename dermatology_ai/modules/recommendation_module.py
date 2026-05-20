"""
Recommendation Module
Example module with dependencies
"""

from typing import Dict, Any, Optional
from core.modules.module import Module, ModuleMetadata
import logging

logger = logging.getLogger(__name__)


class RecommendationModule(Module):
    """Module for recommendation functionality"""
    
    metadata = ModuleMetadata(
        name="recommendation",
        version="1.0.0",
        description="Product recommendation module",
        dependencies=["analysis"],  # Depends on analysis module
        provides=["recommendation_service"],
        requires=["analysis_service"],  # Requires service from analysis module
        tags=["recommendation", "core"]
    )
    
    def __init__(self, metadata: ModuleMetadata = None):
        super().__init__(metadata or self.metadata)
        self.recommendation_service = None
        self.analysis_service = None
    
    async def load(self) -> bool:
        """Load recommendation module"""
        try:
            # Get dependency service
            analysis_module = self.get_dependency("analysis")
            if analysis_module:
                self.analysis_service = analysis_module.get_service("analysis_service")
            
            # Import and create service
            from services.recommendation_service import RecommendationService
            
            self.recommendation_service = RecommendationService(
                analysis_service=self.analysis_service
            )
            
            # Register service
            self.provide_service("recommendation_service", self.recommendation_service)
            
            logger.info("Recommendation module loaded")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load recommendation module: {e}", exc_info=True)
            return False
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize recommendation module"""
        try:
            if config:
                self.config = config
            
            if self.recommendation_service and hasattr(self.recommendation_service, 'initialize'):
                await self.recommendation_service.initialize()
            
            logger.info("Recommendation module initialized")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize recommendation module: {e}", exc_info=True)
            return False
    
    async def start(self) -> bool:
        """Start recommendation module"""
        logger.info("Recommendation module started")
        return True
    
    async def stop(self) -> bool:
        """Stop recommendation module"""
        logger.info("Recommendation module stopped")
        return True
    
    async def unload(self) -> bool:
        """Unload recommendation module"""
        self.recommendation_service = None
        self.analysis_service = None
        self.provided_services.clear()
        logger.info("Recommendation module unloaded")
        return True















