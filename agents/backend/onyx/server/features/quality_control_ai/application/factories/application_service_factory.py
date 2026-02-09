"""
Application Service Factory

Factory for creating application service instances with all dependencies.
"""

import logging

from ..services import (
    InspectionApplicationService,
    ModelTrainingApplicationService,
)
from ...infrastructure.factories import UseCaseFactory, ServiceFactory

logger = logging.getLogger(__name__)


class ApplicationServiceFactory:
    """
    Factory for creating application service instances.
    
    This is the main entry point for creating all application services.
    """
    
    def __init__(self, service_factory: ServiceFactory = None):
        """
        Initialize application service factory.
        
        Args:
            service_factory: Optional service factory (creates default if None)
        """
        self.service_factory = service_factory or ServiceFactory()
        self.use_case_factory = UseCaseFactory(self.service_factory)
    
    def create_inspection_application_service(self) -> InspectionApplicationService:
        """
        Create inspection application service with all dependencies.
        
        Returns:
            InspectionApplicationService instance
        """
        return InspectionApplicationService(
            inspect_image_use_case=self.use_case_factory.create_inspect_image_use_case(),
            inspect_batch_use_case=self.use_case_factory.create_inspect_batch_use_case(),
            start_stream_use_case=self.use_case_factory.create_start_inspection_stream_use_case(),
            stop_stream_use_case=self.use_case_factory.create_stop_inspection_stream_use_case(),
            generate_report_use_case=self.use_case_factory.create_generate_report_use_case(),
        )
    
    def create_model_training_application_service(self) -> ModelTrainingApplicationService:
        """
        Create model training application service.
        
        Returns:
            ModelTrainingApplicationService instance
        """
        return ModelTrainingApplicationService(
            train_model_use_case=self.use_case_factory.create_train_model_use_case(),
        )



