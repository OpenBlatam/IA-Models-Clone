"""
Use Case Factory

Factory for creating use case instances with proper dependencies.
"""

import logging

from ...application.use_cases import (
    InspectImageUseCase,
    InspectBatchUseCase,
    StartInspectionStreamUseCase,
    StopInspectionStreamUseCase,
    TrainModelUseCase,
    GenerateReportUseCase,
)
from ...domain.services import InspectionService
from .service_factory import ServiceFactory

logger = logging.getLogger(__name__)


class UseCaseFactory:
    """
    Factory for creating use case instances.
    
    This factory wires use cases with their required dependencies.
    """
    
    def __init__(self, service_factory: ServiceFactory):
        """
        Initialize use case factory.
        
        Args:
            service_factory: Service factory for creating dependencies
        """
        self.service_factory = service_factory
        self._inspection_service = None
        self._anomaly_service = None
        self._object_detection_service = None
        self._defect_classification_service = None
    
    def create_inspect_image_use_case(self) -> InspectImageUseCase:
        """
        Create inspect image use case.
        
        Returns:
            InspectImageUseCase instance
        """
        if self._inspection_service is None:
            self._inspection_service = self.service_factory.create_inspection_service()
        
        if self._anomaly_service is None:
            self._anomaly_service = self.service_factory.create_anomaly_detection_service()
        
        if self._defect_classification_service is None:
            self._defect_classification_service = (
                self.service_factory.create_defect_classification_service()
            )
        
        # Get image processor
        image_processor = self.service_factory.create_image_processor()
        
        return InspectImageUseCase(
            inspection_service=self._inspection_service,
            defect_detector=self._defect_classification_service,
            anomaly_detector=self._anomaly_service,
            image_processor=image_processor,
        )
    
    def create_inspect_batch_use_case(self) -> InspectBatchUseCase:
        """
        Create inspect batch use case.
        
        Returns:
            InspectBatchUseCase instance
        """
        inspect_image_use_case = self.create_inspect_image_use_case()
        return InspectBatchUseCase(inspect_image_use_case)
    
    def create_start_inspection_stream_use_case(self) -> StartInspectionStreamUseCase:
        """
        Create start inspection stream use case.
        
        Returns:
            StartInspectionStreamUseCase instance
        """
        camera_adapter = self.service_factory.get_camera_adapter()
        return StartInspectionStreamUseCase(camera_adapter=camera_adapter)
    
    def create_stop_inspection_stream_use_case(self) -> StopInspectionStreamUseCase:
        """
        Create stop inspection stream use case.
        
        Returns:
            StopInspectionStreamUseCase instance
        """
        camera_adapter = self.service_factory.get_camera_adapter()
        return StopInspectionStreamUseCase(camera_adapter=camera_adapter)
    
    def create_train_model_use_case(self) -> TrainModelUseCase:
        """
        Create train model use case.
        
        Returns:
            TrainModelUseCase instance
        """
        from ...training import ModelTrainer
        from ..repositories import ModelRepository
        
        model_trainer = ModelTrainer()  # Would be configured properly
        model_repository = self.service_factory.model_repository
        
        return TrainModelUseCase(
            model_trainer=model_trainer,
            model_repository=model_repository,
        )
    
    def create_generate_report_use_case(self) -> GenerateReportUseCase:
        """
        Create generate report use case.
        
        Returns:
            GenerateReportUseCase instance
        """
        from ...utils import ReportGenerator
        
        report_generator = ReportGenerator()
        return GenerateReportUseCase(report_generator=report_generator)

