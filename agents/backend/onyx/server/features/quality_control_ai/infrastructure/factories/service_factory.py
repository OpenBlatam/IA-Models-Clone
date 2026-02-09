"""
Service Factory

Factory for creating and configuring service instances.
"""

import logging
from typing import Optional

from ...domain.services import (
    InspectionService,
    QualityAssessmentService,
    DefectClassificationService,
)
from ..repositories import InspectionRepository, ModelRepository, ConfigurationRepository
from ..adapters import CameraAdapter, MLModelLoader, StorageAdapter
from ..ml_services import (
    AnomalyDetectionService,
    ObjectDetectionService,
    DefectClassificationService as MLDefectClassificationService,
)
from ..image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory for creating service instances with proper dependency injection.
    
    This factory ensures all dependencies are properly wired together.
    """
    
    def __init__(
        self,
        inspection_repository: Optional[InspectionRepository] = None,
        model_repository: Optional[ModelRepository] = None,
        config_repository: Optional[ConfigurationRepository] = None,
        camera_adapter: Optional[CameraAdapter] = None,
        model_loader: Optional[MLModelLoader] = None,
        storage_adapter: Optional[StorageAdapter] = None,
    ):
        """
        Initialize service factory.
        
        Args:
            inspection_repository: Repository for inspections
            model_repository: Repository for models
            config_repository: Repository for configuration
            camera_adapter: Camera adapter
            model_loader: ML model loader
            storage_adapter: Storage adapter
        """
        self.inspection_repository = inspection_repository or InspectionRepository()
        self.model_repository = model_repository or ModelRepository()
        self.config_repository = config_repository or ConfigurationRepository()
        self.camera_adapter = camera_adapter or CameraAdapter()
        self.model_loader = model_loader or MLModelLoader()
        self.storage_adapter = storage_adapter or StorageAdapter()
    
    def create_inspection_service(self) -> InspectionService:
        """
        Create inspection domain service.
        
        Returns:
            InspectionService instance
        """
        quality_service = QualityAssessmentService()
        defect_service = DefectClassificationService()
        
        return InspectionService(
            quality_assessment_service=quality_service,
            defect_classification_service=defect_service,
        )
    
    def create_anomaly_detection_service(
        self,
        use_autoencoder: bool = True,
        use_diffusion: bool = False,
    ) -> AnomalyDetectionService:
        """
        Create anomaly detection service.
        
        Args:
            use_autoencoder: Whether to use autoencoder
            use_diffusion: Whether to use diffusion model
        
        Returns:
            AnomalyDetectionService instance
        """
        return AnomalyDetectionService(
            model_loader=self.model_loader,
            use_autoencoder=use_autoencoder,
            use_diffusion=use_diffusion,
        )
    
    def create_object_detection_service(
        self,
        model_type: str = "yolov8",
        confidence_threshold: float = 0.5,
    ) -> ObjectDetectionService:
        """
        Create object detection service.
        
        Args:
            model_type: Type of detection model
            confidence_threshold: Confidence threshold
        
        Returns:
            ObjectDetectionService instance
        """
        return ObjectDetectionService(
            model_loader=self.model_loader,
            model_type=model_type,
            confidence_threshold=confidence_threshold,
        )
    
    def create_defect_classification_service(
        self,
        model_type: str = "vit",
        confidence_threshold: float = 0.5,
    ) -> MLDefectClassificationService:
        """
        Create defect classification service.
        
        Args:
            model_type: Type of model
            confidence_threshold: Confidence threshold
        
        Returns:
            DefectClassificationService instance
        """
        return MLDefectClassificationService(
            model_loader=self.model_loader,
            model_type=model_type,
            confidence_threshold=confidence_threshold,
        )
    
    def get_camera_adapter(self) -> CameraAdapter:
        """Get camera adapter."""
        return self.camera_adapter
    
    def get_model_loader(self) -> MLModelLoader:
        """Get model loader."""
        return self.model_loader
    
    def get_storage_adapter(self) -> StorageAdapter:
        """Get storage adapter."""
        return self.storage_adapter
    
    def create_image_processor(self) -> ImageProcessor:
        """
        Create image processor.
        
        Returns:
            ImageProcessor instance
        """
        return ImageProcessor()

