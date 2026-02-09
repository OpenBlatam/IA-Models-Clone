"""
Inspect Image Use Case

Use case for inspecting a single image.
"""

import time
import logging
from typing import Optional
import numpy as np

from ...domain import (
    InspectionService,
    Inspection,
    ImageMetadata,
    Defect,
    Anomaly,
)
from ...domain.exceptions import InspectionException, InvalidImageException
from ...domain.validators import ImageValidator, InspectionValidator
from ..dto import InspectionRequest, InspectionResponse, DefectDTO, AnomalyDTO

logger = logging.getLogger(__name__)


class InspectImageUseCase:
    """
    Use case for inspecting a single image.
    
    This use case:
    1. Validates the input image
    2. Creates an inspection
    3. Detects defects and anomalies (via infrastructure adapters)
    4. Calculates quality score
    5. Returns inspection response
    """
    
    def __init__(
        self,
        inspection_service: InspectionService,
        # These would be injected from infrastructure layer
        defect_detector=None,  # Will be injected
        anomaly_detector=None,  # Will be injected
        image_processor=None,  # Will be injected
    ):
        """
        Initialize use case.
        
        Args:
            inspection_service: Domain service for inspections
            defect_detector: Infrastructure adapter for defect detection
            anomaly_detector: Infrastructure adapter for anomaly detection
            image_processor: Infrastructure adapter for image processing
        """
        self.inspection_service = inspection_service
        self.defect_detector = defect_detector
        self.anomaly_detector = anomaly_detector
        self.image_processor = image_processor
        self.image_validator = ImageValidator()
        self.inspection_validator = InspectionValidator()
    
    def execute(self, request: InspectionRequest) -> InspectionResponse:
        """
        Execute the inspection use case.
        
        Args:
            request: Inspection request
        
        Returns:
            Inspection response
        
        Raises:
            InspectionException: If inspection fails
            InvalidImageException: If image is invalid
        """
        start_time = time.time()
        
        try:
            # Track metrics
            from ...infrastructure.metrics import get_metrics_collector
            metrics = get_metrics_collector()
            metrics.increment("inspections.total")
            
            # Validate request
            is_valid, error = self.inspection_validator.validate_inspection_request(
                request.image_data,
                request.image_format
            )
            if not is_valid:
                raise InvalidImageException(error or "Invalid inspection request")
            
            # 1. Load and validate image
            image, image_metadata = self._load_and_validate_image(request)
            
            # Validate loaded image
            is_valid, error = self.image_validator.validate_image(image)
            if not is_valid:
                raise InvalidImageException(error or "Invalid image data")
            
            # 2. Create inspection
            inspection = self.inspection_service.create_inspection(image_metadata)
            
            # 3. Detect defects (if detector available)
            if self.defect_detector:
                defects = self._detect_defects(image, inspection)
                for defect in defects:
                    inspection = self.inspection_service.add_defect_to_inspection(
                        inspection, defect
                    )
            
            # 4. Detect anomalies (if detector available)
            if self.anomaly_detector:
                anomalies = self._detect_anomalies(image, inspection)
                for anomaly in anomalies:
                    inspection = self.inspection_service.add_anomaly_to_inspection(
                        inspection, anomaly
                    )
            
            # 5. Complete inspection
            inspection = self.inspection_service.complete_inspection(inspection)
            
            # Validate completed inspection
            is_valid, error = self.inspection_validator.validate_inspection(inspection)
            if not is_valid:
                logger.warning(f"Inspection validation warning: {error}")
            
            # 6. Calculate processing time
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Record timing metrics
            metrics.record_timing("inspections.duration_ms", inference_time_ms)
            metrics.increment("inspections.successful")
            
            # 7. Generate visualization if requested
            visualization = None
            if request.include_visualization:
                visualization = self._generate_visualization(image, inspection)
            
            # 8. Convert to response DTO
            response = self._to_response(inspection, inference_time_ms, visualization)
            
            logger.info(
                f"Inspection {inspection.id} completed: "
                f"score={inspection.quality_score.score:.2f}, "
                f"defects={inspection.total_defects}, "
                f"anomalies={inspection.total_anomalies}"
            )
            
            return response
        
        except Exception as e:
            # Record error metrics
            from ...infrastructure.metrics import get_metrics_collector
            metrics = get_metrics_collector()
            metrics.increment("inspections.failed")
            metrics.record_error(
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            logger.error(f"Inspection failed: {str(e)}", exc_info=True)
            if isinstance(e, (InspectionException, InvalidImageException)):
                raise
            raise InspectionException(f"Inspection failed: {str(e)}")
    
    def _load_and_validate_image(
        self, request: InspectionRequest
    ) -> tuple[np.ndarray, ImageMetadata]:
        """
        Load and validate image from request.
        
        Args:
            request: Inspection request
        
        Returns:
            Tuple of (image array, image metadata)
        
        Raises:
            InvalidImageException: If image is invalid
        """
        # Use image processor if available
        if self.image_processor:
            try:
                image, img_metadata = self.image_processor.load_image(
                    request.image_data,
                    request.image_format
                )
                
                # Create domain ImageMetadata
                metadata = ImageMetadata(
                    width=img_metadata["width"],
                    height=img_metadata["height"],
                    channels=img_metadata["channels"],
                    source="inspection_request",
                )
                
                return image, metadata
            except Exception as e:
                raise InvalidImageException(str(e))
        
        # Fallback: basic implementation
        if request.image_format == "numpy":
            if not isinstance(request.image_data, np.ndarray):
                raise InvalidImageException("Invalid numpy array")
            image = request.image_data
        elif request.image_format in ["bytes", "file_path", "base64"]:
            raise InvalidImageException(
                f"Image processor required for format: {request.image_format}"
            )
        else:
            raise InvalidImageException(f"Unsupported image format: {request.image_format}")
        
        # Create image metadata
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        metadata = ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            source="inspection_request",
        )
        
        return image, metadata
    
    def _detect_defects(
        self, image: np.ndarray, inspection: Inspection
    ) -> list[Defect]:
        """
        Detect defects in image.
        
        Args:
            image: Image array
            inspection: Current inspection
        
        Returns:
            List of detected defects
        """
        if not self.defect_detector:
            logger.warning("Defect detector not available")
            return []
        
        try:
            # First detect objects/regions that might contain defects
            if hasattr(self.defect_detector, 'detect_objects'):
                objects = self.defect_detector.detect_objects(image)
            else:
                # Fallback: use full image as region
                h, w = image.shape[:2]
                objects = [{"bbox": [0, 0, w, h]}]
            
            # Classify defects in detected regions
            if hasattr(self.defect_detector, 'classify_defects'):
                defects = self.defect_detector.classify_defects(image, objects)
            else:
                defects = []
            
            return defects
        
        except Exception as e:
            logger.error(f"Defect detection failed: {str(e)}", exc_info=True)
            return []
    
    def _detect_anomalies(
        self, image: np.ndarray, inspection: Inspection
    ) -> list[Anomaly]:
        """
        Detect anomalies in image.
        
        Args:
            image: Image array
            inspection: Current inspection
        
        Returns:
            List of detected anomalies
        """
        if not self.anomaly_detector:
            logger.warning("Anomaly detector not available")
            return []
        
        try:
            # Use anomaly detection service
            if hasattr(self.anomaly_detector, 'detect_anomalies'):
                anomalies = self.anomaly_detector.detect_anomalies(image)
            else:
                anomalies = []
            
            return anomalies
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}", exc_info=True)
            return []
    
    def _generate_visualization(
        self, image: np.ndarray, inspection: Inspection
    ) -> Optional[any]:
        """
        Generate visualization of inspection results.
        
        Args:
            image: Original image
            inspection: Inspection results
        
        Returns:
            Visualization (bytes or base64 string)
        """
        # This would use visualization service
        # Placeholder
        return None
    
    def _to_response(
        self,
        inspection: Inspection,
        inference_time_ms: float,
        visualization: Optional[any] = None
    ) -> InspectionResponse:
        """
        Convert domain inspection to response DTO.
        
        Args:
            inspection: Domain inspection entity
            inference_time_ms: Inference time in milliseconds
            visualization: Optional visualization
        
        Returns:
            Inspection response DTO
        """
        return InspectionResponse(
            inspection_id=inspection.id,
            quality_score=inspection.quality_score.score,
            quality_status=inspection.quality_score.status.value,
            defects=[DefectDTO.from_domain_entity(d) for d in inspection.defects],
            anomalies=[AnomalyDTO.from_domain_entity(a) for a in inspection.anomalies],
            is_acceptable=inspection.quality_score.is_acceptable,
            recommendation=inspection.quality_score.recommendation,
            inference_time_ms=inference_time_ms,
            visualization=visualization,
            metadata={
                "total_defects": inspection.total_defects,
                "total_anomalies": inspection.total_anomalies,
            },
            created_at=inspection.created_at,
        )

