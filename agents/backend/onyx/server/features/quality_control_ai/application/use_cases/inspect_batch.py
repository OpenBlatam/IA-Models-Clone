"""
Inspect Batch Use Case

Use case for inspecting multiple images in batch.
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from ...domain.exceptions import InspectionException
from ..dto import (
    BatchInspectionRequest,
    BatchInspectionResponse,
    InspectionRequest,
    InspectionResponse,
    QualityMetricsDTO,
)
from .inspect_image import InspectImageUseCase

logger = logging.getLogger(__name__)


class InspectBatchUseCase:
    """
    Use case for inspecting multiple images in batch.
    
    This use case:
    1. Processes multiple images in parallel or sequentially
    2. Aggregates results
    3. Calculates batch metrics
    4. Returns batch response
    """
    
    def __init__(self, inspect_image_use_case: InspectImageUseCase):
        """
        Initialize batch inspection use case.
        
        Args:
            inspect_image_use_case: Use case for single image inspection
        """
        self.inspect_image_use_case = inspect_image_use_case
    
    def execute(
        self, request: BatchInspectionRequest
    ) -> BatchInspectionResponse:
        """
        Execute batch inspection use case.
        
        Args:
            request: Batch inspection request
        
        Returns:
            Batch inspection response
        
        Raises:
            InspectionException: If batch inspection fails
        """
        start_time = time.time()
        
        try:
            # Process images
            if request.parallel:
                inspections = self._process_parallel(request)
            else:
                inspections = self._process_sequential(request)
            
            # Calculate metrics
            total_processed = len(request.images)
            total_succeeded = len([i for i in inspections if i is not None])
            total_failed = total_processed - total_succeeded
            
            successful_inspections = [i for i in inspections if i is not None]
            
            if successful_inspections:
                average_quality_score = sum(
                    i.quality_score for i in successful_inspections
                ) / len(successful_inspections)
            else:
                average_quality_score = 0.0
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(successful_inspections)
            
            # Calculate total processing time
            total_processing_time_ms = (time.time() - start_time) * 1000
            
            response = BatchInspectionResponse(
                inspections=successful_inspections,
                total_processed=total_processed,
                total_succeeded=total_succeeded,
                total_failed=total_failed,
                average_quality_score=average_quality_score,
                total_processing_time_ms=total_processing_time_ms,
                quality_metrics=quality_metrics,
            )
            
            logger.info(
                f"Batch inspection completed: "
                f"processed={total_processed}, "
                f"succeeded={total_succeeded}, "
                f"failed={total_failed}, "
                f"avg_score={average_quality_score:.2f}"
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Batch inspection failed: {str(e)}", exc_info=True)
            raise InspectionException(f"Batch inspection failed: {str(e)}")
    
    def _process_parallel(
        self, request: BatchInspectionRequest
    ) -> List[InspectionResponse]:
        """
        Process images in parallel.
        
        Args:
            request: Batch inspection request
        
        Returns:
            List of inspection responses
        """
        max_workers = request.max_workers or 4
        batch_size = request.batch_size or len(request.images)
        
        inspections = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_request = {
                executor.submit(
                    self._inspect_single, img_request
                ): img_request
                for img_request in request.images
            }
            
            # Collect results
            for future in as_completed(future_to_request):
                try:
                    result = future.result()
                    inspections.append(result)
                except Exception as e:
                    logger.error(f"Failed to process image: {str(e)}")
                    inspections.append(None)
        
        return inspections
    
    def _process_sequential(
        self, request: BatchInspectionRequest
    ) -> List[InspectionResponse]:
        """
        Process images sequentially.
        
        Args:
            request: Batch inspection request
        
        Returns:
            List of inspection responses
        """
        inspections = []
        
        for img_request in request.images:
            try:
                result = self._inspect_single(img_request)
                inspections.append(result)
            except Exception as e:
                logger.error(f"Failed to process image: {str(e)}")
                inspections.append(None)
        
        return inspections
    
    def _inspect_single(
        self, request: InspectionRequest
    ) -> InspectionResponse:
        """
        Inspect a single image.
        
        Args:
            request: Single image inspection request
        
        Returns:
            Inspection response
        """
        return self.inspect_image_use_case.execute(request)
    
    def _calculate_quality_metrics(
        self, inspections: List[InspectionResponse]
    ) -> QualityMetricsDTO:
        """
        Calculate aggregated quality metrics.
        
        Args:
            inspections: List of successful inspections
        
        Returns:
            Quality metrics DTO
        """
        if not inspections:
            return QualityMetricsDTO(
                total_inspections=0,
                average_quality_score=0.0,
            )
        
        total_inspections = len(inspections)
        average_quality_score = sum(i.quality_score for i in inspections) / total_inspections
        
        total_defects = sum(len(i.defects) for i in inspections)
        total_anomalies = sum(len(i.anomalies) for i in inspections)
        
        rejected_count = sum(
            1 for i in inspections if not i.is_acceptable
        )
        rejection_rate = (rejected_count / total_inspections) * 100.0 if total_inspections > 0 else 0.0
        
        return QualityMetricsDTO(
            total_inspections=total_inspections,
            average_quality_score=average_quality_score,
            defects_count=total_defects,
            anomalies_count=total_anomalies,
            rejection_rate=rejection_rate,
            acceptance_rate=100.0 - rejection_rate,
            defects_per_inspection=total_defects / total_inspections if total_inspections > 0 else 0.0,
            anomalies_per_inspection=total_anomalies / total_inspections if total_inspections > 0 else 0.0,
        )



