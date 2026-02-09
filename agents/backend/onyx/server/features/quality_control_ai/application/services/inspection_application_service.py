"""
Inspection Application Service

Application service that orchestrates inspection-related use cases.
"""

import logging
from typing import Optional

from ..use_cases import (
    InspectImageUseCase,
    InspectBatchUseCase,
    StartInspectionStreamUseCase,
    StopInspectionStreamUseCase,
    GenerateReportUseCase,
)
from ..dto import (
    InspectionRequest,
    InspectionResponse,
    BatchInspectionRequest,
    BatchInspectionResponse,
)

logger = logging.getLogger(__name__)


class InspectionApplicationService:
    """
    Application service for inspection operations.
    
    This service coordinates multiple use cases and provides
    a high-level interface for inspection operations.
    """
    
    def __init__(
        self,
        inspect_image_use_case: InspectImageUseCase,
        inspect_batch_use_case: InspectBatchUseCase,
        start_stream_use_case: StartInspectionStreamUseCase,
        stop_stream_use_case: StopInspectionStreamUseCase,
        generate_report_use_case: GenerateReportUseCase,
    ):
        """
        Initialize application service.
        
        Args:
            inspect_image_use_case: Use case for single image inspection
            inspect_batch_use_case: Use case for batch inspection
            start_stream_use_case: Use case for starting stream
            stop_stream_use_case: Use case for stopping stream
            generate_report_use_case: Use case for generating reports
        """
        self.inspect_image_use_case = inspect_image_use_case
        self.inspect_batch_use_case = inspect_batch_use_case
        self.start_stream_use_case = start_stream_use_case
        self.stop_stream_use_case = stop_stream_use_case
        self.generate_report_use_case = generate_report_use_case
    
    def inspect_image(self, request: InspectionRequest) -> InspectionResponse:
        """
        Inspect a single image.
        
        Args:
            request: Inspection request
        
        Returns:
            Inspection response
        """
        return self.inspect_image_use_case.execute(request)
    
    def inspect_batch(
        self, request: BatchInspectionRequest
    ) -> BatchInspectionResponse:
        """
        Inspect multiple images in batch.
        
        Args:
            request: Batch inspection request
        
        Returns:
            Batch inspection response
        """
        return self.inspect_batch_use_case.execute(request)
    
    def start_inspection_stream(
        self,
        camera_index: int,
        resolution: Optional[tuple] = None,
        fps: Optional[int] = None,
    ) -> dict:
        """
        Start real-time inspection stream.
        
        Args:
            camera_index: Camera index
            resolution: Optional resolution
            fps: Optional frames per second
        
        Returns:
            Stream information
        """
        return self.start_stream_use_case.execute(camera_index, resolution, fps)
    
    def stop_inspection_stream(self, camera_index: int) -> dict:
        """
        Stop real-time inspection stream.
        
        Args:
            camera_index: Camera index
        
        Returns:
            Confirmation
        """
        return self.stop_stream_use_case.execute(camera_index)
    
    def generate_report(
        self,
        inspections: list[InspectionResponse],
        report_format: str = "json",
        **kwargs
    ) -> dict:
        """
        Generate inspection report.
        
        Args:
            inspections: List of inspection responses
            report_format: Report format
            **kwargs: Additional report options
        
        Returns:
            Report data
        """
        return self.generate_report_use_case.execute(
            inspections, report_format, **kwargs
        )



