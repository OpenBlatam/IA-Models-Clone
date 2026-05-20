"""
Use case for analyzing skin images.

This use case orchestrates the complete flow of analyzing a skin image:
validation, processing, analysis, persistence, and event publishing.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from ..domain.entities import Analysis, AnalysisStatus
from ..domain.interfaces import (
    IAnalysisRepository,
    IAnalysisService,
    IImageProcessor,
    IEventPublisher,
)
from ..domain.exceptions import InvalidImageError
from .base import UseCase
from .exceptions import ValidationError, ProcessingError
from .validators import ImageValidator, UserIdValidator, MetadataValidator
from ...infrastructure.logging_utils import StructuredLogger
from ...infrastructure.performance_monitor import monitor_performance
from ...infrastructure.security_utils import InputSanitizer

logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(__name__)


class AnalyzeImageUseCase(UseCase):
    """
    Use case for analyzing skin images.
    
    Orchestrates the complete analysis workflow including validation,
    processing, domain service invocation, persistence, and event publishing.
    """
    
    def __init__(
        self,
        analysis_repository: IAnalysisRepository,
        image_processor: IImageProcessor,
        analysis_service: IAnalysisService,
        event_publisher: Optional[IEventPublisher] = None
    ) -> None:
        """
        Initialize analyze image use case.
        
        Args:
            analysis_repository: Repository for persisting analysis data
            image_processor: Service for processing and validating images
            analysis_service: Domain service for analysis business logic
            event_publisher: Optional event publisher for domain events
        """
        self.analysis_repository = analysis_repository
        self.image_processor = image_processor
        self.analysis_service = analysis_service
        self.event_publisher = event_publisher
    
    @monitor_performance(operation_name="analyze_image", log_threshold=5.0)
    async def execute(
        self,
        user_id: str,
        image_data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Analysis:
        """
        Execute image analysis use case.
        
        Performs complete analysis workflow:
        1. Sanitize and validate inputs
        2. Create initial analysis record
        3. Process image and perform analysis
        4. Update analysis with results
        5. Publish domain events
        
        Args:
            user_id: Identifier of the user requesting analysis
            image_data: Raw image bytes to analyze
            metadata: Optional metadata dictionary (filename, etc.)
            
        Returns:
            Completed Analysis entity with metrics and conditions
            
        Raises:
            ValidationError: If input validation fails
            ProcessingError: If analysis processing fails
        """
        # Sanitize and validate inputs
        user_id = InputSanitizer.sanitize_user_input(user_id, "string")
        if metadata:
            metadata = {
                k: InputSanitizer.sanitize_user_input(v, "string")
                for k, v in metadata.items()
            }
        
        # Validate inputs using centralized validators
        UserIdValidator.validate_user_id(user_id)
        ImageValidator.validate_image_data(image_data)
        MetadataValidator.validate_metadata(metadata)
        
        structured_logger.set_context(user_id=user_id, image_size=len(image_data))
        
        with structured_logger.operation("analyze_image", user_id=user_id):
            if not await self.image_processor.validate(image_data):
                raise ValidationError("Invalid image format or corrupted image data")
            
            if not self.analysis_service:
                raise ProcessingError("Analysis service not available")
            
            analysis = Analysis(
                id=str(uuid.uuid4()),
                user_id=user_id,
                status=AnalysisStatus.PROCESSING,
                metadata=metadata or {}
            )
            
            try:
                analysis = await self.analysis_repository.create(analysis)
            except Exception as e:
                raise ProcessingError(f"Failed to create analysis: {e}") from e
            
            if self.event_publisher:
                asyncio.create_task(
                    self.event_publisher.publish(
                        "analysis.started",
                        {"analysis_id": analysis.id, "user_id": user_id}
                    )
                )
            
            try:
                result = await self.analysis_service.analyze_image(
                    user_id,
                    image_data,
                    metadata
                )
                
                analysis.metrics = result.metrics
                analysis.conditions = result.conditions
                analysis.skin_type = result.skin_type
                analysis.status = AnalysisStatus.COMPLETED
                analysis.completed_at = datetime.utcnow()
                
                tasks = [self.analysis_repository.update(analysis)]
                if self.event_publisher:
                    tasks.append(
                        self.event_publisher.publish(
                            "analysis.completed",
                            {"analysis_id": analysis.id, "user_id": user_id}
                        )
                    )
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                updated_analysis = results[0]
                if isinstance(updated_analysis, Exception):
                    logger.warning(f"Failed to update analysis: {updated_analysis}")
                else:
                    analysis = updated_analysis
                
                return analysis
                
            except (ValidationError, InvalidImageError) as e:
                await self._handle_analysis_failure(analysis, e, "validation_error")
                raise ValidationError(str(e)) if isinstance(e, InvalidImageError) else e
                
            except Exception as e:
                await self._handle_analysis_failure(analysis, e, "processing_error")
                raise ProcessingError(f"Analysis failed: {e}") from e
    
    async def _handle_analysis_failure(
        self,
        analysis: Analysis,
        error: Exception,
        error_type: str
    ) -> None:
        """
        Handle analysis failure by updating status and publishing events.
        
        Args:
            analysis: Analysis entity to mark as failed
            error: Exception that caused the failure
            error_type: Type of error (validation_error, processing_error, etc.)
        """
        analysis.mark_failed()
        
        tasks = [self.analysis_repository.update(analysis)]
        if self.event_publisher:
            tasks.append(
                self.event_publisher.publish(
                    "analysis.failed",
                    {
                        "analysis_id": analysis.id,
                        "error": str(error),
                        "type": error_type
                    }
                )
            )
        
        await asyncio.gather(*tasks, return_exceptions=True)

